"""Module 3 prototype: peephole mutations gated by Module 1 + Module 2.
Код берется только из dis.insns (visited, рекурсивный спуск). Гейт живости
для каждой замены: добавленные defs не должны быть живы (live_after),
пропавшие defs не должны быть живы, лишние чтения запрещены кроме
вход-независимых (xor/sub r,r: результат = 0), identity-записи считаются
не-дефами только при размере операнда != 4 (32-битная запись обнуляет
старшую половину родителя), любое касание памяти сверх оригинала,
пропавшая запись памяти и изменение stack_delta запрещены.
"""
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone.x86_const import X86_OP_REG, X86_OP_IMM
from keystone import Ks, KS_ARCH_X86, KS_MODE_64

from mod1_disasm import disassemble, parent, verify, control_flags
from mod2_ir import lift, _effect, recompute_live_after_prefix, _liveness
from pipeline_config import PipelineConfig

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True
ks = Ks(KS_ARCH_X86, KS_MODE_64)


def asm(s):
    enc, _ = ks.asm(s)
    return bytes(enc)


NOP_TABLE = {
    1: b"\x90",
    2: b"\x66\x90",
    3: b"\x0f\x1f\x00",
    4: b"\x0f\x1f\x40\x00",
    5: b"\x0f\x1f\x44\x00\x00",
    6: b"\x66\x0f\x1f\x44\x00\x00",
    7: b"\x0f\x1f\x80\x00\x00\x00\x00",
    8: b"\x0f\x1f\x84\x00\x00\x00\x00\x00",
    9: b"\x66\x0f\x1f\x84\x00\x00\x00\x00\x00",
}

for _n, _b in NOP_TABLE.items():
    _d = next(md.disasm(_b, 0x1000), None)
    if _d is None or _d.mnemonic != "nop":
        raise ValueError(f"NOP_TABLE[{_n}] does not decode as nop: {_b.hex()}")


def pad(b, n):
    if len(b) > n:
        return None
    out = bytes(b)
    while len(out) < n:
        out += NOP_TABLE[min(9, n - len(out))]
    return out


def effect_of(raw, addr, imports):
    agg_reg_use, agg_reg_def = set(), set()
    agg_flag_use, agg_flag_def = set(), set()
    agg_mem_use, agg_mem_def = set(), set()
    agg_delta = 0
    covered = 0
    for det in md.disasm(raw, addr):
        covered += det.size
        if det.mnemonic == "nop":
            continue
        # CS_GRP_JUMP включает jcc — флаги только по mnemonic.
        cf = control_flags(det.mnemonic)
        fake = SimpleNamespace(
            mnemonic=det.mnemonic, addr=det.address,
            regs_read=tuple(md.reg_name(r) for r in det.regs_read),
            regs_write=tuple(md.reg_name(r) for r in det.regs_write),
            **cf)
        eff = _effect(fake, det, md, imports)
        agg_reg_use.update(eff.reg_use)
        agg_reg_def.update(eff.reg_def)
        agg_flag_use.update(eff.flag_use)
        agg_flag_def.update(eff.flag_def)
        agg_mem_use.update(eff.mem_use)
        agg_mem_def.update(eff.mem_def)
        if eff.stack_delta is None:
            agg_delta = None
        elif agg_delta is not None:
            agg_delta += eff.stack_delta
    if covered != len(raw):
        return None
    return SimpleNamespace(
        reg_use=frozenset(agg_reg_use), reg_def=frozenset(agg_reg_def),
        flag_use=frozenset(agg_flag_use), flag_def=frozenset(agg_flag_def),
        mem_use=frozenset(agg_mem_use), mem_def=frozenset(agg_mem_def),
        stack_delta=agg_delta)


def mutate(insn):
    ops = insn.operands
    if len(ops) != 2:
        return None
    o0, o1 = ops
    reg = o0.reg if o0.type == X86_OP_REG else None
    if insn.mnemonic == "sub" and reg is not None \
            and o1.type == X86_OP_REG and o1.reg == o0.reg and o0.size >= 4:
        return "sub2xor", f"xor {reg}, {reg}"
    if insn.mnemonic == "test" and reg is not None \
            and o1.type == X86_OP_REG and o1.reg == o0.reg:
        return "test2or", f"or {reg}, {reg}"
    if insn.mnemonic == "mov" and reg is not None \
            and o1.type == X86_OP_IMM and o1.imm == 0 and o0.size >= 4:
        return "mov02xor", f"xor {reg}, {reg}"
    return None


def mutate_pe(src: str, dst: str, cfg: PipelineConfig | None = None,
              dis=None, mod=None) -> dict:
    """Пройтись по инструкциям src, применить peephole-правила под гейтом
    liveness и записать мутированный PE в dst. Возвращает счётчики
    applied/blocked по правилам.

    После каждой принятой мутации live_after предыдущих инструкций блока
    пересчитывается: мутация меняет def/use, и гейт ниже по блоку (раньше
    по адресу) иначе смотрел бы на устаревшую живость оригинала.
    """
    cfg = PipelineConfig.from_legacy(cfg=cfg)
    if dis is None:
        dis = disassemble(src, cfg=cfg)
    elif getattr(dis, "split_on_call", cfg.split_on_call) != cfg.split_on_call:
        raise ValueError(
            f"DisasmResult.split_on_call={dis.split_on_call} "
            f"не совпадает с PipelineConfig.split_on_call={cfg.split_on_call}"
        )
    verrs, _vwarns = verify(dis)
    if verrs:
        raise RuntimeError(
            "mod1 verify failed, refusing to mutate:\n" + "\n".join(verrs)
        )
    if mod is None:
        mod = lift(src, cfg=cfg, dis=dis)

    data = bytearray(open(src, "rb").read())
    applied, blocked = {}, {}
    # С конца блока: live_after[i] зависит от эффектов i+1..end.
    # Мутация хвоста инвалидирует живость префикса — префикс ещё не трогали.
    for b in dis.blocks:
        for idx in range(len(b.insns) - 1, -1, -1):
            insn = b.insns[idx]
            rule = mutate(insn)
            if not rule:
                continue
            rule, new_asm = rule
            old_eff = mod.effects.get(insn.addr)
            if old_eff is None:
                continue
            try:
                new_bytes = asm(new_asm)
            except Exception:
                continue
            if len(new_bytes) > insn.size:
                continue
            new_bytes = pad(new_bytes, insn.size)
            if not new_bytes:
                continue
            new_eff = effect_of(new_bytes, insn.addr, dis.imports)
            if new_eff is None:
                blocked[rule] = blocked.get(rule, 0) + 1
                continue
            live = mod.live_after.get(insn.addr, set())
            old_defs = old_eff.reg_def | old_eff.flag_def
            new_defs = new_eff.reg_def | new_eff.flag_def
            old_uses = old_eff.reg_use | old_eff.flag_use
            new_uses = new_eff.reg_use | new_eff.flag_use

            excused_read, identity_write = set(), set()
            det_new = next(md.disasm(new_bytes, insn.addr), None)
            if det_new is not None and len(det_new.operands) == 2 \
                    and det_new.operands[0].type == X86_OP_REG \
                    and det_new.operands[1].type == X86_OP_REG \
                    and det_new.operands[0].reg == det_new.operands[1].reg:
                p = parent(md.reg_name(det_new.operands[0].reg))
                if det_new.mnemonic in ("xor", "sub"):
                    excused_read.add(p)
                elif det_new.mnemonic in ("or", "and") and det_new.operands[0].size != 4:
                    identity_write.add(p)

            new_defs_eff = new_defs - identity_write
            new_uses_eff = new_uses - excused_read

            old_mem = old_eff.mem_use | old_eff.mem_def
            new_mem = new_eff.mem_use | new_eff.mem_def

            if ((new_defs_eff - old_defs) & live
                    or (old_defs - new_defs_eff) & live
                    or (new_uses_eff - old_uses)
                    or (new_mem - old_mem)
                    or old_eff.mem_def - new_eff.mem_def
                    or new_eff.stack_delta != old_eff.stack_delta):
                blocked[rule] = blocked.get(rule, 0) + 1
                continue
            file_off = insn.file_off
            old_bytes = data[file_off:file_off + insn.size]
            if bytes(new_bytes) == bytes(old_bytes):
                continue
            print(f"{insn.addr:#x}: {insn.mnemonic} {insn.op_str} ({old_bytes.hex()}) "
                  f"-> {new_asm} ({new_bytes.hex()})")
            data[file_off:file_off + insn.size] = new_bytes
            applied[rule] = applied.get(rule, 0) + 1
            # Обновляем эффект и живость префикса блока: следующая (более
            # ранняя по адресу) мутация должна видеть новый def/use.
            mod.effects[insn.addr] = replace(
                old_eff,
                reg_use=frozenset(new_eff.reg_use),
                reg_def=frozenset(new_eff.reg_def),
                flag_use=frozenset(new_eff.flag_use),
                flag_def=frozenset(new_eff.flag_def),
                mem_use=frozenset(new_eff.mem_use),
                mem_def=frozenset(new_eff.mem_def),
                stack_delta=new_eff.stack_delta,
            )
            if recompute_live_after_prefix(b, mod, idx):
                # live_in блока изменился — пересчитать CFG, иначе
                # upstream live_out останется от оригинала.
                _liveness(dis, mod)

    tmp = dst + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    for rule in sorted(set(applied) | set(blocked)):
        print(f"{rule}: applied={applied.get(rule, 0)} blocked_by_liveness={blocked.get(rule, 0)}")
    print(f"patched {sum(applied.values())} insns -> {dst}")
    return {"applied": applied, "blocked": blocked}


def main(argv=None):
    argv = sys.argv if argv is None else argv
    src = argv[1] if len(argv) > 1 else r"bin\test.exe"
    dst = argv[2] if len(argv) > 2 else r"bin\test_morph.exe"
    mutate_pe(src, dst)


if __name__ == "__main__":
    main()
