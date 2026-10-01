"""Module 2: Lifting to IR + liveness for x64 PE.
Input: DisasmResult (Module 1). Output: IRModule with functions, per-insn
effects (regs/flags/mem def-use, stack delta) and block/instruction
liveness for safe mutations in Module 3.

Принцип безопасности: при сомнениях эффекты КОНСЕРВАТИВНЫ —
uses завышаем, defs занижаем. Лишний live = пропущенная мутация,
недостающий live = сломанный бинарник.
"""
import argparse
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_AC_READ, CS_AC_WRITE
from capstone.x86_const import X86_OP_REG, X86_OP_IMM, X86_OP_MEM

from mod1_disasm import disassemble, parent, COND_JMPS, LOOP_INSNS

# Маппинг регистров к 64-битному родителю берётся из mod1_disasm — единый
# источник, чтобы liveness не разъехалась при изменении таблицы в одном модуле.

GPRS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
VEC = tuple(f"xmm{i}" for i in range(16))  # отслеживаем XMM (SSE-скаляры в CRT)
VOLATILE = ("rax", "rcx", "rdx", "r8", "r9", "r10", "r11")  # clobbered by call (MS x64)
VOLATILE_VEC = tuple(f"xmm{i}" for i in range(6))  # XMM0-XMM5 volatile по MS x64
ARG_REGS = ("rcx", "rdx", "r8", "r9")  # первые целочисленные аргументы (MS x64)
ARG_VEC = tuple(f"xmm{i}" for i in range(4))  # первые FP-аргументы (MS x64)
NON_VOLATILE = {"rbx", "rbp", "rsi", "rdi", "r12", "r13", "r14", "r15", "rsp"}
RET_LIVE = frozenset({"rax", "rdx", "xmm0"} | NON_VOLATILE)
_TRACKED = frozenset(GPRS) | frozenset(VEC)  # остальное (fs/gs/ymm/zmm/rip) не ведем
FLAGS = ("zf", "sf", "cf", "of", "pf", "af", "df")
ARITH_FLAGS = ("zf", "sf", "cf", "of", "pf", "af")  # add/sub/cmp (без DF)

# jcc -> читаемые флаги
CC_FLAGS = {
    "a": ("cf", "zf"), "ae": ("cf",), "b": ("cf",), "be": ("cf", "zf"),
    "c": ("cf",), "e": ("zf",), "g": ("zf", "sf", "of"), "ge": ("sf", "of"),
    "l": ("sf", "of"), "le": ("zf", "sf", "of"), "na": ("cf", "zf"),
    "nae": ("cf",), "nb": ("cf",), "nbe": ("cf", "zf"), "nc": ("cf",),
    "ne": ("zf",), "ng": ("zf", "sf", "of"), "nge": ("sf", "of"),
    "nl": ("sf", "of"), "nle": ("zf", "sf", "of"), "no": ("of",),
    "np": ("pf",), "ns": ("sf",), "nz": ("zf",), "o": ("of",), "p": ("pf",),
    "pe": ("pf",), "po": ("pf",), "s": ("sf",), "z": ("zf",),
}


@dataclass
class MemRef:
    base: str | None = None
    index: str | None = None
    scale: int = 1
    disp: int = 0
    rip_target: int | None = None   # VA при [rip+...]
    kind: str = "unknown"           # stack | global | iat | unknown


@dataclass
class IREffect:
    addr: int
    reg_use: frozenset = frozenset()
    reg_def: frozenset = frozenset()
    flag_use: frozenset = frozenset()
    flag_def: frozenset = frozenset()
    mem_use: frozenset = frozenset()   # сырые слоты "s:rsp+0x20" / "g:0xVA" / "m:*"
    mem_def: frozenset = frozenset()
    vfp_use: frozenset = frozenset()   # VFP-нормализованные (liveness ест только их)
    vfp_def: frozenset = frozenset()
    mems: tuple = ()                # [MemRef]
    stack_delta: int | None = None  # изменение rsp (None = неизвестно; call = 0)
    calls: tuple = ()               # цели/имена (копия из блока)
    is_control: bool = False


@dataclass
class Function:
    begin: int
    end: int
    blocks: list = field(default_factory=list)  # старты блоков
    inferred: bool = False  # True если нет в .pdata (только по call-таргетам)


@dataclass
class IRModule:
    path: str
    functions: list = field(default_factory=list)
    effects: dict = field(default_factory=dict)      # addr -> IREffect
    live_in: dict = field(default_factory=dict)      # block_start -> set[str]
    live_out: dict = field(default_factory=dict)     # block_start -> set[str]
    live_after: dict = field(default_factory=dict)   # insn addr -> set[str] (regs+flags+slots)
    func_of: dict = field(default_factory=dict)      # insn addr -> Function|None
    vfp_in: dict = field(default_factory=dict)       # block_start -> int|None (смещение rsp от входа)
    reachable: set = field(default_factory=set)      # достижимые блоки (от входов функций)
    vfp_degraded: int = 0                            # функций с неизвестным стеком
    unresolved_tables: list = field(default_factory=list)  # table_va без резолва


def _full_width(parent_reg: str, size: int) -> bool:
    """Полная ли перезапись: GPR size>=4 (4 обнуляет верх), XMM size>=16."""
    if parent_reg in VEC:
        return size >= 16
    return size >= 4


def _mems_of(detail_insn, md, imports) -> tuple:
    refs = []
    for op in detail_insn.operands:
        if op.type != X86_OP_MEM:
            continue
        m = op.mem
        try:
            base = md.reg_name(m.base) if m.base else None
        except Exception:
            base = None
        try:
            index = md.reg_name(m.index) if m.index else None
        except Exception:
            index = None
        base_p = parent(base) if base and base != "rip" else base
        ref = MemRef(base=base_p,
                     index=parent(index) if index else None,
                     scale=m.scale or 1, disp=m.disp)
        if base == "rip":
            ref.rip_target = detail_insn.address + detail_insn.size + m.disp
            ref.kind = "iat" if ref.rip_target in imports else "global"
        elif base_p in ("rsp", "rbp"):
            ref.kind = "stack"
        refs.append(ref)
    return tuple(refs)


def _slot_of_memop(detail_insn, op, md, imports) -> str:
    """Канонический слот памяти для liveness: 's:rsp+0x20', 'g:0x14001abc', 'm:*'.
    Слоты только для точных адресов (stack без индекса / RIP). Остальное — 'm:*'."""
    m = op.mem
    try:
        base = md.reg_name(m.base) if m.base else None
    except Exception:
        base = None
    try:
        has_index = bool(m.index)
    except Exception:
        has_index = False
    if base == "rip":
        va = detail_insn.address + detail_insn.size + m.disp
        return f"g:{va:#x}"
    if base and parent(base) in ("rsp", "rbp") and not has_index:
        return f"s:{parent(base)}{m.disp:+#x}"
    return "m:*"


def _effect(insn, detail, md, imports) -> IREffect:
    """Эффекты инструкции. Conservative: uses завышаем, defs только точные."""
    m = insn.mnemonic
    eff = IREffect(addr=insn.addr, is_control=bool(
        insn.is_call or insn.is_ret or insn.is_uncond_jmp or insn.is_cond_jmp or insn.is_loop))
    use, dfn = set(), set()
    fuse, fdef = set(), set()
    muse, mdef = set(), set()
    for op in detail.operands:
        if op.type == X86_OP_REG:
            try:
                p = parent(md.reg_name(op.reg))
            except Exception:
                continue
            if p not in _TRACKED:
                continue  # fs/gs/ymm/zmm и прочее не ведем
            try:
                acc, w = op.access, op.size
            except Exception:
                acc, w = 0, 8
            if acc & CS_AC_WRITE and not acc & CS_AC_READ:
                if _full_width(p, w or 8):
                    dfn.add(p)          # mov rax,1 / mov eax,1 — полный def
                else:
                    use.add(p)          # mov al,1 — частичная: RMW, def НЕТ
            elif acc & CS_AC_READ and not acc & CS_AC_WRITE:
                use.add(p)
            else:  # RW или access=0 (консервативно): чтение + def при полной ширине
                use.add(p)
                if acc & CS_AC_WRITE and _full_width(p, w or 0):
                    dfn.add(p)
        elif op.type == X86_OP_MEM:
            # направление по access-флагам capstone; без них — консервативно use
            try:
                acc = op.access
            except Exception:
                acc = CS_AC_READ
            slot = _slot_of_memop(detail, op, md, imports)
            r = bool(acc & CS_AC_READ)
            w = bool(acc & CS_AC_WRITE)
            if w:
                mdef.add(slot)
            # RMW (add/inc/xadd на памяти) — одновременно и use, и def.
            # access=0 (неизвестно) — консервативно трактуем как read.
            if r or not w:
                muse.add(slot)
    mems = _mems_of(detail, md, imports)
    for r in mems:
        if r.base and r.base in _TRACKED:
            use.add(r.base)
        if r.index and r.index in _TRACKED:
            use.add(r.index)
    # точные defs регистров из capstone regs_write (только трекаемые)
    for rw in insn.regs_write:
        p = parent(rw)
        if p in _TRACKED:
            dfn.add(p)
    # stack_delta: 0 = rsp точно не меняется; None = неизвестно.
    # (Раньше было None всегда — это отравляло VFP через любой mov.)
    delta, delta_known = 0, True
    if m in ("mov", "movsx", "movsxd", "movzx", "lea", "nop", "pause",
             "endbr64") or m.startswith("mov"):
        pass  # флаги не трогает (mov-семейство/lea)
    elif m in ("add", "sub", "adc", "sbb", "neg"):
        fdef.update(ARITH_FLAGS)
    elif m in ("cmp", "test"):
        fdef.update(ARITH_FLAGS)
        dfn.clear()  # cmp/test регистров не пишут
    elif m in ("and", "or", "xor"):
        fdef.update([f for f in ARITH_FLAGS if f != "af"])  # AF undefined
    elif m in ("inc", "dec"):
        fdef.update([f for f in ARITH_FLAGS if f != "cf"])  # CF сохраняется
    elif m in ("shl", "shr", "sar", "rol", "ror", "shld", "shrd"):
        fdef.update([f for f in ARITH_FLAGS if f != "af"])
    elif m in ("mul", "imul", "div", "idiv"):
        fdef.update(("cf", "of"))  # минимум; остальное считаем живым
    elif m in ("push", "pushfq"):
        delta = -8
        use.add("rsp")
        dfn.add("rsp")
        mdef.add("s:rsp-0x8")  # push физически пишет в [rsp-8]
        if m == "pushfq":
            fuse.update(FLAGS)
    elif m in ("pop", "popfq"):
        delta = 8
        use.add("rsp")
        dfn.add("rsp")
        muse.add("s:rsp+0x0")  # pop физически читает [rsp]
        if m == "popfq":
            fdef.update(FLAGS)
    elif m in ("call",):
        use.update(ARG_REGS)
        use.update(ARG_VEC)
        dfn.update(VOLATILE)
        dfn.update(VOLATILE_VEC)
        fdef.update(FLAGS)  # ABI: флаги через call не живут
        delta = 0  # call архитектурно сбалансирован (push ret-addr + возврат)
        # память через call НЕ убиваем (консервативно: остается живой)
    # jcc/cmovcc/setcc читают флаги
    cc = None
    if m in COND_JMPS:
        cc = m[1:]
    elif m.startswith("cmov") or m.startswith("set"):
        cc = m[4:] if m.startswith("cmov") else m[3:]
    if cc and cc in CC_FLAGS:
        fuse.update(CC_FLAGS[cc])
    if m in LOOP_INSNS:
        use.add("rcx")
        if m in ("loope", "loopz"):
            fuse.add("zf")
        if m in ("loopne", "loopnz"):
            fuse.add("zf")
    if m in ("jecxz", "jcxz"):
        use.add("rcx")
    if m == "jrcxz":
        use.add("rcx")
    if m == "ret":
        use.add("rsp")
    # sub/add rsp,imm — точная дельта стека (с явной проверкой числа операндов)
    rsp_imm_exact = False
    if m in ("sub", "add") and len(detail.operands) >= 2:
        try:
            o0, o1 = detail.operands[0], detail.operands[1]
            if (o0.type == X86_OP_REG and parent(md.reg_name(o0.reg)) == "rsp"
                    and o1.type == X86_OP_IMM):
                delta = -o1.imm if m == "sub" else o1.imm
                rsp_imm_exact = True
        except Exception:
            pass
    # rsp пишется на неизвестную величину (mov rsp,reg / and rsp,.. / leave)
    # push/pop/call/sub-add-imm дают точную дельту и сюда не попадают
    if m in ("ret", "retf", "leave", "enter", "iret", "iretd", "iretq"):
        delta_known = False
    elif m not in ("push", "pop", "call") and not rsp_imm_exact:
        try:
            ops = detail.operands
        except Exception:
            ops = ()
        for op in ops:
            if op.type != X86_OP_REG:
                continue
            try:
                is_rsp = parent(md.reg_name(op.reg)) == "rsp"
                acc = op.access
            except Exception:
                continue
            if is_rsp and (acc & CS_AC_WRITE):
                delta_known = False
                break
        if delta_known:
            for rw in insn.regs_write:  # неявные (leave пишет rsp)
                if parent(rw) == "rsp":
                    delta_known = False
                    break
    if not delta_known:
        delta = None
    # capstone regs_read точнее ручного для сложных (xchg, cpuid...): добавляем.
    # regs_write как use-only: ширины нет, полный def додумывать нельзя
    # (единственный достоверный там — rsp у push/pop/call, он уже учтен выше).
    for rr in insn.regs_read:
        p = parent(rr)
        if p in _TRACKED:
            use.add(p)
    for rw in insn.regs_write:
        p = parent(rw)
        if p in _TRACKED and p not in dfn:
            use.add(p)
    use.discard("rip")
    dfn.discard("rip")
    return IREffect(addr=insn.addr, reg_use=frozenset(use), reg_def=frozenset(dfn),
                    flag_use=frozenset(fuse), flag_def=frozenset(fdef),
                    mem_use=frozenset(muse), mem_def=frozenset(mdef),
                    mems=mems, stack_delta=delta, is_control=eff.is_control)


def lift(path: str, split_on_call: bool = True) -> IRModule:
    dis = disassemble(path, split_on_call=split_on_call)
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    mod = IRModule(path=path)
    # эффекты: пере-декодируем каждую инструкцию из raw для операндов
    for insn in dis.insns:
        try:
            det = next(md.disasm(insn.raw, insn.addr))
        except StopIteration:
            continue
        mod.effects[insn.addr] = _effect(insn, det, md, dis.imports)
    # функции: .pdata + прямые call-таргеты вне pdata (inferred).
    # Конец inferred — эвристика: следующее начало функции (pdata или другой
    # таргет), иначе конец exec-диапазона. 1-байтных огрызков больше нет.
    ranges = list(dis.func_bounds)
    pdata_begins = sorted(b for b, _ in ranges)

    def _in_pdata(t: int) -> bool:
        i = bisect_right(pdata_begins, t) - 1
        return i >= 0 and t < ranges[i][1]

    extra = set()
    for insn in dis.insns:
        if insn.is_call and insn.branch_target and insn.branch_target in dis.by_addr:
            t = insn.branch_target
            if not _in_pdata(t):
                extra.add(t)
    exec_end = max((e for _, e in dis.exec_ranges), default=dis.imagebase)
    bounds = sorted({b for b, _ in ranges} | extra | {exec_end})
    for t in sorted(extra):
        nxt = next((x for x in bounds if x > t), exec_end)
        if (t, nxt) not in set(ranges) and t < nxt:
            ranges.append((t, nxt))
    ranges.sort()
    funcs = [Function(b, e, inferred=(b in extra)) for b, e in ranges]
    # привязка блоков/инструкций к функциям — bisect O(log F)
    by_start = {b.start: b for b in dis.blocks}
    begins = [f.begin for f in funcs]  # ranges уже отсортированы
    block_starts = sorted(by_start)
    for f in funcs:
        lo = bisect_left(block_starts, f.begin)
        hi = bisect_left(block_starts, f.end, lo=lo)
        f.blocks = block_starts[lo:hi]
    for insn in dis.insns:
        i = bisect_right(begins, insn.addr) - 1
        mod.func_of[insn.addr] = funcs[i] if i >= 0 and insn.addr < funcs[i].end else None
    mod.functions = funcs
    # неразрезолвленные таблицы — на вход Модулю 2-табличному
    for insn in dis.insns:
        if insn.is_table_jmp and insn.table_va is not None:
            mod.unresolved_tables.append(insn.table_va)
    _vfp_pass(dis, mod)   # сначала стек (vfp_use/vfp_def), потом живость
    _liveness(dis, mod)
    # calls в эффекты из блоков (присваиванием — без потери mem/vfp полей)
    for b in dis.blocks:
        last = b.insns[-1] if b.insns else None
        if last and last.addr in mod.effects and (last.is_call or b.calls):
            mod.effects[last.addr].calls = tuple(b.calls)
    return mod


def _vfp_pass(dis, mod: IRModule):
    """Forward dataflow смещения rsp от входа функции (Virtual Frame Pointer).
    vfp_in[block] = int | None. None (poison) при: динамическом стеке
    (sub rsp,reg), слиянии с разными offset, недостижимости.
    Затем слоты s:rsp+X транслируются в s:vfp+K (vfp_use/vfp_def);
    при None — в m:* (деградация, стек этой функции мутатору закрыт)."""
    by_start = {b.start: b for b in dis.blocks}
    preds: dict = {b.start: set() for b in dis.blocks}
    for b in dis.blocks:
        for s in b.succs:
            if s in preds:
                preds[s].add(b.start)
    func_blocks: dict = {}
    for f in mod.functions:
        func_blocks[id(f)] = set(f.blocks)
    block_func = {}
    for f in mod.functions:
        for bs in f.blocks:
            block_func[bs] = f
    # достижимость от входов функций (паддинг/мертвый код отсекается здесь,
    # а не нулем: мертвый pred больше не подмешивает ложный offset 0)
    reachable = set()
    for f in mod.functions:
        fb = func_blocks[id(f)]
        entry = f.begin if f.begin in fb else (min(fb) if fb else None)
        if entry is None:
            continue
        stack = [entry]
        while stack:
            cur = stack.pop()
            if cur in reachable or cur not in fb:
                continue
            reachable.add(cur)
            stack.extend(s for s in by_start[cur].succs if s in fb)
    mod.reachable = reachable
    # обратные ребра (циклы): в слиянии не участвуют — тело цикла без
    # динамического стека обязано быть сбалансировано (иначе rsp дрейфует
    # и программа падает). Иначе любой цикл травил бы весь граф.
    backedges = set()
    for f in mod.functions:
        fb = func_blocks[id(f)]
        entry = f.begin if f.begin in fb else (min(fb) if fb else None)
        if entry is None:
            continue
        color = {entry: 1}
        stack = [(entry, iter(sorted(s for s in by_start[entry].succs if s in fb)))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if color.get(nxt, 0) == 1:
                    backedges.add((node, nxt))
                elif color.get(nxt, 0) == 0:
                    color[nxt] = 1
                    stack.append((nxt, iter(sorted(s for s in by_start[nxt].succs if s in fb))))
                    advanced = True
                    break
            if not advanced:
                color[node] = 2
                stack.pop()
    vfp = {b.start: None for b in dis.blocks}
    for f in mod.functions:
        fb = func_blocks[id(f)]
        entry = f.begin if f.begin in fb else (min(fb) if fb else None)
        if entry is not None:
            vfp[entry] = 0
    def out_of(bs):
        base = vfp[bs]
        if base is None:
            return None
        total = base
        for insn in by_start[bs].insns:
            e = mod.effects.get(insn.addr)
            d = e.stack_delta if e is not None else None
            if d is None:
                return None
            total += d
        return total
    changed = True
    while changed:
        changed = False
        for b in dis.blocks:
            if b.start not in reachable:
                continue  # мертвый код: vfp None навсегда, чужие слияния не травит
            f = block_func.get(b.start)
            if f is None:
                continue
            fb = func_blocks[id(f)]
            entry = f.begin if f.begin in fb else min(fb)
            if b.start == entry:
                continue  # вход всегда 0
            pin = sorted((preds[b.start] & fb) & reachable - {p for p, d in backedges if d == b.start})
            if not pin:
                continue  # висячий достижимый? только вход; иначе None
            outs = {out_of(p) for p in pin}
            merged = outs.pop() if len(outs) == 1 else None
            if merged != vfp[b.start]:
                vfp[b.start] = merged
                changed = True
    mod.vfp_in = vfp
    # трансляция слотов по инструкциям
    for b in dis.blocks:
        off = vfp[b.start]
        for insn in b.insns:
            e = mod.effects.get(insn.addr)
            if e is None:
                continue
            if off is None:
                vu = {s if not s.startswith("s:rsp") else "m:*" for s in e.mem_use}
                vd = {s if not s.startswith("s:rsp") else "m:*" for s in e.mem_def}
            else:
                vu = {_vfp_slot(s, off) for s in e.mem_use}
                vd = {_vfp_slot(s, off) for s in e.mem_def}
            e.vfp_use, e.vfp_def = frozenset(vu), frozenset(vd)
            d = e.stack_delta
            if d is not None and off is not None:
                off = off + d
            elif d is None:
                off = None
    mod.vfp_degraded = sum(
        1 for f in mod.functions
        if any(vfp.get(bs) is None for bs in f.blocks if bs in reachable))


def _vfp_slot(slot: str, off: int) -> str:
    """s:rsp+X при текущем смещении off -> s:vfp+(off+X).
    Знак парсим вручную: int('-0x8', 16) валиден не на всех версиях Python."""
    if slot.startswith("s:rsp"):
        try:
            raw = slot[5:]
            sign = -1 if raw.startswith("-") else 1
            val = sign * int(raw.lstrip("+-"), 16)
            return f"s:vfp{off + val:+#x}"
        except ValueError:
            return "m:*"
    return slot


def _liveness(dis, mod: IRModule):
    """Backward dataflow по regs+flags+VFP-слотам. m:* kill запрещен."""
    use_b, def_b = {}, {}
    for b in dis.blocks:
        u, d = set(), set()
        for insn in b.insns:
            e = mod.effects.get(insn.addr)
            if e is None:
                continue
            for r in e.reg_use | e.flag_use | e.vfp_use:
                if r not in d:
                    u.add(r)
            # m:* никогда не убивает (парадокс неизвестной памяти):
            # запись в [rax] не затирает чтения из [rbx]
            d.update(e.reg_def | e.flag_def | (set(e.vfp_def) - {"m:*"}))
        use_b[b.start], def_b[b.start] = u, d
        mod.live_in[b.start] = set()
        mod.live_out[b.start] = set()
    for b in dis.blocks:
        if b.kind == "ret":
            mod.live_out[b.start] = set(RET_LIVE)
    changed = True
    while changed:
        changed = False
        for b in reversed(dis.blocks):
            out = set()
            for s in b.succs:
                out |= mod.live_in.get(s, set())
            if b.kind == "ret":
                out |= RET_LIVE
            inn = use_b[b.start] | (out - def_b[b.start])
            if inn != mod.live_in[b.start] or out != mod.live_out[b.start]:
                mod.live_in[b.start], mod.live_out[b.start] = inn, out
                changed = True
    # live_after инструкций — проход назад внутри блока
    for b in dis.blocks:
        live = set(mod.live_out[b.start])
        for insn in reversed(b.insns):
            mod.live_after[insn.addr] = set(live)
            e = mod.effects.get(insn.addr)
            if e is not None:
                # m:* в kill запрещен и тут (см. выше)
                safe_mdef = set(e.vfp_def) - {"m:*"}
                live = ((live - set(e.reg_def) - set(e.flag_def) - safe_mdef)
                        | set(e.reg_use) | set(e.flag_use) | set(e.vfp_use))


def summary(mod: IRModule) -> str:
    n_eff = len(mod.effects)
    n_real = sum(1 for f in mod.functions if not f.inferred)
    n_inf = sum(1 for f in mod.functions if f.inferred)
    n_assign = sum(1 for a in mod.func_of.values() if a is not None)
    total = len(mod.func_of)
    avg_live = (sum(len(v) for v in mod.live_in.values()) / max(1, len(mod.live_in)))
    return (f"IR: {mod.path} funcs={n_real}+{n_inf}inferred "
            f"effects={n_eff} assigned={n_assign}/{total} "
            f"avg_live_in={avg_live:.1f} vfp_degraded={mod.vfp_degraded} "
            f"unresolved_tables={len(mod.unresolved_tables)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["bin/test.exe"])
    ap.add_argument("--ir-blocks", action="store_true")
    args = ap.parse_args()
    for p in args.files:
        mod = lift(p, split_on_call=not args.ir_blocks)
        print(summary(mod))
        f0 = next((f for f in mod.functions if f.blocks), None)
        if f0:
            b0 = f0.blocks[0]
            print(f"  func {f0.begin:#x}-{f0.end:#x} nblocks={len(f0.blocks)} "
                  f"live_in[{b0:#x}]={sorted(mod.live_in.get(b0, ()))[:12]}")
