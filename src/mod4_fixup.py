"""Module 4: Address & Relocation Fixup for x64 PE.
Input: original PE + mutated PE (same-size in-place mutations from Module 3).
Проверяет адресную целостность мутаций: layout файла и секций, границы
измененных байтов (только код .text и только внутри замененных инструкций —
data-блоки и jump-таблицы из gaps неприкосновенны), значения relocation-слотов,
прямые цели ветвлений и RIP-relative ссылки, равенство stack delta; пересчитывает
PE checksum. В same-size модели пересчет смещений не требуется — модуль это
доказывает; перезапись rel/rip для size-changing мутаций строится на тех же
примитивах (карта релокаций, insn-level diff).

CLI: python mod4_fixup.py <orig> <mutated> [--apply | --out FILE]
Exit 0 = PASS, 1 = FAIL.
"""
import argparse
import gc
import os
import struct
import sys
from bisect import bisect_left
from types import SimpleNamespace

import pefile
from capstone import CS_GRP_CALL, CS_GRP_RET, CS_GRP_JUMP

from mod1_disasm import disassemble, verify
from mod2_ir import _effect
from pipeline_config import PipelineConfig

RELOC_SLOT = {1: 2, 2: 2, 3: 4, 4: 4, 10: 8}
CHECKSUM_OFF = 64


def va2file(sections, va, file_size):
    for _name, sva, vsize, raw_ptr, raw_size, _is_exec in sections:
        off = va - sva
        if 0 <= off < min(vsize or raw_size, raw_size):
            fo = raw_ptr + off
            if 0 <= fo < file_size:
                return fo
    return None


def _merge(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


DIFF_CHUNK = 1 << 20


def diff_offsets(data_o, data_m):
    if data_o == data_m:
        return []
    # Чанк-сравнение (дешёвое, PE обычно отличается в сотне-другой байт),
    # а внутри изменённого чанка — байтовый zip. int.from_bytes на мегабайтных
    # буферах создавал bigint порядка 2**(8*2**20) — ненужная работа.
    offs = []
    mv_o, mv_m = memoryview(data_o), memoryview(data_m)
    for base in range(0, len(data_o), DIFF_CHUNK):
        end = min(base + DIFF_CHUNK, len(data_o))
        a, b = mv_o[base:end], mv_m[base:end]
        if a == b:
            continue
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                offs.append(base + i)
    return offs


def _stack_delta(result, addr, md):
    insn = result.by_addr[addr]
    det = next(md.disasm(insn.raw, insn.addr), None)
    if det is None:
        return None
    # Control-flow флаги — из capstone groups, не из хардкода.
    gs = set(det.groups)
    fake = SimpleNamespace(
        mnemonic=det.mnemonic, addr=det.address,
        regs_read=tuple(md.reg_name(r) for r in det.regs_read),
        regs_write=tuple(md.reg_name(r) for r in det.regs_write),
        is_call=CS_GRP_CALL in gs, is_ret=CS_GRP_RET in gs,
        is_uncond_jmp=CS_GRP_JUMP in gs,
        is_cond_jmp=False, is_loop=False)
    return _effect(fake, det, md, result.imports).stack_delta


def fixup(orig_path, mut_path, cfg: PipelineConfig | None = None):
    cfg = PipelineConfig.from_legacy(cfg=cfg)
    rep = {"errors": [], "warns": [], "stats": {}}
    st = rep["stats"]
    errs = rep["errors"]

    ro = disassemble(orig_path, cfg=cfg)
    rm = disassemble(mut_path, cfg=cfg)
    data_o = open(orig_path, "rb").read()
    data_m = open(mut_path, "rb").read()

    if len(data_o) != len(data_m):
        errs.append(f"file size changed: {len(data_o)} -> {len(data_m)}")
        return rep
    st["file_size"] = len(data_o)

    for so, sm in zip(ro.sections, rm.sections):
        if so[:5] != sm[:5]:
            errs.append(f"section layout changed: {so[0]}")
    st["sections"] = len(ro.sections)

    pdata = next((s for s in ro.sections if s[0].startswith(".pdata")), None)
    st["pdata_funcs"] = (pdata[4] // 12) if pdata and pdata[4] else 0

    exec_secs = [s for s in ro.sections if s[5]]
    exec_file_spans = [(s[3], s[3] + s[4]) for s in exec_secs if s[3] and s[4]]
    exec_ids = {id(s) for s in exec_secs}
    for s in ro.sections:
        if id(s) in exec_ids or not s[4] or not s[3]:
            continue
        fo, sz = s[3], s[4]
        if data_o[fo:fo + sz] != data_m[fo:fo + sz]:
            errs.append(f"data changed outside executable sections: section {s[0]}")

    e_lfanew = struct.unpack_from("<I", data_o, 0x3C)[0]
    cs_lo = e_lfanew + 24 + CHECKSUM_OFF
    cs_hi = cs_lo + 4
    diff_all = diff_offsets(data_o, data_m)
    diff_offs = [o for o in diff_all if not (cs_lo <= o < cs_hi)]
    st["changed_bytes"] = len(diff_offs)
    st["checksum_bytes_patched"] = len(diff_all) - len(diff_offs)
    if exec_file_spans:
        outside = [o for o in diff_offs
                   if not any(lo <= o < hi for lo, hi in exec_file_spans)]
        if outside:
            errs.append(f"{len(outside)} changed bytes outside executable sections "
                        f"(first: {[hex(x) for x in outside[:5]]})")

    addr_o, addr_m = set(ro.by_addr), set(rm.by_addr)
    st["insns_orig"] = len(addr_o)
    st["insns_mut"] = len(addr_m)
    only_o = addr_o - addr_m
    only_m = addr_m - addr_o
    if only_o:
        errs.append(f"{len(only_o)} insns visited only in original: "
                    f"{[hex(a) for a in sorted(only_o)[:5]]}")
    if only_m:
        errs.append(f"{len(only_m)} insns visited only in mutated: "
                    f"{[hex(a) for a in sorted(only_m)[:5]]}")

    mut_insns = []
    for a in sorted(addr_o & addr_m):
        io_, im = ro.by_addr[a], rm.by_addr[a]
        if io_.raw == im.raw:
            continue
        if io_.size != im.size:
            errs.append(f"insn {a:#x}: size changed {io_.size} -> {im.size}")
            continue
        mut_insns.append((io_, im))
    st["mutations"] = len(mut_insns)

    spans = _merge((io.file_off, io.file_off + io.size) for io, _ in mut_insns)
    uncovered = [off for off in diff_offs
                 if not any(s <= off < e for s, e in spans)]
    if uncovered:
        errs.append(f"{len(uncovered)} changed bytes outside any mutated insn "
                    f"(first: {[hex(x) for x in uncovered[:5]]})")

    relocs_sorted = sorted(ro.relocs)
    reloc_overlaps = 0
    for io_, im in mut_insns:
        lo, hi = io_.addr, io_.addr + io_.size
        k = bisect_left(relocs_sorted, lo - 8)
        while k < len(relocs_sorted):
            va = relocs_sorted[k]
            if va >= hi:
                break
            slot = RELOC_SLOT.get(ro.reloc_types.get(va, 10), 8)
            if va + slot > lo:
                fo = va2file(ro.sections, va, len(data_o))
                if fo is not None:
                    reloc_overlaps += 1
                    if data_o[fo:fo + slot] != data_m[fo:fo + slot]:
                        errs.append(f"insn {io_.addr:#x}: relocation slot "
                                    f"{va:#x} ({slot}B) value changed")
            k += 1
    st["relocs_total"] = len(relocs_sorted)
    st["reloc_slots_checked"] = reloc_overlaps

    n_branch = n_rip = 0
    for io_, im in mut_insns:
        if io_.branch_target is not None or im.branch_target is not None:
            n_branch += 1
            if io_.branch_target != im.branch_target:
                errs.append(f"insn {io_.addr:#x}: branch target changed "
                            f"{io_.branch_target} -> {im.branch_target}")
        if io_.rip_ref is not None or im.rip_ref is not None:
            n_rip += 1
            if io_.rip_ref != im.rip_ref:
                errs.append(f"insn {io_.addr:#x}: RIP-relative ref changed "
                            f"{io_.rip_ref} -> {im.rip_ref}")
    st["with_branch_target"] = n_branch
    st["with_rip_ref"] = n_rip

    if mut_insns:
        from capstone import Cs, CS_ARCH_X86, CS_MODE_64
        md = Cs(CS_ARCH_X86, CS_MODE_64)
        md.detail = True
        n_delta = 0
        for io_, im in mut_insns:
            do_ = _stack_delta(ro, io_.addr, md)
            dm_ = _stack_delta(rm, im.addr, md)
            n_delta += 1
            if do_ != dm_:
                errs.append(f"insn {io_.addr:#x}: stack delta changed "
                            f"{do_} -> {dm_}")
        st["stack_delta_checked"] = n_delta

    verrs, vwarns = verify(rm)
    for e in verrs:
        errs.append(f"module1 verify on mutated: {e}")
    st["verify_warns"] = len(vwarns)

    pe = pefile.PE(mut_path)
    old_cs = pe.OPTIONAL_HEADER.CheckSum
    new_cs = pe.generate_checksum() & 0xFFFFFFFF
    st["checksum_stored"] = old_cs
    st["checksum_computed"] = new_cs
    if old_cs != new_cs:
        rep["warns"].append(f"PE checksum stale: stored {old_cs:#010x}, "
                            f"computed {new_cs:#010x} (run with --apply)")
    try:
        dd = pe.OPTIONAL_HEADER.DATA_DIRECTORY[4]  # IMAGE_DIRECTORY_ENTRY_SECURITY
        if dd.VirtualAddress and dd.Size:
            rep["warns"].append(
                "Authenticode signature present; mutations invalidate the signature"
            )
    except Exception:
        pass
    rep["checksum_fix"] = new_cs
    return rep


def apply_checksum(mut_path, new_cs, out_path=None):
    pe = pefile.PE(mut_path)
    base = pe.OPTIONAL_HEADER.get_file_offset() + CHECKSUM_OFF
    if hasattr(pe, "close"):
        pe.close()
    del pe
    gc.collect()
    data = bytearray(open(mut_path, "rb").read())
    struct.pack_into("<I", data, base, new_cs)
    target = out_path or mut_path
    tmp = target + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(bytes(data))
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orig", nargs="?", default="bin/test.exe")
    ap.add_argument("mutated", nargs="?", default="bin/test_morph.exe")
    ap.add_argument("--apply", action="store_true",
                    help="in-place checksum fix on PASS")
    ap.add_argument("--out", help="write checksum-fixed copy to FILE")
    args = ap.parse_args()

    rep = fixup(args.orig, args.mutated)
    st = rep["stats"]
    print("=== Module 4: Address & Relocation Fixup ===")
    print(f"orig: {args.orig}  mutated: {args.mutated}")
    print(f"file_size={st.get('file_size')} sections={st.get('sections')} "
          f"insns {st.get('insns_orig')} -> {st.get('insns_mut')}")
    print(f".pdata: {st.get('pdata_funcs', 0)} RUNTIME_FUNCTIONs, "
          f"byte-identical (same-size model)")
    print(f"mutations: {st.get('mutations')} insns, "
          f"{st.get('changed_bytes')} bytes changed")
    print(f"relocs: {st.get('relocs_total')} slots total, "
          f"{st.get('reloc_slots_checked')} inside mutated insns "
          f"(values preserved)")
    print(f"branch targets preserved: {st.get('with_branch_target', 0)}, "
          f"RIP refs preserved: {st.get('with_rip_ref', 0)}")
    if "stack_delta_checked" in st:
        print(f"stack delta checked: {st.get('stack_delta_checked')}")
    print(f"module1 verify on mutated: {st.get('verify_warns')} warns")
    print(f"checksum: stored {st.get('checksum_stored', 0):#010x}, "
          f"computed {st.get('checksum_computed', 0):#010x}")
    for w in rep["warns"]:
        print("  WARN:", w)
    for e in rep["errors"]:
        print("  ERROR:", e)
    verdict = "PASS" if not rep["errors"] else "FAIL"
    print(f"VERDICT: {verdict} "
          f"({len(rep['errors'])} errors, {len(rep['warns'])} warnings)")

    if rep["errors"] and (args.apply or args.out):
        print("--apply/--out refused: fix errors first")
        sys.exit(1)
    if args.out:
        apply_checksum(args.mutated, rep["checksum_fix"], args.out)
        print(f"checksum-fixed copy written: {args.out}")
    elif args.apply:
        apply_checksum(args.mutated, rep["checksum_fix"])
        print(f"checksum fixed in place: {args.mutated}")
    sys.exit(0 if not rep["errors"] else 1)


if __name__ == "__main__":
    main()
