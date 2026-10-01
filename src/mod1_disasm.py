"""Module 1: Recursive Descent Disassembly & Control Flow for x64 PE.
Input: PE path. Output: DisasmResult with insns, basic blocks, CFG edges,
imports, relocs, RIP-relative refs, resolved jump tables, data gaps.
For use by Module 2 (IR) / Module 3.

Драйвер декодирования — рекурсивный спуск: worklist
стартует с entry_va и всех begin из .pdata, дальше разбираются только
цели call/jcc/loop/jmp и case-ветки jump-таблиц. Терминалы линии разбора:
ret, косвенный jmp, int3, hlt, недекодируемый байт, call noreturn.
Байты exec-секций, не попавшие в visited (jump-таблицы, строки,
выравнивание, недостижимый код), остаются в gaps как Data Blocks:
при пересборке копируются байт-в-байт, мутатору не выдаются.
"""
import argparse
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, field

import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CS_AC_WRITE
from capstone.x86_const import X86_OP_IMM, X86_OP_MEM, X86_OP_REG


@dataclass
class Insn:
    addr: int          # VA
    file_off: int      # file offset
    size: int
    raw: bytes
    mnemonic: str
    op_str: str
    regs_read: tuple = ()
    regs_write: tuple = ()
    groups: tuple = ()
    is_call: bool = False
    is_ret: bool = False
    is_uncond_jmp: bool = False
    is_cond_jmp: bool = False
    is_indirect: bool = False      # call/jmp через регистр/память, цель неизвестна
    is_table_jmp: bool = False     # jmp [reg*scale+...] — кандидат на jump-table (switch)
    is_import_jmp: bool = False    # jmp/call [rip+IAT] — thunk на импорт, НЕ switch
    is_noreturn: bool = False      # call на exit/abort/TerminateProcess (по имени импорта)
    is_loop: bool = False          # loop/loope/loopne/jcxz/jecxz/jrcxz
    is_padding: bool = False       # int3/nop/hlt
    table_va: int | None = None    # VA таблицы; None после успешного резолва
    table_targets: tuple = ()      # case-ветки после резолва таблицы
    branch_target: int | None = None   # absolute VA if direct
    rip_ref: int | None = None         # absolute VA if RIP-relative mem (для IAT это СЛОТ, не цель!)
    import_name: str | None = None     # "dll.func" если rip_ref/branch_target попал в IAT
    section: str = ""


@dataclass
class BasicBlock:
    start: int
    insns: list = field(default_factory=list)
    succs: list = field(default_factory=list)  # только INTRA-procedural (fall/ветвления/case-ветки)
    calls: list = field(default_factory=list)  # INTER-procedural: VA цели, import-имя (str) или None
    call_names: list = field(default_factory=list)  # параллельно calls: "dll.func" или None
    kind: str = "fall"  # fall | uncond | cond | call | call_noreturn | ret | indirect | loop
    is_padding: bool = False

    @property
    def end(self):
        return self.insns[-1].addr if self.insns else self.start


@dataclass
class DisasmResult:
    path: str
    imagebase: int
    entry_va: int
    sections: list       # [(name, vaddr, vsize, raw_ptr, raw_size, is_exec)]
    insns: list          # [Insn] — только visited (реальный код)
    by_addr: dict
    blocks: list         # [BasicBlock]
    imports: dict        # va -> "dll.func"
    relocs: set          # set of VA that have relocations (все типы != ABSOLUTE)
    reloc_types: dict = field(default_factory=dict)  # VA -> type
    exec_ranges: list = field(default_factory=list)  # [(va_start, va_end)]
    gaps: list = field(default_factory=list)  # [(va_start, va_end)] Data Blocks: всё не-visited
    func_bounds: list = field(default_factory=list)  # [(begin_va, end_va)] из .pdata
    jump_tables: dict = field(default_factory=dict)  # insn VA -> (table_va, (targets...))
    is_dll: bool = False
    dynamic_base: bool = False   # IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
    has_reloc_dir: bool = False
    file_size: int = 0


COND_JMPS = {
    "ja", "jae", "jb", "jbe", "jc", "je", "jz", "jg", "jge", "jl", "jle",
    "jna", "jnae", "jnb", "jnbe", "jnc", "jne", "jnz", "jng", "jnge", "jnl",
    "jnle", "jno", "jnp", "jns", "jo", "jp", "jpe", "jpo", "js",
}
LOOP_INSNS = {"loop", "loope", "loopne", "loopnz", "loopz", "jcxz", "jecxz", "jrcxz"}

PADDING_MNEMS = {"int3", "nop", "hlt"}

NORETURN_SUBSTR = ("exitprocess", "terminateprocess", "abort", "fatalexit", "longjmp")

MAX_TABLE_ENTRIES = 1024

_PARENTS = {}
for _fam, _subs in {
    "rax": ("al", "ah", "ax", "eax"), "rbx": ("bl", "bh", "bx", "ebx"),
    "rcx": ("cl", "ch", "cx", "ecx"), "rdx": ("dl", "dh", "dx", "edx"),
    "rsi": ("sil", "si", "esi"), "rdi": ("dil", "di", "edi"),
    "rbp": ("bpl", "bp", "ebp"), "rsp": ("spl", "sp", "esp"),
}.items():
    _PARENTS[_fam] = _fam
    for _s in _subs:
        _PARENTS[_s] = _fam
for _n in range(8, 16):
    _r = f"r{_n}"
    _PARENTS[_r] = _r
    for _s in (f"r{_n}b", f"r{_n}w", f"r{_n}d"):
        _PARENTS[_s] = _r


def parent(reg: str) -> str:
    return _PARENTS.get(reg.lower(), reg.lower())


@dataclass
class PEInfo:
    """Результат парсинга PE."""
    pe: object
    imagebase: int
    entry_va: int
    sections: list
    exec_ranges: list      # [(va_start, raw_ptr, raw_bytes, name)]
    imports: dict
    relocs: set
    reloc_types: dict
    is_dll: bool = False
    dynamic_base: bool = False
    has_reloc_dir: bool = False
    func_bounds: list = field(default_factory=list)  # [(begin_va, end_va)] из .pdata
    file_size: int = 0


def _parse_pe(path) -> PEInfo:
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Module1: file not found: {path}")
    try:
        pe = pefile.PE(path)
    except pefile.PEFormatError as e:
        raise ValueError(f"Module1: not a valid PE: {path} ({e})")
    file_size = os.path.getsize(path)
    imagebase = pe.OPTIONAL_HEADER.ImageBase
    entry_va = imagebase + pe.OPTIONAL_HEADER.AddressOfEntryPoint
    is_dll = bool(pe.FILE_HEADER.Characteristics & 0x2000)
    dll_chars = getattr(pe.OPTIONAL_HEADER, "DllCharacteristics", 0) or 0
    dynamic_base = bool(dll_chars & 0x0040)
    sections = []
    exec_ranges = []
    for s in pe.sections:
        name = s.Name.rstrip(b"\x00").decode(errors="replace")
        va = imagebase + s.VirtualAddress
        is_exec = bool(s.Characteristics & 0x20000000)
        sections.append((name, va, s.Misc_VirtualSize,
                         s.PointerToRawData, s.SizeOfRawData, is_exec))
        if is_exec and s.SizeOfRawData and s.PointerToRawData:
            valid = min(s.Misc_VirtualSize or s.SizeOfRawData, s.SizeOfRawData)
            raw = s.get_data()[:valid]
            max_avail = max(0, file_size - s.PointerToRawData)
            raw = raw[:max_avail]
            if raw:
                exec_ranges.append((va, s.PointerToRawData, bytes(raw), name))
    imports = {}
    try:
        imp_dir = pe.DIRECTORY_ENTRY_IMPORT
    except AttributeError:
        imp_dir = []
    for entry in imp_dir or []:
        try:
            dll = (entry.dll or b"?").decode(errors="replace")
        except Exception:
            dll = "?"
        for imp in getattr(entry, "imports", []) or []:
            try:
                if imp.address:
                    n = imp.name.decode(errors="replace") if imp.name else f"ord{imp.ordinal}"
                    imports[imp.address] = f"{dll}.{n}"
            except Exception:
                continue
    relocs = set()
    reloc_types = {}
    has_reloc_dir = False
    try:
        reloc_dir = pe.DIRECTORY_ENTRY_BASERELOC
    except AttributeError:
        reloc_dir = []
    if reloc_dir:
        has_reloc_dir = True
        for block in reloc_dir:
            for e in getattr(block, "entries", []) or []:
                if e.type != 0:
                    va = imagebase + e.rva
                    relocs.add(va)
                    reloc_types[va] = e.type
    func_bounds = []
    try:
        exc_dir = pe.DIRECTORY_ENTRY_EXCEPTION
    except AttributeError:
        exc_dir = []
    for r in exc_dir or []:
        try:
            b, e = imagebase + r.struct.BeginAddress, imagebase + r.struct.EndAddress
            if b < e:
                func_bounds.append((b, e))
        except Exception:
            continue
    func_bounds = sorted(set(func_bounds))
    return PEInfo(pe=pe, imagebase=imagebase, entry_va=entry_va,
                  sections=sections, exec_ranges=exec_ranges, imports=imports,
                  relocs=relocs, reloc_types=reloc_types, is_dll=is_dll,
                  dynamic_base=dynamic_base, has_reloc_dir=has_reloc_dir,
                  func_bounds=func_bounds, file_size=file_size)


def disassemble(path: str, split_on_call: bool = True) -> DisasmResult:
    """Рекурсивный спуск: worklist (entry + .pdata begin), trace-линии с
    остановкой на терминалах, резолвинг jump-таблиц. split_on_call=True —
    блок закрывается на call (гранулярность для Модуля 3), False —
    классические basic blocks для IR (Модуля 2)."""
    info = _parse_pe(path)
    imagebase, entry_va = info.imagebase, info.entry_va
    sections, exec_ranges = info.sections, info.exec_ranges
    imports, relocs, reloc_types = info.imports, info.relocs, info.reloc_types
    is_dll, dynamic_base = info.is_dll, info.dynamic_base
    has_reloc_dir, file_size = info.has_reloc_dir, info.file_size
    func_bounds = info.func_bounds

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    raw_data = info.pe.__data__

    exec_spans = [(va, va + len(code)) for va, _, code, _ in exec_ranges]
    exec_sorted = sorted(exec_ranges)
    exec_starts = [r[0] for r in exec_sorted]

    def in_exec(va: int) -> bool:
        return any(a <= va < b for a, b in exec_spans)

    def va2file(va: int):
        for _name, sva, vsize, raw_ptr, raw_size, _is_exec in sections:
            off = va - sva
            if 0 <= off < min(vsize or raw_size, raw_size):
                fo = raw_ptr + off
                if 0 <= fo < file_size:
                    return fo
        return None

    def read_at(va: int, n: int):
        fo = va2file(va)
        if fo is None or fo + n > file_size:
            return None
        return bytes(raw_data[fo:fo + n])

    def decode_one(addr: int):
        k = bisect_right(exec_starts, addr) - 1
        if k < 0:
            return None, None, None, None
        va_start, raw_ptr, code, sec_name = exec_sorted[k]
        if not (va_start <= addr < va_start + len(code)):
            return None, None, None, None
        off = addr - va_start
        first = None
        try:
            for cand in md.disasm(code[off:off + 16], addr, count=1):
                first = cand
                break
        except Exception:
            first = None
        if first is not None and off + first.size > len(code):
            first = None
        return first, va_start, raw_ptr, sec_name

    def make_insn(i, va_start, raw_ptr, sec_name) -> Insn:
        file_off = raw_ptr + (i.address - va_start)
        cs_regs_r = tuple(md.reg_name(r) for r in i.regs_read)
        cs_regs_w = tuple(md.reg_name(r) for r in i.regs_write)
        try:
            grp_names = tuple(md.group_name(g) for g in i.groups)
        except Exception:
            grp_names = ()
        ins = Insn(addr=i.address, file_off=file_off, size=i.size,
                   raw=bytes(i.bytes), mnemonic=i.mnemonic, op_str=i.op_str,
                   regs_read=cs_regs_r, regs_write=cs_regs_w,
                   groups=grp_names, section=sec_name)
        m = i.mnemonic
        ins.is_call = (m == "call")
        ins.is_ret = m in ("ret", "retf", "iret", "iretd", "iretq")
        ins.is_uncond_jmp = (m == "jmp")
        ins.is_cond_jmp = (m in COND_JMPS)
        ins.is_loop = (m in LOOP_INSNS)
        ins.is_padding = (m in PADDING_MNEMS)
        if (ins.is_call or ins.is_uncond_jmp or ins.is_cond_jmp or ins.is_loop) and i.operands:
            op = i.operands[0]
            if op.type == X86_OP_IMM:
                ins.branch_target = op.imm
            elif ins.is_call or ins.is_uncond_jmp:
                ins.is_indirect = True
        for op in i.operands:
            if op.type == X86_OP_MEM:
                try:
                    if md.reg_name(op.mem.base) == "rip":
                        ref = i.address + i.size + op.mem.disp
                        if ins.rip_ref is None:
                            ins.rip_ref = ref
                except Exception:
                    pass
        if ins.rip_ref is not None and ins.rip_ref in imports:
            ins.import_name = imports[ins.rip_ref]
        elif (ins.branch_target is not None and ins.branch_target in imports
                and not ins.is_indirect):
            ins.import_name = imports[ins.branch_target]
        if (ins.is_uncond_jmp or ins.is_call) and ins.is_indirect and ins.rip_ref is not None:
            ins.is_import_jmp = ins.rip_ref in imports
        if ins.is_uncond_jmp and ins.is_indirect:
            memop = next((o for o in i.operands if o.type == X86_OP_MEM), None)
            if memop is not None:
                try:
                    base = md.reg_name(memop.mem.base) if memop.mem.base else None
                except Exception:
                    base = None
                if base != "rip" and memop.mem.index != 0 \
                        and (memop.mem.scale or 1) in (2, 4, 8):
                    ins.is_table_jmp = True
                    if memop.mem.disp:
                        ins.table_va = memop.mem.disp & 0xFFFFFFFFFFFFFFFF
        if ins.is_call and ins.import_name:
            low = ins.import_name.lower()
            if any(s in low for s in NORETURN_SUBSTR):
                ins.is_noreturn = True
        return ins

    def read_table(table_va: int, scale: int, cand_fn, lo=None, hi=None,
                   signed: bool = False) -> list:
        targets = []
        va = table_va
        for _ in range(MAX_TABLE_ENTRIES):
            b = read_at(va, scale)
            if b is None:
                break
            val = int.from_bytes(b, "little", signed=signed and scale < 8)
            hit = None
            for c in cand_fn(val):
                if not in_exec(c):
                    continue
                if lo is not None and not (lo <= c < hi):
                    return targets
                det, _, _, _ = decode_one(c)
                if det is not None:
                    hit = c
                    break
            if hit is None:
                break
            if hit not in targets:
                targets.append(hit)
            va += scale
        return targets

    func_begins = [b for b, _ in func_bounds]

    def table_range(va: int):
        i = bisect_right(func_begins, va) - 1
        if i >= 0 and va < func_bounds[i][1]:
            return func_bounds[i]
        return (None, None)

    def resolve_table(ins: Insn, line):
        det = _deco(ins)
        if det is None:
            return None
        memop = next((op.mem for op in det.operands if op.type == X86_OP_MEM), None)
        if memop is None:
            return None
        scale = memop.scale or 1
        if scale not in (2, 4, 8):
            return None
        try:
            base = md.reg_name(memop.base) if memop.base else None
        except Exception:
            base = None
        if base == "rip":
            table_va = ins.rip_ref
        elif base is None:
            table_va = memop.disp & 0xFFFFFFFFFFFFFFFF if memop.disp else None
        else:
            bv = _reg_value(line_lookup(line), parent(base), ins.addr)
            if bv is None:
                return None
            table_va = bv + memop.disp
        if table_va is None:
            return None
        tv = table_va

        def cand_fn(val):
            return (val,) if scale == 8 else (imagebase + val, tv + val)

        lo, hi = table_range(ins.addr)
        tgts = read_table(tv, scale, cand_fn, lo, hi)
        return (table_va, tgts) if tgts else None

    def _deco(ins: Insn):
        try:
            return next(md.disasm(ins.raw, ins.addr))
        except StopIteration:
            return None

    def _writes_of(ins: Insn) -> set:
        ws = {parent(w) for w in ins.regs_write}
        d = _deco(ins)
        if d is not None:
            for op in d.operands:
                if op.type == X86_OP_REG and (op.access & CS_AC_WRITE):
                    try:
                        ws.add(parent(md.reg_name(op.reg)))
                    except Exception:
                        pass
        return ws

    def line_lookup(line):
        idx = {i.addr: k for k, i in enumerate(line)}

        def lookup(reg: str, before_addr: int):
            k = idx.get(before_addr)
            if k is None:
                return None
            for j in range(k - 1, max(-1, k - 65), -1):
                i = line[j]
                if i.is_call or i.is_ret:
                    return None
                if reg in _writes_of(i):
                    return i
            return None

        return lookup

    def _reg_value(lookup, reg: str, before_addr: int, depth: int = 0):
        if depth > 4:
            return None
        di = lookup(reg, before_addr)
        if di is None:
            return None
        d = _deco(di)
        if d is None or len(d.operands) < 2:
            return None
        o0, o1 = d.operands[0], d.operands[1]
        if o0.type != X86_OP_REG:
            return None
        if di.mnemonic == "lea":
            mem = o1.mem if o1.type == X86_OP_MEM else None
            if mem is None:
                return None
            try:
                if mem.base == 0 or md.reg_name(mem.base) == "rip":
                    return di.rip_ref
                bv = _reg_value(lookup, parent(md.reg_name(mem.base)), di.addr,
                                depth + 1)
            except Exception:
                return None
            return None if bv is None else bv + mem.disp
        if di.mnemonic in ("mov", "movabs") and o1.type == X86_OP_IMM:
            return o1.imm
        if di.mnemonic == "mov" and o1.type == X86_OP_REG:
            try:
                return _reg_value(lookup, parent(md.reg_name(o1.reg)), di.addr,
                                  depth + 1)
            except Exception:
                return None
        if di.mnemonic == "mov" and o1.type == X86_OP_MEM:
            try:
                if o1.mem.base and md.reg_name(o1.mem.base) == "rip" \
                        and di.rip_ref is not None:
                    b = read_at(di.rip_ref, 8)
                    if b is not None:
                        return int.from_bytes(b, "little")
            except Exception:
                pass
            return None
        if di.mnemonic == "xor" and o1.type == X86_OP_REG:
            try:
                if parent(md.reg_name(o0.reg)) == parent(md.reg_name(o1.reg)):
                    return 0
            except Exception:
                pass
        return None

    def resolve_reg_jmp(ins: Insn, lookup):
        d = _deco(ins)
        if d is None or not d.operands or d.operands[0].type != X86_OP_REG:
            return None
        try:
            dst = parent(md.reg_name(d.operands[0].reg))
        except Exception:
            return None
        di = lookup(dst, ins.addr)
        if di is None:
            return None
        ddi = _deco(di)
        if ddi is None:
            return None
        addend_val = None
        load = None
        if di.mnemonic == "add" and len(ddi.operands) >= 2 \
                and ddi.operands[1].type == X86_OP_REG:
            try:
                addend = parent(md.reg_name(ddi.operands[1].reg))
            except Exception:
                return None
            addend_val = _reg_value(lookup, addend, di.addr)
            load = lookup(dst, di.addr)
            if load is None:
                return None
        else:
            load = di
        if load.mnemonic not in ("mov", "movzx", "movsxd", "movsx"):
            return None
        dl = _deco(load)
        if dl is None:
            return None
        memop = next((op.mem for op in dl.operands if op.type == X86_OP_MEM), None)
        if memop is None:
            return None
        scale = memop.scale or 1
        if scale not in (2, 4, 8):
            return None
        signed = load.mnemonic in ("movsxd", "movsx")
        try:
            base = md.reg_name(memop.base) if memop.base else None
        except Exception:
            base = None
        if base == "rip":
            table_va = load.rip_ref
        elif base is not None:
            bv = _reg_value(lookup, parent(base), load.addr)
            if bv is None:
                return None
            table_va = bv + memop.disp
        else:
            table_va = memop.disp
        if table_va is None:
            return None
        if addend_val is not None:
            av = addend_val

            def cand_fn(val):
                return (av + val,)
        else:
            tv = table_va

            def cand_fn(val):
                return (val,) if scale == 8 else (imagebase + val, tv + val)

        lo, hi = table_range(ins.addr)
        tgts = read_table(table_va, scale, cand_fn, lo, hi, signed)
        return (table_va, tgts) if tgts else None

    insns, by_addr = [], {}
    visited = set()
    jump_tables = {}
    deferred = []
    seeds = {entry_va}
    seeds.update(b for b, _ in func_bounds)
    worklist = deque(a for a in sorted(seeds) if in_exec(a))

    def _apply_table(ins, res):
        tva, tgts = res
        ins.is_table_jmp = True
        ins.table_targets = tuple(tgts)
        jump_tables[ins.addr] = (tva, tuple(tgts))
        ins.table_va = None
        for t in tgts:
            if t not in visited:
                worklist.append(t)

    def _trace_from(start):
        cur = start
        line = []
        while True:
            if cur in visited or not in_exec(cur):
                return
            det, va_start, raw_ptr, sec_name = decode_one(cur)
            if det is None:
                return
            file_off = raw_ptr + (cur - va_start)
            if file_off < 0 or file_off + det.size > file_size:
                return
            ins = make_insn(det, va_start, raw_ptr, sec_name)
            visited.add(cur)
            insns.append(ins)
            by_addr[cur] = ins
            line.append(ins)

            if ins.is_uncond_jmp:
                det_j = _deco(ins)
                op0 = det_j.operands[0] if det_j is not None and det_j.operands else None
                if op0 is not None and op0.type == X86_OP_IMM:
                    if ins.branch_target is not None and ins.branch_target not in visited:
                        worklist.append(ins.branch_target)
                elif op0 is not None and op0.type == X86_OP_MEM:
                    if ins.is_table_jmp:
                        res = resolve_table(ins, line)
                        if res is not None:
                            _apply_table(ins, res)
                elif op0 is not None and op0.type == X86_OP_REG:
                    res = resolve_reg_jmp(ins, line_lookup(line))
                    if res is not None:
                        _apply_table(ins, res)
                    else:
                        deferred.append(ins)
                return
            if ins.is_ret:
                return
            if ins.is_padding and ins.mnemonic in ("int3", "hlt"):
                return
            if ins.mnemonic in ("ud0", "ud1", "ud2"):
                return
            if ins.is_cond_jmp or ins.is_loop:
                if ins.branch_target is not None and ins.branch_target not in visited:
                    worklist.append(ins.branch_target)
                cur += ins.size
                continue
            if ins.is_call:
                if ins.branch_target is not None and ins.branch_target not in visited:
                    worklist.append(ins.branch_target)
                if ins.is_noreturn:
                    return
                cur += ins.size
                continue
            cur += ins.size

    while worklist:
        _trace_from(worklist.popleft())

    pred_map = {}
    for ins in insns:
        nxt = ins.addr + ins.size
        if nxt in by_addr:
            pred_map.setdefault(nxt, set()).add(ins.addr)
        if ins.branch_target is not None and ins.branch_target in by_addr:
            pred_map.setdefault(ins.branch_target, set()).add(ins.addr)

    while deferred:
        ins = deferred.pop()

        def cross_lookup(reg: str, before_addr: int):
            seen = set()
            stack = [before_addr]
            n = 0
            while stack and n < 512:
                a = stack.pop()
                if a in seen:
                    continue
                seen.add(a)
                n += 1
                for p in pred_map.get(a, ()):
                    if p in seen:
                        continue
                    pi = by_addr.get(p)
                    if pi is None:
                        continue
                    if reg in _writes_of(pi):
                        return pi
                    if pi.is_call or pi.is_ret:
                        continue
                    stack.append(p)
            return None

        res = resolve_reg_jmp(ins, cross_lookup)
        if res is not None:
            _apply_table(ins, res)
            while worklist:
                _trace_from(worklist.popleft())

    insns.sort(key=lambda x: x.addr)
    blocks = _build_blocks(insns, by_addr, split_on_call=split_on_call)

    covered = []
    for ins in insns:
        s, e = ins.addr, ins.addr + ins.size
        if covered and s <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], e)
        else:
            covered.append([s, e])
    gaps = []
    for va_start, _raw_ptr, code, _name in exec_ranges:
        e = va_start + len(code)
        rel = sorted((max(a, va_start), min(b, e))
                     for a, b in covered if b > va_start and a < e)
        pos = va_start
        for a, b in rel:
            if a > pos:
                gaps.append((pos, a))
            pos = max(pos, b)
            if pos >= e:
                break
        if pos < e:
            gaps.append((pos, e))
    gaps.sort()

    return DisasmResult(path=path, imagebase=imagebase, entry_va=entry_va,
                        sections=sections, insns=insns, by_addr=by_addr,
                        blocks=blocks, imports=imports, relocs=relocs,
                        reloc_types=reloc_types,
                        exec_ranges=[(v, v + len(c)) for v, _, c, _ in exec_ranges],
                        gaps=gaps, func_bounds=func_bounds,
                        jump_tables=jump_tables,
                        is_dll=is_dll, dynamic_base=dynamic_base,
                        has_reloc_dir=has_reloc_dir, file_size=file_size)


def _block_ends(ins, split_on_call: bool) -> bool:
    if ins.is_ret or ins.is_uncond_jmp or ins.is_cond_jmp or ins.is_loop:
        return True
    if ins.is_call and split_on_call:
        return True
    return False


def _build_blocks(insns, by_addr, split_on_call: bool = True):
    def is_branch(ins):
        return ins.is_cond_jmp or ins.is_uncond_jmp or ins.is_call or ins.is_loop

    starts = set()
    if insns:
        starts.add(insns[0].addr)
    for ins in insns:
        if is_branch(ins):
            if ins.branch_target is not None and ins.branch_target in by_addr:
                starts.add(ins.branch_target)
            for t in ins.table_targets:
                if t in by_addr:
                    starts.add(t)
    for k, ins in enumerate(insns):
        if ins.is_cond_jmp or ins.is_uncond_jmp or ins.is_ret or ins.is_loop \
                or (ins.is_call and split_on_call):
            if k + 1 < len(insns):
                starts.add(insns[k + 1].addr)
    blocks, cur = [], None
    for ins in insns:
        if ins.addr in starts or cur is None:
            if cur:
                blocks.append(cur)
            cur = BasicBlock(start=ins.addr)
        cur.insns.append(ins)
        if _block_ends(ins, split_on_call):
            blocks.append(cur)
            cur = None
    if cur:
        blocks.append(cur)
    for b in blocks:
        if b.insns and all(i.is_padding for i in b.insns):
            b.is_padding = True
    by_start = {b.start: b for b in blocks}
    for bi, b in enumerate(blocks):
        last = b.insns[-1]
        nxt_start = blocks[bi + 1].start if bi + 1 < len(blocks) else None
        if last.is_ret:
            b.kind = "ret"
        elif last.is_uncond_jmp:
            if last.is_indirect:
                b.kind = "indirect"
                for t in last.table_targets:
                    if t in by_start:
                        b.succs.append(t)
            else:
                b.kind = "uncond"
                if last.branch_target is not None and last.branch_target in by_start:
                    b.succs.append(last.branch_target)
        elif last.is_cond_jmp or last.is_loop:
            b.kind = "loop" if last.is_loop else "cond"
            if last.branch_target is not None and last.branch_target in by_start:
                b.succs.append(last.branch_target)
            if nxt_start is not None and last.addr + last.size == nxt_start:
                b.succs.append(nxt_start)
        elif last.is_call:
            if last.is_noreturn:
                b.kind = "call_noreturn"
                if last.is_indirect:
                    b.calls.append(last.import_name or None)
                elif last.branch_target is not None:
                    b.calls.append(last.branch_target)
                else:
                    b.calls.append(None)
                b.call_names.append(last.import_name)
            else:
                b.kind = "call"
                if last.is_indirect:
                    b.calls.append(last.import_name or None)
                    b.call_names.append(last.import_name)
                elif last.branch_target is not None:
                    b.calls.append(last.branch_target)
                    b.call_names.append(last.import_name)
                if nxt_start is not None and last.addr + last.size == nxt_start:
                    b.succs.append(nxt_start)
        else:
            b.kind = "fall"
            if nxt_start is not None and last.addr + last.size == nxt_start:
                b.succs.append(nxt_start)
        if not split_on_call:
            calls, names = [], []
            for i in b.insns:
                if not i.is_call:
                    continue
                if i.is_indirect:
                    calls.append(i.import_name or None)
                elif i.branch_target is not None:
                    calls.append(i.branch_target)
                else:
                    calls.append(None)
                names.append(i.import_name)
            b.calls = calls
            b.call_names = names
            if any(i.is_noreturn for i in b.insns):
                b.kind = "call_noreturn"
                b.succs = []
    return blocks


def verify(r: DisasmResult):
    """Инварианты модуля 1. errors = битая структура (должно быть пусто).
    warnings = подозрительные места (фантомные call-цели, перекрытия инструкций)."""
    errors, warnings = [], []
    starts = {b.start for b in r.blocks}
    for b in r.blocks:
        for s in b.succs:
            if s not in starts:
                errors.append(f"block {b.start:#x}: succ {s:#x} not a block start")
        for s in b.calls:
            if isinstance(s, int) and s not in r.by_addr:
                warnings.append(f"block {b.start:#x}: call target {s:#x} not disassembled")
        if b.kind in ("call",) and b.calls and b.calls[0] in b.succs:
            errors.append(f"block {b.start:#x}: callee leaked into succs")
        if not b.insns:
            errors.append(f"block {b.start:#x}: empty")
    prev = None
    for i in r.insns:
        if prev is not None and prev.addr + prev.size > i.addr:
            warnings.append(f"insn {i.addr:#x}: overlaps insn {prev.addr:#x}")
        prev = i
        if i.file_off < 0 or i.file_off + i.size > r.file_size:
            errors.append(f"insn {i.addr:#x}: file_off out of range")
    fi = 0
    n_ins = len(r.insns)
    for fb, fe in r.func_bounds:
        while fi < n_ins and r.insns[fi].addr + r.insns[fi].size <= fb:
            fi += 1
        k = fi
        covered = 0
        while k < n_ins and r.insns[k].addr < fe:
            covered += r.insns[k].size
            k += 1
        if fe - fb >= 16 and covered * 2 < fe - fb:
            warnings.append(f"func {fb:#x}-{fe:#x}: covered {covered}/{fe - fb} bytes "
                            f"(unresolved jump table / indirect entry?)")
    return errors, warnings


def summary(r: DisasmResult) -> str:
    lines = [f"PE: {r.path} base={r.imagebase:#x} entry={r.entry_va:#x}",
             f"dll={r.is_dll} dynamic_base={r.dynamic_base} has_reloc={r.has_reloc_dir}",
             f"sections: {len(r.sections)} insns: {len(r.insns)} blocks: {len(r.blocks)}",
             f"imports: {len(r.imports)} relocs: {len(r.relocs)}"]
    n_call = sum(1 for i in r.insns if i.is_call)
    n_cj = sum(1 for i in r.insns if i.is_cond_jmp)
    n_uj = sum(1 for i in r.insns if i.is_uncond_jmp)
    n_ret = sum(1 for i in r.insns if i.is_ret)
    n_rip = sum(1 for i in r.insns if i.rip_ref is not None)
    n_ind = sum(1 for i in r.insns if i.is_indirect)
    n_tab = sum(1 for i in r.insns if i.is_table_jmp)
    n_imp = sum(1 for i in r.insns if i.import_name and (i.is_call or i.is_uncond_jmp))
    n_nor = sum(1 for i in r.insns if i.is_noreturn)
    n_loop = sum(1 for i in r.insns if i.is_loop)
    n_pad = sum(1 for b in r.blocks if b.is_padding)
    gap_bytes = sum(e - s for s, e in r.gaps)
    lines.append(f"call={n_call} condjmp={n_cj} uncondjmp={n_uj} ret={n_ret} rip_ref={n_rip} "
                 f"indirect={n_ind} tablejmp={n_tab} import_ref={n_imp} noreturn={n_nor} "
                 f"loop={n_loop} padblocks={n_pad} gaps={len(r.gaps)} gap_bytes={gap_bytes} "
                 f"tables_resolved={len(r.jump_tables)}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=["bin/test.exe"])
    ap.add_argument("--ir-blocks", action="store_true",
                    help="классические basic blocks для IR (без сплита на call)")
    args = ap.parse_args()
    for p in args.files:
        r = disassemble(p, split_on_call=not args.ir_blocks)
        print(summary(r))
        errs, warns = verify(r)
        for e in errs:
            print("  VERIFY ERROR:", e)
        for w in warns[:5]:
            print("  VERIFY WARN:", w)
        if len(warns) > 5:
            print(f"  ... +{len(warns) - 5} warns")
        print(f"  verify: {'ok' if not errs else 'FAIL'} ({len(warns)} warns)")
        for b in r.blocks[:5]:
            last = b.insns[-1]
            print(f"  block {b.start:#x} n={len(b.insns)} kind={b.kind} "
                  f"last={last.mnemonic} {last.op_str} succs={[hex(s) for s in b.succs]}")
