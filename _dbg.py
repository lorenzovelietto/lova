import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from mod1_disasm import disassemble, verify

r = disassemble("bin/switch2.exe")
errs, warns = verify(r)
print("tables_resolved:", len(r.jump_tables))
for ia, (tva, tgts) in sorted(r.jump_tables.items()):
    if 0x140001000 < ia < 0x140002000:
        print(f"jmp {ia:#x} -> table {tva:#x}, targets={len(tgts)} (unique)")
        print("  ", [hex(t) for t in tgts])
print("verify errors:", len(errs), "warnings:", len(warns))
for w in warns[:5]:
    print("  WARN:", w)
