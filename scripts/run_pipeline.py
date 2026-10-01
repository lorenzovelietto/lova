"""End-to-end pipeline runner: disassemble -> lift -> mutate -> fixup/verify.

Запускает все четыре модуля последовательно и печатает сводный отчёт.
Пайплайн намеренно не параллельный: каждый шаг зависит от предыдущего
(Mod2 нужен Mod1-результат, Mod3 нужен Mod2-liveness, Mod4 нужны оба PE).

CLI:
    python scripts/run_pipeline.py <orig.exe> [-o <mutated.exe>]
                                   [--apply-checksum] [--keep-mutated]
"""
import argparse
import os
import sys
import tempfile

# Разрешаем запуск как `python scripts/run_pipeline.py ...` без установки.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(HERE, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from mod1_disasm import disassemble, verify   # noqa: E402
from mod2_ir import lift                      # noqa: E402
from mod3_morph import mutate_pe              # noqa: E402
from mod4_fixup import fixup, apply_checksum  # noqa: E402
from pipeline_config import PipelineConfig    # noqa: E402


def _print_mod1(dis):
    n_call = sum(1 for i in dis.insns if i.is_call)
    n_ret = sum(1 for i in dis.insns if i.is_ret)
    n_tab = sum(1 for i in dis.insns if i.is_table_jmp)
    print(f"[mod1] insns={len(dis.insns)} blocks={len(dis.blocks)} "
          f"calls={n_call} rets={n_ret} tables={n_tab} "
          f"gaps={len(dis.gaps)}")


def _print_mod2(mod):
    print(f"[mod2] effects={len(mod.effects)} functions={len(mod.functions)} "
          f"vfp_degraded={mod.vfp_degraded}")


def _print_mod3(stats):
    applied = stats.get("applied", {})
    blocked = stats.get("blocked", {})
    total_a = sum(applied.values())
    total_b = sum(blocked.values())
    print(f"[mod3] applied={total_a} blocked={total_b} rules={len(applied)}")


def _print_mod4(rep):
    st = rep.get("stats", {})
    errs = rep.get("errors", [])
    warns = rep.get("warns", [])
    print(f"[mod4] changed_bytes={st.get('changed_bytes')} "
          f"mutated_insns={st.get('mutated_insns')} "
          f"verify_warns={st.get('verify_warns')} "
          f"errors={len(errs)} warns={len(warns)}")
    for e in errs:
        print(f"  ERR: {e}")


def run(orig: str, mutated: str | None, apply_checksum_flag: bool,
        keep_mutated: bool, split_on_call: bool = True) -> int:
    cleanup_mut = False
    if mutated is None:
        fd, mutated = tempfile.mkstemp(suffix=".exe", prefix="lova_mut_")
        os.close(fd)
        cleanup_mut = not keep_mutated

    cfg = PipelineConfig(split_on_call=split_on_call)
    print(f"orig: {orig}")
    print(f"mutated: {mutated}")
    print(f"config: split_on_call={cfg.split_on_call}")

    dis = disassemble(orig, cfg=cfg)
    _print_mod1(dis)
    errs, warns = verify(dis)
    if errs:
        print(f"[mod1-verify] {len(errs)} ERRORS — abort")
        for e in errs[:10]:
            print(f"  ERR: {e}")
        return 1
    if warns:
        print(f"[mod1-verify] {len(warns)} warnings")
        for w in warns[:5]:
            print(f"  {w}")

    mod = lift(orig, cfg=cfg, dis=dis)
    _print_mod2(mod)

    stats = mutate_pe(orig, mutated, cfg=cfg, dis=dis, mod=mod)
    _print_mod3(stats)

    rep = fixup(orig, mutated, cfg=cfg)
    _print_mod4(rep)
    rc = 0 if not rep.get("errors") else 1

    if rc == 0 and apply_checksum_flag and rep.get("checksum_fix"):
        apply_checksum(mutated, rep["checksum_fix"])
        print(f"[mod4] checksum fixed in place: {mutated}")

    if cleanup_mut and os.path.exists(mutated):
        os.remove(mutated)
        print(f"[cleanup] removed temp mutated: {mutated}")

    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("orig", help="Original PE path")
    ap.add_argument("-o", "--out", default=None,
                    help="Mutated PE output path (default: temp file)")
    ap.add_argument("--apply-checksum", action="store_true",
                    help="Patch PE OptionalHeader checksum after fixup")
    ap.add_argument("--keep-mutated", action="store_true",
                    help="Do not delete the temp mutated PE on success")
    ap.add_argument("--ir-blocks", action="store_true",
                    help="Классические basic blocks (split_on_call=False) во всех модулях")
    args = ap.parse_args(argv)
    return run(args.orig, args.out, args.apply_checksum, args.keep_mutated,
               split_on_call=not args.ir_blocks)


if __name__ == "__main__":
    sys.exit(main())
