"""Unit tests that do not require a real PE.

Helpers in pipeline_config have no third-party deps. Tests that import
mod1/mod2/mod4 skip if capstone/pefile are missing.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
# Local sandbox layout (flat) plus repo layout (src/).
for p in (SRC, ROOT, os.path.join(ROOT, "lova")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from pipeline_config import PipelineConfig  # noqa: E402


class TestPipelineConfig(unittest.TestCase):
    def test_default(self):
        cfg = PipelineConfig()
        self.assertTrue(cfg.split_on_call)
        with self.assertRaises(Exception):
            cfg.split_on_call = False  # type: ignore[misc]

    def test_from_legacy_none(self):
        cfg = PipelineConfig.from_legacy()
        self.assertEqual(cfg, PipelineConfig(split_on_call=True))

    def test_from_legacy_bool(self):
        cfg = PipelineConfig.from_legacy(False)
        self.assertFalse(cfg.split_on_call)

    def test_mismatch_raises(self):
        cfg = PipelineConfig(split_on_call=True)
        with self.assertRaises(ValueError):
            PipelineConfig.from_legacy(False, cfg)

    def test_matching_legacy_ok(self):
        cfg = PipelineConfig(split_on_call=False)
        self.assertIs(PipelineConfig.from_legacy(False, cfg), cfg)


try:
    from mod1_disasm import (
        Insn, DisasmResult, parent, verify, _build_blocks,
        decode_jump_table_entry, jump_table_entry_is_signed,
    )
    HAS_MOD1 = True
except ImportError:
    HAS_MOD1 = False


def _insn(addr, size=1, **kw):
    mnemonic = kw.pop("mnemonic", "nop")
    op_str = kw.pop("op_str", "")
    return Insn(addr=addr, file_off=addr, size=size, raw=b"\x90" * size,
                mnemonic=mnemonic, op_str=op_str, **kw)


@unittest.skipUnless(HAS_MOD1, "capstone/pefile not installed")
class TestMod1Helpers(unittest.TestCase):
    def test_parent_gpr(self):
        self.assertEqual(parent("eax"), "rax")
        self.assertEqual(parent("r8d"), "r8")
        self.assertEqual(parent("sil"), "rsi")
        self.assertEqual(parent("xmm0"), "xmm0")

    def test_jump_table_signedness(self):
        self.assertTrue(jump_table_entry_is_signed(4))
        self.assertTrue(jump_table_entry_is_signed(2))
        self.assertFalse(jump_table_entry_is_signed(8))

    def test_decode_negative_int32(self):
        raw = (0xFFFFFF80).to_bytes(4, "little")
        self.assertEqual(decode_jump_table_entry(raw, signed=False), 0xFFFFFF80)
        self.assertEqual(decode_jump_table_entry(raw, signed=True), -128)

    def test_build_blocks_splits_on_address_gap(self):
        insns = [_insn(0x1000), _insn(0x1001), _insn(0x2000)]
        by_addr = {i.addr: i for i in insns}
        blocks = _build_blocks(insns, by_addr)
        self.assertIn(0x1000, {b.start for b in blocks})
        self.assertIn(0x2000, {b.start for b in blocks})

    def test_build_blocks_splits_on_pdata_begin(self):
        insns = [_insn(0x1000), _insn(0x1001), _insn(0x1002)]
        by_addr = {i.addr: i for i in insns}
        blocks = _build_blocks(insns, by_addr, func_bounds=[(0x1000, 0x1002),
                                                            (0x1002, 0x1003)])
        self.assertIn(0x1002, {b.start for b in blocks})

    def test_overlap_is_error(self):
        insns = [_insn(0x1000, 3), _insn(0x1001, 1)]
        r = DisasmResult(
            path="", imagebase=0, entry_va=0, sections=[],
            insns=insns, by_addr={i.addr: i for i in insns}, blocks=[],
            imports={}, relocs=set(), file_size=0x2000,
        )
        errs, _warns = verify(r)
        self.assertTrue(any("overlaps" in e for e in errs))


try:
    from mod2_ir import (
        IREffect, IRModule, RET_LIVE, NON_VOLATILE_VEC,
        recompute_live_after_prefix, _step_live_backward,
    )
    HAS_MOD2 = True
except ImportError:
    HAS_MOD2 = False


@unittest.skipUnless(HAS_MOD2 and HAS_MOD1, "capstone/pefile not installed")
class TestMod2Liveness(unittest.TestCase):
    def test_xmm6_15_live_at_ret(self):
        for i in range(6, 16):
            self.assertIn(f"xmm{i}", RET_LIVE)
        self.assertEqual(NON_VOLATILE_VEC, frozenset(f"xmm{i}" for i in range(6, 16)))
        self.assertNotIn("xmm1", RET_LIVE)  # volatile except xmm0 return

    def test_recompute_prefix_sees_new_def(self):
        a = _insn(0x1000)
        b = _insn(0x1001)
        from mod1_disasm import BasicBlock
        block = BasicBlock(start=0x1000, insns=[a, b])
        mod = IRModule(path="")
        mod.effects[a.addr] = IREffect(addr=a.addr)
        # Originally B does not def rax, so rax is live after A.
        mod.effects[b.addr] = IREffect(addr=b.addr)
        mod.live_after[a.addr] = {"rax"}
        mod.live_after[b.addr] = set()
        # Mutate B: now it defs rax, so rax should die before B.
        mod.effects[b.addr] = IREffect(addr=b.addr, reg_def=frozenset({"rax"}))
        recompute_live_after_prefix(block, mod, mutated_idx=1)
        self.assertNotIn("rax", mod.live_after[a.addr])

    def test_step_live_mstar_does_not_kill(self):
        live = {"m:*", "rax"}
        e = IREffect(addr=0, vfp_def=frozenset({"m:*"}), reg_def=frozenset({"rax"}))
        out = _step_live_backward(live, e)
        self.assertIn("m:*", out)
        self.assertNotIn("rax", out)


try:
    from mod4_fixup import diff_offsets
    HAS_MOD4 = True
except ImportError:
    HAS_MOD4 = False


@unittest.skipUnless(HAS_MOD4, "capstone/pefile not installed")
class TestMod4Diff(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(diff_offsets(b"abc", b"abc"), [])

    def test_single_byte(self):
        self.assertEqual(diff_offsets(b"abc", b"aXc"), [1])

    def test_multi(self):
        self.assertEqual(diff_offsets(b"aaaa", b"baab"), [0, 3])


if __name__ == "__main__":
    unittest.main()
