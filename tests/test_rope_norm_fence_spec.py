#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec regression tests for the v0.7.27 qwen3 / qwen3_next rope-norm
fold patches.

The fold must run on plain single-request decode (Python int
cache.offset) and fall through to the composed path on batched
(mx.array cache.offset) or multi-request (B>1) inputs. The full
behavior test is the live 4-concurrent batch on the M2 (see
scripts/test_rope_norm_fence_smoke.sh); this file pins the
SPECIFIC contract the patcher must honor in source:

  1. Each patcher reads its OWN kill env (qwen3: MLX_OMARCHY_QWEN3_
     ROPE_NORM_FUSE; qwen3_next: MLX_OMARCHY_QWEN3_NEXT_ROPE_NORM_FUSE)
     and falls back to the shared MLX_OMARCHY_ROPE_NORM_FUSE.
  2. Each patcher gates the fold on `cache is None or
     isinstance(cache.offset, int)` -- mx.array offsets come from
     BatchGenerator even at B == 1, and mx.fast.rope_rms_norm needs
     a Python int (v0.7.27 root cause).
  3. Each patcher keeps the gate `and B == 1` (multi-request
     broadcast on a different axis).
  4. The else branch (composed chain) is preserved.

Without these, a future change to either patcher defaulting the
fold to ON without the offset-type fence would re-introduce the
v0.7.27 [broadcast_shapes] cache corruption without the test
suite noticing.
"""
import os
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN3_PATCHER = REPO_ROOT / "scripts" / "patch-mlx-lm-qwen3-rope-norm.py"
QWEN3_NEXT_PATCHER = REPO_ROOT / "scripts" / "patch-mlx-lm-rope-norm.py"


def _norm_new(patcher_path: Path) -> str:
    src = patcher_path.read_text()
    i = src.find('NORM_NEW = """')
    assert i >= 0, f"NORM_NEW not found in {patcher_path}"
    j = src.find('"""', i + len('NORM_NEW = """'))
    return src[i + len('NORM_NEW = """'):j]


@unittest.skipUnless(QWEN3_PATCHER.exists() and QWEN3_NEXT_PATCHER.exists(),
                     "patchers not in expected repo location")
class RopeNormFenceSpecTests(unittest.TestCase):

    def setUp(self):
        self.qwen3 = _norm_new(QWEN3_PATCHER)
        self.qwen3_next = _norm_new(QWEN3_NEXT_PATCHER)

    def test_qwen3_uses_own_kill_env_with_shared_fallback(self):
        # MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE first, with a nested
        # MLX_OMARCHY_ROPE_NORM_FUSE as the fallback.
        self.assertIn("MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE", self.qwen3,
            msg="qwen3 patcher must read its own kill env")
        self.assertRegex(self.qwen3,
            re.compile(r"MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE.*?MLX_OMARCHY_ROPE_NORM_FUSE", re.S),
            msg="qwen3 patcher must read MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE with "
                "MLX_OMARCHY_ROPE_NORM_FUSE as the shared fallback")
        # Default for the inner shared env is "1" (fold on by default).
        self.assertRegex(self.qwen3, r'MLX_OMARCHY_ROPE_NORM_FUSE",\s*"1"',
            msg="shared kill env default must remain on (1) for the decode "
                "perf win")

    def test_qwen3_next_uses_own_kill_env_with_shared_fallback(self):
        self.assertIn("MLX_OMARCHY_QWEN3_NEXT_ROPE_NORM_FUSE", self.qwen3_next,
            msg="qwen3_next patcher must read its own kill env")
        self.assertRegex(self.qwen3_next,
            re.compile(r"MLX_OMARCHY_QWEN3_NEXT_ROPE_NORM_FUSE.*?MLX_OMARCHY_ROPE_NORM_FUSE", re.S),
            msg="qwen3_next patcher must read MLX_OMARCHY_QWEN3_NEXT_ROPE_NORM_FUSE "
                "with MLX_OMARCHY_ROPE_NORM_FUSE as the shared fallback")
        self.assertRegex(self.qwen3_next, r'MLX_OMARCHY_ROPE_NORM_FUSE",\s*"1"',
            msg="shared kill env default must remain on (1)")

    def test_qwen3_offset_type_fence_present(self):
        # The v0.7.27 root cause: cache.offset is an mx.array in
        # BatchGenerator; the kernel requires a Python int.
        self.assertRegex(self.qwen3,
            r"isinstance\(cache\.offset, int\)",
            msg="qwen3 patcher must gate the fold on "
                "isinstance(cache.offset, int)")

    def test_qwen3_next_offset_type_fence_present(self):
        self.assertRegex(self.qwen3_next,
            r"isinstance\(cache\.offset, int\)",
            msg="qwen3_next patcher must gate the fold on "
                "isinstance(cache.offset, int)")

    def test_qwen3_B_eq_1_fence_present(self):
        self.assertRegex(self.qwen3, r"and B == 1",
            msg="qwen3 patcher must gate the fold on B == 1")

    def test_qwen3_next_B_eq_1_fence_present(self):
        self.assertRegex(self.qwen3_next, r"and B == 1",
            msg="qwen3_next patcher must gate the fold on B == 1")

    def test_qwen3_preserves_composed_fallback(self):
        # The else branch (the bit-identical composed chain) must
        # still call q_norm, k_norm, transpose, and rope. A future
        # change that drops the composed path under the gate would
        # silently break correctness for batched paths.
        for needle in (
            "self.q_norm(queries",
            "self.k_norm(",
            ".transpose(",
            "self.rope(queries, offset=cache.offset)",
            "self.rope(keys, offset=cache.offset)",
        ):
            self.assertIn(needle, self.qwen3,
                msg=f"qwen3 patcher must keep {needle!r} in the "
                    f"composed-fallback else branch")

    def test_qwen3_next_preserves_composed_fallback(self):
        for needle in (
            "self.q_norm(queries",
            "self.k_norm(",
            ".transpose(",
            "self.rope(queries, offset=cache.offset)",
            "self.rope(keys, offset=cache.offset)",
        ):
            self.assertIn(needle, self.qwen3_next,
                msg=f"qwen3_next patcher must keep {needle!r} in the "
                    f"composed-fallback else branch")


if __name__ == "__main__":
    unittest.main()
