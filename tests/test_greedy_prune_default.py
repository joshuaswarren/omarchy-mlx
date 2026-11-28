# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavioral tests for greedy-head default gating in mlx-lm-greedy-prune patches."""

import ast
import os
import pathlib
import re
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
DEVICE_H = REPO / "overlay" / "mlx" / "backend" / "omarchy" / "device.h"
PATCH_PATHS = (
    REPO / "patches" / "mlx-lm-greedy-prune.patch",
    REPO / "patches" / "mlx-lm-0.32" / "mlx-lm-greedy-prune.patch",
)
BATCH_PATCHER = REPO / "scripts" / "patch-mlx-lm-batch-greedy-head.py"


def _extract_greedy_head_source(patch_path: pathlib.Path) -> str:
    """Return the added _greedy_head function source from hunk 1 of patch_path."""
    lines = patch_path.read_text(encoding="utf-8").splitlines()
    start = lines.index("+def _greedy_head(model):")
    collected = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        collected.append(line[1:])
    src = "\n".join(collected).rstrip() + "\n"
    ast.parse(src)
    return src


def _load_batch_head_old() -> str:
    tree = ast.parse(BATCH_PATCHER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "HEAD_OLD":
                    val = ast.literal_eval(node.value)
                    assert isinstance(val, str)
                    return val
    raise AssertionError("HEAD_OLD not found in patch-mlx-lm-batch-greedy-head.py")


class GreedyPruneDefaultTests(unittest.TestCase):
    def _eval_greedy_head(self, src: str, dev_info_fn, env_val):
        class qwen3_5:  # noqa: N801
            class Model:
                pass

            class TextModel:
                pass

        class FakeFast:
            @staticmethod
            def greedy_quantized_argmax(*_args, **_kwargs):
                return ("tok", "rest")

        class FakeMx:
            fast = FakeFast()
            bfloat16 = "bf16"
            array = object

        if dev_info_fn is not None:
            FakeMx.device_info = staticmethod(dev_info_fn)

        class FakeEmb:
            bits = 4
            group_size = 64
            mode = "affine"
            weight = types.SimpleNamespace(shape=(100, 256))
            scales = object()
            biases = object()
            _greedy_high_bits = object()

            def __contains__(self, key):
                return key == "biases"

            def as_linear(self, _hidden):
                return "logits"

        class TiedTextModel(qwen3_5.TextModel):
            args = types.SimpleNamespace(tie_word_embeddings=True)
            model = types.SimpleNamespace(embed_tokens=FakeEmb())

        prev = os.environ.get("MLX_OMARCHY_NO_GREEDY_PRUNE")
        try:
            if env_val is None:
                os.environ.pop("MLX_OMARCHY_NO_GREEDY_PRUNE", None)
            else:
                os.environ["MLX_OMARCHY_NO_GREEDY_PRUNE"] = env_val
            ns = {
                "mx": FakeMx(),
                "nn": types.SimpleNamespace(QuantizedEmbedding=FakeEmb),
                "qwen3_5": qwen3_5,
            }
            patched = src.replace("from .models import qwen3_5", "")
            exec(patched, ns)
            return ns["_greedy_head"](TiedTextModel())
        finally:
            if prev is None:
                os.environ.pop("MLX_OMARCHY_NO_GREEDY_PRUNE", None)
            else:
                os.environ["MLX_OMARCHY_NO_GREEDY_PRUNE"] = prev

    def test_032_patch_preserves_batch_greedy_head_anchor(self):
        src = _extract_greedy_head_source(PATCH_PATHS[1])
        head_old = _load_batch_head_old()
        self.assertIn(head_old, src)

    def test_g14c_default_and_overrides_on_both_patch_series(self):
        def g14c():
            return {"device_name": "Apple M2 Max (G14C B1)"}

        def g13c():
            return {"device_name": "Apple M1 Max (G13C C0)"}

        def g16g():
            return {"device_name": "Apple M4 (G16G B0)"}

        def raises():
            raise RuntimeError("no device info")

        for patch_path in PATCH_PATHS:
            src = _extract_greedy_head_source(patch_path)
            with self.subTest(patch=patch_path.as_posix()):
                self.assertIsNotNone(self._eval_greedy_head(src, g14c, None))
                self.assertIsNotNone(self._eval_greedy_head(src, g14c, ""))
                self.assertIsNone(self._eval_greedy_head(src, g14c, "1"))
                self.assertIsNotNone(self._eval_greedy_head(src, g14c, "0"))

                self.assertIsNotNone(self._eval_greedy_head(src, g13c, None))
                self.assertIsNotNone(self._eval_greedy_head(src, g13c, "0"))
                self.assertIsNone(self._eval_greedy_head(src, g13c, "1"))

                self.assertIsNotNone(self._eval_greedy_head(src, g16g, None))
                self.assertIsNone(self._eval_greedy_head(src, g16g, "1"))

                # Missing or failing device_info fails open to enabled behavior.
                self.assertIsNotNone(self._eval_greedy_head(src, None, None))
                self.assertIsNotNone(self._eval_greedy_head(src, raises, None))

    def test_031_word_route_skips_k2048_token_sync_on_word_default_parts(self):
        src = _extract_greedy_head_source(PATCH_PATHS[0])
        sentinel_syncs = []

        class FakeToken:
            def __int__(self):
                sentinel_syncs.append(1)
                return 0

        class FakeLogits:
            def __sub__(self, _other):
                return self

            def squeeze(self, _axis):
                return self

        class qwen3_5:  # noqa: N801
            class TextModel:
                pass

        class FakeFast:
            @staticmethod
            def greedy_quantized_argmax(*_args, **_kwargs):
                return (FakeToken(), "rest")

        class FakeEmb:
            bits = 4
            group_size = 64
            mode = "affine"
            weight = types.SimpleNamespace(shape=(100, 256))
            scales = object()
            biases = object()
            _greedy_high_bits = object()

            def __contains__(self, key):
                return key == "biases"

            def as_linear(self, _hidden):
                return FakeLogits()

        class TiedTextModel(qwen3_5.TextModel):
            args = types.SimpleNamespace(tie_word_embeddings=True)
            model = types.SimpleNamespace(embed_tokens=FakeEmb())

        def run_case(dev_name, gw_env):
            sentinel_syncs.clear()
            prev_np = os.environ.get("MLX_OMARCHY_NO_GREEDY_PRUNE")
            prev_gw = os.environ.get("MLX_OMARCHY_GREEDY_WORD")
            try:
                os.environ.pop("MLX_OMARCHY_NO_GREEDY_PRUNE", None)
                if gw_env is None:
                    os.environ.pop("MLX_OMARCHY_GREEDY_WORD", None)
                else:
                    os.environ["MLX_OMARCHY_GREEDY_WORD"] = gw_env

                class FakeMx:
                    fast = FakeFast()
                    bfloat16 = "bf16"
                    array = object

                    @staticmethod
                    def device_info():
                        return {"device_name": dev_name}

                    @staticmethod
                    def logsumexp(_x, **_kw):
                        return FakeLogits()

                    @staticmethod
                    def argmax(_x):
                        return 0

                ns = {
                    "mx": FakeMx(),
                    "nn": types.SimpleNamespace(QuantizedEmbedding=FakeEmb),
                    "qwen3_5": qwen3_5,
                }
                exec(src.replace("from .models import qwen3_5", ""), ns)
                head = ns["_greedy_head"](TiedTextModel())
                head(types.SimpleNamespace(shape=(1, 2048)))
                return len(sentinel_syncs)
            finally:
                if prev_np is None:
                    os.environ.pop("MLX_OMARCHY_NO_GREEDY_PRUNE", None)
                else:
                    os.environ["MLX_OMARCHY_NO_GREEDY_PRUNE"] = prev_np
                if prev_gw is None:
                    os.environ.pop("MLX_OMARCHY_GREEDY_WORD", None)
                else:
                    os.environ["MLX_OMARCHY_GREEDY_WORD"] = prev_gw

        # G14C default uses the 3-dispatch word route: no per-token int(token) sync.
        self.assertEqual(run_case("Apple M2 Max (G14C B1)", None), 0)
        # Explicit MLX_OMARCHY_GREEDY_WORD=0 restores the staged route and its sync.
        self.assertEqual(run_case("Apple M2 Max (G14C B1)", "0"), 1)
        # G13G default uses the 3-dispatch word route: no per-token int(token) sync.
        self.assertEqual(run_case("Apple M1 (G13G B1)", None), 0)
        # Explicit MLX_OMARCHY_GREEDY_WORD=0 restores the staged route and its sync.
        self.assertEqual(run_case("Apple M1 (G13G B1)", "0"), 1)
        # G13C default uses the 3-dispatch word route: no per-token int(token) sync.
        self.assertEqual(run_case("Apple M1 Max (G13C C0)", None), 0)
        # Explicit MLX_OMARCHY_GREEDY_WORD=0 restores the staged route and its sync.
        self.assertEqual(run_case("Apple M1 Max (G13C C0)", "0"), 1)
        # Unmeasured parts stay on the staged K=2048 route and keep the sync.
        self.assertEqual(run_case("Apple M4 (G16G B1)", None), 1)
        # Explicit MLX_OMARCHY_GREEDY_WORD=1 skips the sync on any chip.
        self.assertEqual(run_case("Apple M4 (G16G B1)", "1"), 0)

        # Guard against C++/Python drift: every chip token listed in
        # greedy_word_over_staged_part() in device.h must skip the int(token)
        # sync in patches/mlx-lm-greedy-prune.patch by default.
        device_h = DEVICE_H.read_text(encoding="utf-8")
        m = re.search(
            r"greedy_word_over_staged_part\s*\([^)]*\)\s*\{(.*?)\}",
            device_h,
            re.DOTALL,
        )
        self.assertIsNotNone(m, "greedy_word_over_staged_part not found in device.h")
        cpp_chips = re.findall(r'device_name\.find\("([^"]+)"\)', m.group(1))
        self.assertTrue(cpp_chips, "no chip names parsed from greedy_word_over_staged_part")
        for chip in cpp_chips:
            with self.subTest(cpp_chip=chip):
                self.assertEqual(run_case(f"Apple Silicon ({chip} B1)", None), 0)
                self.assertEqual(run_case(f"Apple Silicon ({chip} B1)", "0"), 1)


if __name__ == "__main__":
    unittest.main()
