# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Behavioral tests for G14C default-off gating in mlx-lm-greedy-prune patches."""

import ast
import os
import pathlib
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
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
                self.assertIsNone(self._eval_greedy_head(src, g14c, None))
                self.assertIsNone(self._eval_greedy_head(src, g14c, ""))
                self.assertIsNone(self._eval_greedy_head(src, g14c, "1"))
                self.assertIsNotNone(self._eval_greedy_head(src, g14c, "0"))

                self.assertIsNotNone(self._eval_greedy_head(src, g13c, None))
                self.assertIsNotNone(self._eval_greedy_head(src, g13c, "0"))
                self.assertIsNone(self._eval_greedy_head(src, g13c, "1"))

                self.assertIsNotNone(self._eval_greedy_head(src, g16g, None))
                self.assertIsNone(self._eval_greedy_head(src, g16g, "1"))

                # Missing or failing device_info fails open to prior enabled behavior.
                self.assertIsNotNone(self._eval_greedy_head(src, None, None))
                self.assertIsNotNone(self._eval_greedy_head(src, raises, None))


if __name__ == "__main__":
    unittest.main()
