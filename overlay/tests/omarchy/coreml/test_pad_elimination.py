# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for frontend pad elimination (Apple-tool evidence)."""

import os
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.pad_elimination import (
    PadEliminationError,
    eliminate_pads,
    plan_pad_elimination,
)
from coreml.proto import load_model
from coreml.schema import Model_pb2

FLOAT16 = 10
INT32 = 23
_HF_REVISION = "b650695c2322ee5281dff48d7345b2f3a58ff018"
_ENCODER = (
    "mweinbach1/parakeet-tdt-0.6b-v3-coreml" f"/{_HF_REVISION}"
    "/encoder.mlpackage"
)


def _cache() -> Path | None:
    base = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    roots = []
    if base:
        roots.append(Path(base) / "parakeet-reference")
    roots.append(Path.home() / ".cache" / "mlx-omarchy" / "parakeet-reference")
    for root in roots:
        candidate = root / _ENCODER
        if candidate.is_dir():
            return candidate
    return None


def _fp16_const(block, name, values, shape):
    array = np.asarray(values, dtype=np.float16).reshape(shape)
    op = block.operations.add()
    op.type = "const"
    value = op.attributes["val"]
    tensor_type = value.type.tensorType
    tensor_type.dataType = FLOAT16
    tensor_type.rank = array.ndim
    for dim in array.shape:
        tensor_type.dimensions.add().constant.size = int(dim)
    value.immediateValue.tensor.bytes.values = array.tobytes()
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = FLOAT16
    out.type.tensorType.rank = array.ndim
    for dim in array.shape:
        out.type.tensorType.dimensions.add().constant.size = int(dim)


def _ints_const(block, name, values):
    op = block.operations.add()
    op.type = "const"
    value = op.attributes["val"]
    tensor_type = value.type.tensorType
    tensor_type.dataType = INT32
    tensor_type.rank = 1
    tensor_type.dimensions.add().constant.size = len(values)
    value.immediateValue.tensor.ints.values.extend(values)
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = INT32
    out.type.tensorType.rank = 1
    out.type.tensorType.dimensions.add().constant.size = len(values)


def _string_const(block, name, text):
    op = block.operations.add()
    op.type = "const"
    value = op.attributes["val"]
    value.type.tensorType.dataType = 2
    value.immediateValue.tensor.strings.values.append(text)
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = 2


def _tensor_out(op, name, shape):
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = FLOAT16
    out.type.tensorType.rank = len(shape)
    for dim in shape:
        out.type.tensorType.dimensions.add().constant.size = int(dim)


def _build_pad_fixture(shape, amounts):
    """A minimal program: input -> pad(amounts) -> output."""
    spec = Model_pb2.Model()
    spec.specificationVersion = 9
    block = spec.mlProgram.functions["main"].block_specializations["CoreML8"]

    # x is a boundary input: declare via function inputs
    named = spec.mlProgram.functions["main"].inputs.add()
    named.name = "x"
    named.type.tensorType.dataType = FLOAT16
    named.type.tensorType.rank = len(shape)
    for dim in shape:
        named.type.tensorType.dimensions.add().constant.size = int(dim)

    pad_amounts_name = "pad_amounts"
    _ints_const(block, pad_amounts_name, list(amounts))
    mode_name = "pad_mode"
    _string_const(block, mode_name, "constant")
    zero_name = "pad_zero"
    _fp16_const(block, zero_name, [0.0], ())

    op = block.operations.add()
    op.type = "pad"
    op.inputs["x"].arguments.add().name = "x"
    op.inputs["pad"].arguments.add().name = pad_amounts_name
    op.inputs["mode"].arguments.add().name = mode_name
    op.inputs["constant_val"].arguments.add().name = zero_name
    padded = tuple(
        dim + amounts[2 * axis] + amounts[2 * axis + 1]
        for axis, dim in enumerate(shape)
    )
    _tensor_out(op, "padded", padded)
    return spec


def _mil_pad_reference(x: np.ndarray, amounts) -> np.ndarray:
    pads = tuple(
        (amounts[2 * axis], amounts[2 * axis + 1])
        for axis in range(x.ndim)
    )
    return np.pad(x, pads, mode="constant", constant_values=0.0)


def _identity_conv_reference(x: np.ndarray, conv_pad) -> np.ndarray:
    # conv_pad = [h1, h2, w1, w2]; depthwise 1x1 kernel of ones.
    h1, h2, w1, w2 = conv_pad
    padded = np.pad(
        x, ((0, 0), (0, 0), (h1, h2), (w1, w2)),
        mode="constant", constant_values=0.0,
    )
    return (padded * np.float16(1.0)).astype(np.float16)


_ADVERSARIAL = [
    0.0, -0.0, 6.0e-8, -6.0e-8, 1.0, -2.75, 65504.0, -65504.0,
    float("inf"), float("-inf"), float("nan"), 0.5, -0.5, 2048.0,
]


class PadEliminationTest(unittest.TestCase):
    def test_rewrite_is_bit_exact_on_adversarial_values(self):
        shape = (2, 3, 4, 5)
        amounts = (0, 0, 0, 0, 0, 1, 1, 0)  # H end+0? begin H 1, begin W 1
        spec = _build_pad_fixture(shape, amounts)
        plan = plan_pad_elimination(spec)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "ELIMINABLE")

        performed = eliminate_pads(spec)
        self.assertEqual(len(performed), 1)
        # The op is now a conv with depthwise ones weight and custom pad.
        block = spec.mlProgram.functions["main"].block_specializations[
            "CoreML8"
        ]
        conv = next(
            op for op in block.operations if op.type == "conv"
        )
        self.assertEqual(conv.outputs[0].name, "padded")
        by_name = {
            op.outputs[0].name: op
            for op in block.operations
            if op.type == "const" and op.outputs
        }
        pad_values = list(
            by_name[
                conv.inputs["pad"].arguments[0].name
            ].attributes["val"].immediateValue.tensor.ints.values
        )
        self.assertEqual(pad_values, [0, 1, 1, 0])
        groups = list(
            by_name[
                conv.inputs["groups"].arguments[0].name
            ].attributes["val"].immediateValue.tensor.ints.values
        )
        self.assertEqual(groups, [3])
        weight_bytes = bytes(
            by_name[
                conv.inputs["weight"].arguments[0].name
            ].attributes["val"].immediateValue.tensor.bytes.values
        )
        self.assertEqual(
            weight_bytes,
            np.asarray([1.0] * 3, dtype=np.float16).tobytes(),
        )

        # Numerics: MIL pad semantics vs depthwise identity conv on the
        # same adversarial payload — BITWISE (uint16) equality.
        rng = np.random.default_rng(77)
        x = np.asarray(_ADVERSARIAL[:20], dtype=np.float16)
        x = np.concatenate(
            [x, rng.uniform(-2, 2, int(np.prod(shape)) - x.size).astype(
                np.float16
            )]
        ).reshape(shape)
        reference = _mil_pad_reference(x, amounts)
        conv_pad = (amounts[4], amounts[5], amounts[6], amounts[7])
        rewritten = _identity_conv_reference(x, conv_pad)
        self.assertEqual(reference.shape, rewritten.shape)
        self.assertTrue(
            np.array_equal(
                reference.view("<u2"), rewritten.view("<u2")
            ),
            "identity-conv rewrite must be bit-exact, including "
            "±0, subnormals, ±inf and NaN payloads",
        )

    def test_encoder_shape_case_is_bit_exact(self):
        # The real pattern: [1,8,375,749] with begin-W 1 -> [1,8,375,750].
        shape = (1, 8, 375, 749)
        amounts = (0, 0, 0, 0, 0, 0, 1, 0)
        rng = np.random.default_rng(2026)
        x = rng.uniform(-4, 4, size=shape).astype(np.float16)
        x.flat[0] = np.float16(-0.0)
        x.flat[1] = np.float16(0.0)
        x.flat[2] = np.float16(6.0e-8)
        reference = _mil_pad_reference(x, amounts)
        rewritten = _identity_conv_reference(x, (0, 0, 1, 0))
        self.assertEqual(reference.shape, (1, 8, 375, 750))
        self.assertTrue(
            np.array_equal(reference.view("<u2"), rewritten.view("<u2"))
        )

    def test_nonzero_constant_and_spatial_violations_are_named(self):
        shape = (2, 3, 4, 5)
        # Non-zero constant: build fixture then corrupt the zero const.
        spec = _build_pad_fixture(shape, (0, 0, 0, 0, 0, 0, 1, 0))
        block = spec.mlProgram.functions["main"].block_specializations[
            "CoreML8"
        ]
        for op in block.operations:
            if op.type == "const" and op.outputs and op.outputs[0].name == "pad_zero":
                op.attributes["val"].immediateValue.tensor.bytes.values = (
                    np.asarray([1.5], dtype=np.float16).tobytes()
                )
        plan = plan_pad_elimination(spec)
        self.assertEqual(plan[0].disposition, "REJECTED")
        self.assertIn("only +0.0", plan[0].reason)
        with self.assertRaisesRegex(PadEliminationError, "non-eliminable"):
            eliminate_pads(spec)

        # Batch-dim amount: rejected, named.
        spec2 = _build_pad_fixture(shape, (1, 0, 0, 0, 0, 0, 0, 0))
        plan2 = plan_pad_elimination(spec2)
        self.assertEqual(plan2[0].disposition, "REJECTED")
        self.assertIn("batch/channel dim 0", plan2[0].reason)

    def test_real_encoder_pads_all_eliminated(self):
        cache = _cache()
        if cache is None:
            self.skipTest("parakeet reference cache not present")
        from coreml.mlpackage import open_mlpackage

        package = open_mlpackage(cache)
        spec = load_model(package.model_path.read_bytes())

        before = Counter()
        block = spec.mlProgram.functions["main"].block_specializations[
            "CoreML8"
        ]
        for op in block.operations:
            before[op.type] += 1
        self.assertEqual(before["pad"], 24)
        self.assertEqual(before["conv"], 77)

        plan = plan_pad_elimination(spec)
        self.assertEqual(len(plan), 24)
        self.assertTrue(all(e.disposition == "ELIMINABLE" for e in plan))

        performed = eliminate_pads(spec)
        self.assertEqual(len(performed), 24)
        after = Counter()
        for op in block.operations:
            after[op.type] += 1
        self.assertEqual(after["pad"], 0)
        self.assertEqual(after["conv"], 101)  # 77 + 24 identity convs
        # Every rewritten op keeps its original output name and shape.
        for record in performed:
            self.assertEqual(record.shape, (1, 8, 375, 750))
            self.assertEqual(record.amounts, (0, 0, 0, 0, 0, 0, 1, 0))
        # No pad outputs remain referenced.
        names = {op.outputs[0].name for op in block.operations if op.outputs}
        for record in performed:
            self.assertIn(record.output_name, names)


if __name__ == "__main__":
    unittest.main()
