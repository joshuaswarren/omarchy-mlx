# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the fp16 0/1 boolean/mask lowering prototype."""

import os
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.mask_lowering import (
    MaskLoweringError,
    lower_mask_ops,
    plan_mask_lowering,
)
from coreml.proto import load_model
from coreml.schema import Model_pb2

FLOAT16 = 10
_HF_REVISION = "b650695c2322ee5281dff48d7345b2f3a58ff018"


def _cache_root() -> Path | None:
    base = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    roots = []
    if base:
        roots.append(Path(base) / "parakeet-reference")
    roots.append(Path.home() / ".cache" / "mlx-omarchy" / "parakeet-reference")
    for root in roots:
        candidate = (
            root / "mweinbach1" / "parakeet-tdt-0.6b-v3-coreml" / _HF_REVISION
        )
        if candidate.is_dir():
            return candidate
    return None


class _Fixture:
    """A synthetic fp16-0/1 mask program built with schema bindings."""

    def __init__(self):
        self.spec = Model_pb2.Model()
        self.spec.specificationVersion = 9
        self.block = (
            self.spec.mlProgram.functions["main"].block_specializations[
                "CoreML8"
            ]
        )

    def const(self, name, values, shape=None):
        array = np.asarray(values, dtype=np.float16)
        if shape is not None:
            array = array.reshape(shape)
        op = self.block.operations.add()
        op.type = "const"
        value_attr = op.attributes["val"]
        tensor_type = value_attr.type.tensorType
        tensor_type.dataType = FLOAT16
        tensor_type.rank = array.ndim
        for dim in array.shape:
            tensor_type.dimensions.add().constant.size = int(dim)
        value_attr.immediateValue.tensor.bytes.values = array.tobytes()
        out = op.outputs.add()
        out.name = name
        out.type.tensorType.dataType = FLOAT16
        out.type.tensorType.rank = array.ndim
        for dim in array.shape:
            out.type.tensorType.dimensions.add().constant.size = int(dim)
        return name

    def logical_and(self, name, x, y, shape):
        op = self.block.operations.add()
        op.type = "logical_and"
        self._wire(op, "x", x)
        self._wire(op, "y", y)
        self._out(op, name, shape)

    def logical_not(self, name, x, shape):
        op = self.block.operations.add()
        op.type = "logical_not"
        self._wire(op, "x", x)
        self._out(op, name, shape)

    def select(self, name, a, b, cond, shape):
        op = self.block.operations.add()
        op.type = "select"
        self._wire(op, "a", a)
        self._wire(op, "b", b)
        self._wire(op, "cond", cond)
        self._out(op, name, shape)

    def reduce_min(self, name, x, axes, shape, reduced):
        keep = self.const(name + "_keep", 1.0)
        op = self.block.operations.add()
        op.type = "reduce_min"
        self._wire(op, "x", x)
        self._wire(op, "axes", axes)
        self._wire(op, "keep_dims", keep)
        self._out(op, name, reduced)

    @staticmethod
    def _wire(op, key, name):
        op.inputs[key].arguments.add().name = name

    @staticmethod
    def _out(op, name, shape):
        out = op.outputs.add()
        out.name = name
        out.type.tensorType.dataType = FLOAT16
        out.type.tensorType.rank = len(shape)
        for dim in shape:
            out.type.tensorType.dimensions.add().constant.size = int(dim)


def _evaluate(spec) -> dict[str, np.ndarray]:
    """Tiny evaluator for const/mul/add/reduce_max fp16 programs."""
    values: dict[str, np.ndarray] = {}
    block = spec.mlProgram.functions["main"].block_specializations["CoreML8"]
    for op in block.operations:
        if op.type == "const":
            attr = op.attributes["val"]
            payload = bytes(attr.immediateValue.tensor.bytes.values)
            dims = [
                d.constant.size
                for d in attr.type.tensorType.dimensions
                if d.WhichOneof("dimension") == "constant"
            ]
            array = np.frombuffer(payload, "<f2")
            values[op.outputs[0].name] = (
                array.reshape(dims) if dims else array.reshape(())
            )
            continue

        def arg(key):
            binding = op.inputs[key].arguments[0]
            assert binding.WhichOneof("binding") == "name"
            return values[binding.name]

        if op.type == "mul":
            values[op.outputs[0].name] = arg("x") * arg("y")
        elif op.type == "add":
            values[op.outputs[0].name] = arg("x") + arg("y")
        elif op.type == "reduce_max":
            axes = tuple(int(v) for v in arg("axes"))
            values[op.outputs[0].name] = arg("x").max(axis=axes, keepdims=True)
        else:
            raise AssertionError(f"unexpected op {op.type} in lowered graph")
    return values


_B_ADVERSARIAL = [
    0.0, -0.0, 6.0e-8, -6.0e-8, 65504.0, -65504.0, 1.0, -2.75,
    0.5, -0.5, 2048.0, 3.7e-4, -8192.0, 511.0, -0.0001, 42.0,
]


class MaskLoweringTest(unittest.TestCase):
    def test_and_not_select0_reduce_min_are_bit_exact(self):
        shape = (4, 4)
        rng = np.random.default_rng(4242)
        b_arr = np.asarray(_B_ADVERSARIAL, dtype=np.float16).reshape(shape)
        m_arr = rng.integers(0, 2, size=shape).astype(np.float16)
        not_expected = np.where(
            m_arr == 0, np.float16(1.0), np.float16(0.0)
        )

        fx = _Fixture()
        fx.const("m_t", m_arr)
        fx.const("b", b_arr)
        a_zero = fx.const("a_zero", 0.0)
        axes = fx.const("axes", [1.0])
        fx.logical_and("and_out", "m_t", "m_t", shape)
        fx.logical_not("not_out", "m_t", shape)
        fx.select("sel0", "a_zero", "b", "m_t", shape)
        fx.logical_not("self_min", "m_t", shape)
        fx.reduce_min("rmin", "self_min", "axes", shape, (4, 1))

        plan = plan_mask_lowering(fx.spec)
        dispositions = {e.op_type: e.disposition for e in plan}
        self.assertEqual(dispositions["logical_and"], "LOWERABLE")
        self.assertEqual(dispositions["logical_not"], "LOWERABLE")
        self.assertEqual(dispositions["select"], "LOWERABLE")
        self.assertEqual(dispositions["reduce_min"], "LOWERABLE")

        rewritten = lower_mask_ops(fx.spec)
        for name in ("and_out", "not_out", "sel0", "rmin"):
            self.assertIn(name, rewritten)

        out = _evaluate(fx.spec)

        # and: bit-exact
        self.assertTrue(np.array_equal(
            out["and_out"].view("<u2"), (m_arr * m_arr).view("<u2")))
        # not: bit-exact (0 -> 1.0, 1 -> +0.0)
        self.assertTrue(np.array_equal(
            out["not_out"].view("<u2"), not_expected.view("<u2")))
        # select(m, +0.0, b): value-exact; the only bit differences are
        # sign(b)-signed zeros on fill lanes (m=1 with b < 0 yields -0.0
        # where the fill constant is +0.0) — value-equal by IEEE.
        expected_sel = np.where(m_arr == 1, np.float16(0.0), b_arr)
        self.assertTrue(np.array_equal(out["sel0"], expected_sel))
        non_negative_fill = ~(m_arr == 1) | (np.signbit(b_arr) == False)
        self.assertTrue(np.array_equal(
            out["sel0"].view("<u2")[non_negative_fill],
            expected_sel.view("<u2")[non_negative_fill],
        ))
        # reduce_min over not(mask), axes=[1], keep_dims: bit-exact
        expected_min = not_expected.min(axis=1, keepdims=True)
        self.assertTrue(np.array_equal(
            out["rmin"].view("<u2"), expected_min.view("<u2")))

    def test_finite_fill_select_general_identity(self):
        shape = (3, 3)
        rng = np.random.default_rng(99)
        b_arr = rng.uniform(-2, 2, size=shape).astype(np.float16)
        m_arr = rng.integers(0, 2, size=shape).astype(np.float16)

        fx = _Fixture()
        fx.const("m_t", m_arr)
        fx.const("b", b_arr)
        fx.const("a_fin", -7.5)
        fx.select("selfin", "a_fin", "b", "m_t", shape)

        lower_mask_ops(fx.spec)
        out = _evaluate(fx.spec)
        expected = np.where(m_arr == 1, np.float16(-7.5), b_arr)
        # -0.0 in a selected b lane normalizes to +0.0 (value-equal);
        # everything else is bit-exact.
        self.assertTrue(np.array_equal(out["selfin"], expected))

    def test_nonfinite_fill_select_is_refused_with_counterexample(self):
        shape = (4,)
        b_arr = np.array([1.0, -2.0, 0.5, -0.0], dtype=np.float16)
        m_arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16)

        fx = _Fixture()
        fx.const("m_t", m_arr)
        fx.const("b", b_arr)
        fx.const("a_ninf", float("-inf"))
        fx.select("selinf", "a_ninf", "b", "m_t", shape)

        plan = plan_mask_lowering(fx.spec)
        entry = next(e for e in plan if e.op_type == "select")
        self.assertEqual(entry.disposition, "NEEDS_COMPILER_OP")
        self.assertIn("0 * inf = NaN", entry.reason)

        with self.assertRaisesRegex(MaskLoweringError, "non-finite"):
            lower_mask_ops(fx.spec)

        # The hazard itself: the naive identity poisons unmasked lanes.
        naive = m_arr * np.float16(-np.inf) + (1 - m_arr) * b_arr
        self.assertTrue(np.isnan(naive[1:]).all())

    def test_real_encoder_planner_classification(self):
        cache = _cache_root()
        if cache is None:
            self.skipTest("parakeet reference cache not present")
        spec = load_model(
            (cache / "encoder.mlpackage" / "Data" / "com.apple.CoreML"
             / "model.mlmodel").read_bytes()
        )
        plan = plan_mask_lowering(spec)
        counts = Counter((e.op_type, e.disposition) for e in plan)
        self.assertEqual(counts[("select", "NEEDS_COMPILER_OP")], 24)
        self.assertEqual(counts[("select", "LOWERABLE")], 24)
        self.assertEqual(counts[("logical_and", "LOWERABLE")], 1)
        self.assertEqual(counts[("logical_not", "LOWERABLE")], 1)
        self.assertEqual(counts[("reduce_min", "LOWERABLE")], 1)
        self.assertEqual(counts[("less", "NEEDS_COMPILER_OP")], 4)
        self.assertEqual(counts[("floor", "NEEDS_COMPILER_OP")], 3)
        self.assertEqual(counts[("floor_div", "NEEDS_COMPILER_OP")], 3)
        self.assertEqual(counts[("cast", "BOUNDARY")], 11)
        self.assertEqual(len(plan), 72)


if __name__ == "__main__":
    unittest.main()
