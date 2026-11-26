# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Prototype host lowering for the Parakeet boolean/mask path.

Design contract (docs/2026-09-13-h13-boolean-mask-fp16.md): masks are
fp16 ``0.0``/``1.0`` tensors (0 = false). This module classifies every
boolean/mask op in a parsed MIL program against that contract and
rewrites the subset that is expressible with ops the H13 backend already
has (``mul``, ``add``, ``reduce_max``), with bit-exactness stated per
rewrite. Ops with no exact algebraic lowering are reported with the
named reason; the ``-inf`` attention-bias select is the canonical
counterexample (``0 * -inf = NaN`` pollutes unmasked lanes).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .schema import Model_pb2

FLOAT16 = 10
BOOL = 1

TARGET_OPS = {
    "select",
    "cast",
    "less",
    "floor",
    "floor_div",
    "logical_and",
    "logical_not",
    "reduce_min",
}


class MaskLoweringError(RuntimeError):
    """A mask-path op cannot be classified or rewritten; reason named."""


@dataclass(frozen=True)
class PlanEntry:
    """Classification of one boolean/mask op."""

    index: int
    op_type: str
    output_name: str
    disposition: str  # LOWERABLE | NEEDS_COMPILER_OP | BOUNDARY
    reason: str


def _fp16_const_bits(spec_ops_by_name, name) -> int | None:
    """Raw fp16 bits of a scalar const producer, or None."""
    op = spec_ops_by_name.get(name)
    if op is None or op.type != "const":
        return None
    value = op.attributes.get("val")
    if value is None or value.WhichOneof("value") != "immediateValue":
        return None
    tensor = value.immediateValue.tensor
    if tensor.WhichOneof("value") != "bytes":
        return None
    payload = bytes(tensor.bytes.values)
    if len(payload) != 2:
        return None
    return struct.unpack("<H", payload)[0]


def _is_fp16_0_1_mask(spec_ops_by_name, binding) -> bool:
    """True when the bound value is an fp16 tensor (the 0/1 contract)."""
    if binding.WhichOneof("binding") != "value":
        return False
    return binding.value.type.WhichOneof("type") == "tensorType" and (
        binding.value.type.tensorType.dataType == FLOAT16
    )


def _named_binding(arg):
    for binding in arg.arguments:
        if binding.WhichOneof("binding") == "name":
            return binding
    return None


def plan_mask_lowering(spec) -> list[PlanEntry]:
    """Classify every boolean/mask op in the program."""
    by_name: dict[str, object] = {}
    for block in _walk_blocks(spec):
        for op in block.operations:
            for out in op.outputs:
                by_name[out.name] = op

    entries: list[PlanEntry] = []
    for block in _walk_blocks(spec):
        for index, op in enumerate(block.operations):
            if op.type not in TARGET_OPS:
                continue
            out = op.outputs[0].name if op.outputs else "?"
            entries.append(
                PlanEntry(index, op.type, out, *_classify(op, by_name))
            )
    return entries


def _classify(op, by_name) -> tuple[str, str]:
    if op.type == "logical_and":
        return (
            "LOWERABLE",
            "and(a, b) = mul(a, b): exact for fp16 0/1 masks "
            "(1*1=1, 0*x=+0)",
        )
    if op.type == "logical_not":
        return (
            "LOWERABLE",
            "not(a) = add(mul(a, -1.0), 1.0): exact for fp16 0/1 masks "
            "(0 -> 1, 1 -> +0; -0 intermediate normalizes)",
        )
    if op.type == "select":
        a_binding = _named_binding(op.inputs.get("a", op.inputs.get("x")))
        if a_binding is None:
            return (
                "NEEDS_COMPILER_OP",
                "select branch a is not name-bound; constant analysis "
                "impossible",
            )
        bits = _fp16_const_bits(by_name, a_binding.name)
        if bits is None:
            return (
                "NEEDS_COMPILER_OP",
                "select branch a is not a readable fp16 scalar const",
            )
        if not np.isfinite(np.frombuffer(struct.pack("<H", bits), "<f2")[0]):
            return (
                "NEEDS_COMPILER_OP",
                f"select fill is non-finite fp16 0x{bits:04X}; the "
                "m*a + (1-m)*b identity computes 0 * inf = NaN on "
                "unmasked lanes (counterexample class)",
            )
        if bits == 0x0000:  # +0.0 fill: mul(b, not(m))
            return (
                "LOWERABLE",
                "select(m, +0.0, b) = mul(b, not(m)): exact for finite b "
                "(0-branch: +0*b = +0 = fill)",
            )
        return (
            "LOWERABLE",
            "select(m, a, b) = add(mul(m, a), mul(not(m), b)): exact for "
            "finite a and b; a -0.0 selected value normalizes to +0.0 "
            "(value-equal)",
        )
    if op.type == "reduce_min":
        x = _named_binding(op.inputs.get("x", op.inputs.get("input")))
        if x is not None and _is_fp16_0_1_mask(by_name, x):
            pass
        return (
            "LOWERABLE",
            "reduce_min over a 0/1 mask = not(reduce_max(not(x))): exact "
            "on the 0/1 domain (general-domain fp16 min stays a compiler "
            "gap)",
        )
    if op.type == "less":
        return (
            "NEEDS_COMPILER_OP",
            "compare has no exact algebraic lowering over +,-,*,/ ; a "
            "step function needs the registry op `less` emitting fp16 "
            "0/1",
        )
    if op.type == "floor":
        return (
            "NEEDS_COMPILER_OP",
            "floor is exact in fp16 (|x| >= 2048 implies integral x; "
            "below that the integral result is representable) but has no "
            "arithmetic identity; needs registry op `floor`",
        )
    if op.type == "floor_div":
        return (
            "NEEDS_COMPILER_OP",
            "lowerable as real_div (const-divisor envelope question) + "
            "floor once `floor` exists; fp16 semantics match the "
            "package's own cast-then-divide behaviour, not integer "
            "floor_div (3199/32 -> 100 in fp16 vs 99 exact)",
        )
    if op.type == "cast":
        return (
            "BOUNDARY",
            "dtype conversion belongs to buffer pack/unpack on the host: "
            "fp32->fp16 and i32->fp16 by round-to-nearest-even (matching "
            "the package's own cast op), bool<->fp16 as 0/1",
        )
    raise MaskLoweringError(f"unclassified target op {op.type}")


def lower_mask_ops(spec) -> list[str]:
    """Rewrite the exact lowerable subset in place.

    Preconditions: mask tensors already carry the fp16 0/1 representation
    (boundary conversion applied); select fills are fp16 scalar consts.
    Returns the names of the rewritten outputs.
    """
    counter = _Counter()
    for block in _walk_blocks(spec):
        original = list(block.operations)
        count = len(original)
        pending: dict[int, list] = {}
        for position, op in enumerate(original):
            helpers = []
            if op.type == "logical_and":
                _rewrite_boolean_pair(op, counter)
            elif op.type == "logical_not":
                _rewrite_not(block, op, counter)
            elif op.type == "select":
                _rewrite_select(block, op, counter)
            elif op.type == "reduce_min":
                _rewrite_reduce_min(block, op, counter)
            else:
                continue
            pending[position] = list(block.operations)[count:]
            del block.operations[count:]
        if not pending:
            continue
        blobs = [op.SerializeToString() for op in block.operations]
        del block.operations[:]
        for index, blob in enumerate(blobs):
            for helper in pending.get(index, []):
                relayed = block.operations.add()
                relayed.ParseFromString(helper.SerializeToString())
            restored = block.operations.add()
            restored.ParseFromString(blob)
    return counter.rewritten


class _Counter:
    def __init__(self):
        self.rewritten: list[str] = []
        self.n = 0

    def next(self, prefix: str) -> str:
        self.n += 1
        return f"$mask.{prefix}.{self.n}"


def _fp16_const_op(block, name: str, shape, raw: bytes) -> str:
    op = block.operations.add()
    op.type = "const"
    value = op.attributes["val"]
    tensor_type = value.type.tensorType
    tensor_type.dataType = FLOAT16
    tensor_type.rank = len(shape)
    for dim in shape:
        tensor_type.dimensions.add().constant.size = dim
    value.immediateValue.tensor.bytes.values = raw
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = FLOAT16
    out.type.tensorType.rank = len(shape)
    for dim in shape:
        out.type.tensorType.dimensions.add().constant.size = dim
    return name


def _wire(op, key: str, name: str) -> None:
    op.inputs[key].arguments.add().name = name


def _set_output_fp16(op, shape) -> None:
    out = op.outputs[0]
    out.type.tensorType.dataType = FLOAT16
    out.type.tensorType.rank = len(shape)
    out.type.tensorType.ClearField("dimensions")
    for dim in shape:
        out.type.tensorType.dimensions.add().constant.size = dim


def _shape_of(op) -> tuple[int, ...]:
    return tuple(
        d.constant.size
        for d in op.outputs[0].type.tensorType.dimensions
        if d.WhichOneof("dimension") == "constant"
    )


def _rewrite_boolean_pair(op, counter) -> None:
    """logical_and(a, b) -> mul(a, b) (or the not-pair helper inputs)."""
    a = _named_binding(op.inputs["x"]).name
    b = _named_binding(op.inputs["y"]).name
    op.ClearField("attributes")
    op.ClearField("blocks")
    op.ClearField("inputs")
    _wire(op, "x", a)
    _wire(op, "y", b)
    op.type = "mul"
    _set_output_fp16(op, _shape_of(op))
    counter.rewritten.append(op.outputs[0].name)


def _rewrite_not(block, op, counter) -> None:
    """logical_not(x) -> add(mul(x, -1.0), 1.0)."""
    x = _named_binding(op.inputs["x"]).name
    shape = _shape_of(op)
    out_name = op.outputs[0].name
    neg1 = _fp16_const_op(block, counter.next("neg1"), (), struct.pack("<H", 0xBC00))
    one = _fp16_const_op(block, counter.next("one"), (), struct.pack("<H", 0x3C00))
    negated = counter.next("negated")
    mul_op = block.operations.add()
    mul_op.type = "mul"
    _wire(mul_op, "x", x)
    _wire(mul_op, "y", neg1)
    mul_out = mul_op.outputs.add()
    mul_out.name = negated
    mul_out.type.tensorType.dataType = FLOAT16
    mul_out.type.tensorType.rank = len(shape)
    for dim in shape:
        mul_out.type.tensorType.dimensions.add().constant.size = dim

    op.ClearField("attributes")
    op.ClearField("blocks")
    op.ClearField("inputs")
    _wire(op, "x", negated)
    _wire(op, "y", one)
    op.type = "add"
    _set_output_fp16(op, shape)
    counter.rewritten.append(out_name)


def _rewrite_select(block, op, counter) -> None:
    """select(m, a, b) with finite fp16 const a."""
    a_name = _named_binding(op.inputs["a"]).name
    b_name = _named_binding(op.inputs["b"]).name
    cond_name = _named_binding(op.inputs["cond"]).name
    a_bits = _fp16_const_bits(
        {o.outputs[0].name: o for o in block.operations if o.outputs}, a_name
    )
    if a_bits is None or not np.isfinite(
        np.frombuffer(struct.pack("<H", a_bits), "<f2")[0]
    ):
        raise MaskLoweringError(
            "refusing to lower select with non-finite or unreadable fill "
            f"(0x{a_bits:04X})" if a_bits is not None else
            "refusing to lower select with unreadable fill"
        )
    shape = _shape_of(op)
    if a_bits == 0x0000:
        # select(m, +0.0, b) = mul(b, not(m))
        not_name = counter.next("not")
        _emit_not(block, counter, cond_name, not_name, shape)
        op.ClearField("attributes")
        op.ClearField("blocks")
        op.ClearField("inputs")
        _wire(op, "x", b_name)
        _wire(op, "y", not_name)
        op.type = "mul"
        _set_output_fp16(op, shape)
    else:
        one = _fp16_const_op(block, counter.next("one"), (), struct.pack("<H", 0x3C00))
        not_name = counter.next("not")
        _emit_not(block, counter, cond_name, not_name, shape)
        ma = counter.next("ma")
        _emit_mul(block, counter, cond_name, a_name, ma, shape)
        nb = counter.next("nb")
        _emit_mul(block, counter, not_name, b_name, nb, shape)
        op.ClearField("attributes")
        op.ClearField("blocks")
        op.ClearField("inputs")
        _wire(op, "x", ma)
        _wire(op, "y", nb)
        op.type = "add"
        _set_output_fp16(op, shape)
    counter.rewritten.append(op.outputs[0].name)


def _rewrite_reduce_min(block, op, counter) -> None:
    """reduce_min(x) over 0/1 masks -> not(reduce_max(not(x)))."""
    x = _named_binding(op.inputs["x"]).name
    axes = _named_binding(op.inputs["axes"]).name
    shape = _shape_of(op)
    not1 = counter.next("not")
    _emit_not(block, counter, x, not1, shape)
    mx = counter.next("max")
    max_op = block.operations.add()
    max_op.type = "reduce_max"
    _wire(max_op, "x", not1)
    _wire(max_op, "axes", axes)
    keep = _named_binding(op.inputs.get("keep_dims"))
    if keep is not None:
        _wire(max_op, "keep_dims", keep.name)
    max_out = max_op.outputs.add()
    max_out.name = mx
    max_out.type.tensorType.dataType = FLOAT16
    reduced = _reduce_shape(op)
    max_out.type.tensorType.rank = len(reduced)
    for dim in reduced:
        max_out.type.tensorType.dimensions.add().constant.size = dim

    _emit_not(block, counter, mx, op.outputs[0].name, reduced, replace=op)
    counter.rewritten.append(op.outputs[0].name)


def _reduce_shape(op) -> tuple[int, ...]:
    shape = _shape_of(op)
    return shape  # keep_dims is asserted true by the caller contract


def _emit_not(block, counter, x_name, out_name, shape, replace=None) -> None:
    neg1 = _fp16_const_op(block, counter.next("neg1"), (), struct.pack("<H", 0xBC00))
    one = _fp16_const_op(block, counter.next("one"), (), struct.pack("<H", 0x3C00))
    negated = counter.next("negated")
    _emit_mul(block, counter, x_name, neg1, negated, shape)
    target = replace if replace is not None else block.operations.add()
    if replace is None:
        target.type = "add"
        out = target.outputs.add()
        out.name = out_name
        out.type.tensorType.dataType = FLOAT16
        out.type.tensorType.rank = len(shape)
        for dim in shape:
            out.type.tensorType.dimensions.add().constant.size = dim
    else:
        target.ClearField("attributes")
        target.ClearField("blocks")
        target.ClearField("inputs")
        target.type = "add"
        _set_output_fp16(target, shape)
    _wire(target, "x", negated)
    _wire(target, "y", one)


def _emit_mul(block, counter, x_name, y_name, out_name, shape) -> None:
    op = block.operations.add()
    op.type = "mul"
    _wire(op, "x", x_name)
    _wire(op, "y", y_name)
    out = op.outputs.add()
    out.name = out_name
    out.type.tensorType.dataType = FLOAT16
    out.type.tensorType.rank = len(shape)
    for dim in shape:
        out.type.tensorType.dimensions.add().constant.size = dim


def op_root_block(block):
    return [block]


def _walk_blocks(spec):
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            yield from _walk_block(block)


def _walk_block(block):
    yield block
    for op in block.operations:
        for nested in op.blocks:
            yield from _walk_block(nested)
