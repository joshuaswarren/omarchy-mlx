# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Frontend elimination of ``pad`` ops (Apple-tool evidence, 2026-09-13).

Apple's own ANE compiler (ane-compile-hwx, oracle witness in
mil-hwx-compiler receipts/2026-09-13-batched-oracle-mint-round2)
REJECTS ``pad`` in every form — including the identity view — so macOS
must eliminate pads from the graph before compilation. The mlx-omarchy
frontend does the same: every constant-zero ``pad`` whose amounts fall
on the spatial (H, W) dims of a rank-4 input is rewritten in place into
a depthwise identity convolution whose native ``pad`` attribute carries
the amounts:

    pad(x, amounts, constant_val=0, mode="constant")
        ≡ conv(x, weight=ones[C,1,1,1], groups=C, strides=[1,1],
               dilations=[1,1], pad=[h1,h2,w1,w2], pad_type="custom")

Bit-exactness proof, per lane:

- padded lanes: the conv pads with zeros and multiplies by 1.0:
  ``1.0 * 0.0 = +0.0`` — identical bits to the pad op's ``+0.0``
  constant (fp16 ``0x0000``).
- data lanes: depthwise (groups=C) convolution computes exactly
  ``1.0 * v`` — IEEE 754 multiplication by 1.0 is exact for every fp16
  value: ±0 keeps its sign (``0x8000`` stays ``0x8000``), subnormals
  are unchanged, ±inf pass through, quiet-NaN payloads are preserved.
  No cross-channel terms exist, so no ``0 * x`` accumulation can perturb
  a −0 lane into +0 (the failure mode of a full [C,C,1,1] identity
  kernel).

Pads with a non-zero constant, non-"constant" mode, non-const amounts,
amounts on the batch or channel dims, or a non-rank-4 input are
rejected with named reasons — the op then remains a compiler-side gap.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .schema import Model_pb2

FLOAT16 = 10
INT32 = 23
BOOL = 1


class PadEliminationError(RuntimeError):
    """A pad cannot be eliminated; the reason is named."""


@dataclass(frozen=True)
class PadPlanEntry:
    """Classification of one ``pad`` op."""

    index: int
    output_name: str
    disposition: str  # ELIMINABLE | REJECTED
    reason: str


@dataclass(frozen=True)
class PadElimination:
    """One performed rewrite."""

    output_name: str
    shape: tuple[int, ...]
    amounts: tuple[int, ...]


def _tensor_shape(op):
    out = op.outputs[0]
    which = out.type.WhichOneof("type")
    if which != "tensorType":
        return None
    return tuple(
        d.constant.size
        for d in out.type.tensorType.dimensions
        if d.WhichOneof("dimension") == "constant"
    )


def _const_values(spec_ops_by_name, name, expected_dtype):
    op = spec_ops_by_name.get(name)
    if op is None or op.type != "const":
        return None
    value = op.attributes.get("val")
    if value is None or value.WhichOneof("value") != "immediateValue":
        return None
    tensor = value.immediateValue.tensor
    which = tensor.WhichOneof("value")
    if which == "ints":
        return list(tensor.ints.values)
    if which == "bytes":
        payload = bytes(tensor.bytes.values)
        if expected_dtype == FLOAT16 and len(payload) == 2:
            return struct.unpack("<H", payload)[0]
        return None
    if which == "strings":
        return tensor.strings.values[0] if tensor.strings.values else None
    if which == "bools":
        return bool(tensor.bools.values[0]) if tensor.bools.values else None
    return None


def _named_binding(argument):
    for binding in argument.arguments:
        if binding.WhichOneof("binding") == "name":
            return binding.name
    return None


def plan_pad_elimination(spec) -> list[PadPlanEntry]:
    """Classify every ``pad`` op in the program."""
    by_name: dict[str, object] = {}
    for block in _walk_blocks(spec):
        for op in block.operations:
            for out in op.outputs:
                by_name[out.name] = op

    entries: list[PadPlanEntry] = []
    for block in _walk_blocks(spec):
        for index, op in enumerate(block.operations):
            if op.type != "pad":
                continue
            out = op.outputs[0].name if op.outputs else "?"
            entries.append(
                PadPlanEntry(index, out, *_classify(op, by_name))
            )
    return entries


def _classify(op, by_name) -> tuple[str, str]:
    shape = _tensor_shape(op)
    if shape is None or len(shape) != 4:
        return ("REJECTED", "pad input is not a static rank-4 tensor")

    mode = _const_values(by_name, _named_binding(op.inputs["mode"]) or "", None)
    if mode != "constant":
        return ("REJECTED", f"pad mode {mode!r} is not 'constant'")

    constant = _const_values(
        by_name, _named_binding(op.inputs["constant_val"]) or "", FLOAT16
    )
    if constant != 0x0000:
        return (
            "REJECTED",
            f"pad constant is fp16 0x{constant:04X}; only +0.0 is "
            "expressible as conv zero padding",
        )

    amounts_name = _named_binding(op.inputs["pad"])
    if amounts_name is None:
        return ("REJECTED", "pad amounts are not a single named const")
    amounts = _const_values(by_name, amounts_name, INT32)
    if not isinstance(amounts, list) or len(amounts) != 2 * len(shape):
        return ("REJECTED", "pad amounts are not a const [2*rank] tensor")

    # (begin, end) pairs per dim; only dims 2 (H) and 3 (W) may move.
    begin = amounts[0::2]
    end = amounts[1::2]
    for axis in (0, 1):
        if begin[axis] != 0 or end[axis] != 0:
            return (
                "REJECTED",
                f"pad amount on batch/channel dim {axis} is not "
                "expressible as conv spatial padding",
            )
    if any(value < 0 for value in amounts):
        return ("REJECTED", "negative pad amounts are not conv padding")
    return (
        "ELIMINABLE",
        f"depthwise identity-conv with pad=[{begin[2]},{end[2]},"
        f"{begin[3]},{end[3]}]",
    )


def eliminate_pads(spec) -> list[PadElimination]:
    """Rewrite every eliminable ``pad`` op in place.

    Each pad becomes a depthwise identity convolution: the amounts move
    onto the conv ``pad`` attribute ([h1, h2, w1, w2], ``custom``
    pad_type), the weight is a ones ``[C, 1, 1, 1]`` fp16 const, groups
    = C, strides and dilations stay 1. Output names and shapes are
    unchanged, so downstream consumers need no rewiring.
    """
    by_name: dict[str, object] = {}
    for block in _walk_blocks(spec):
        for op in block.operations:
            for out in op.outputs:
                by_name[out.name] = op

    counter = 0
    performed: list[PadElimination] = []
    for block in _walk_blocks(spec):
        for op in block.operations:
            if op.type != "pad":
                continue
            disposition, reason = _classify(op, by_name)
            if disposition != "ELIMINABLE":
                raise PadEliminationError(
                    f"refusing to rewrite non-eliminable pad "
                    f"({op.outputs[0].name if op.outputs else '?'}): {reason}"
                )
            shape = _tensor_shape(op)
            amounts = _const_values(
                by_name, _named_binding(op.inputs["pad"]), INT32
            )
            begin = amounts[0::2]
            end = amounts[1::2]
            channels = shape[1]
            counter += 1

            weight = _const_op(
                block,
                f"$padconv.weight.{counter}",
                (channels, 1, 1, 1),
                b"\x00\x3c" * channels,
            )
            strides = _ints_const(block, f"$padconv.strides.{counter}", [1, 1])
            dilations = _ints_const(
                block, f"$padconv.dilations.{counter}", [1, 1]
            )
            pad = _ints_const(
                block,
                f"$padconv.pad.{counter}",
                [begin[2], end[2], begin[3], end[3]],
            )
            pad_type = _string_const(
                block, f"$padconv.pad_type.{counter}", "custom"
            )
            groups = _ints_const(
                block, f"$padconv.groups.{counter}", [channels]
            )

            x_name = _named_binding(op.inputs["x"])
            out_type = op.outputs[0].type

            op.ClearField("inputs")
            op.ClearField("attributes")
            op.ClearField("blocks")
            op.type = "conv"
            op.inputs["x"].arguments.add().name = x_name
            op.inputs["weight"].arguments.add().name = weight
            op.inputs["strides"].arguments.add().name = strides
            op.inputs["dilations"].arguments.add().name = dilations
            op.inputs["pad"].arguments.add().name = pad
            op.inputs["pad_type"].arguments.add().name = pad_type
            op.inputs["groups"].arguments.add().name = groups
            # Output name/type untouched: same shape, same consumers.
            performed.append(
                PadElimination(
                    output_name=op.outputs[0].name,
                    shape=shape,
                    amounts=tuple(amounts),
                )
            )
    return performed


def _const_op(block, name: str, shape, raw: bytes) -> str:
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


def _ints_const(block, name: str, values: list[int]) -> str:
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
    return name


def _string_const(block, name: str, text: str) -> str:
    op = block.operations.add()
    op.type = "const"
    value = op.attributes["val"]
    tensor_type = value.type.tensorType
    tensor_type.dataType = 2  # STRING
    tensor_type.rank = 0
    value.immediateValue.tensor.strings.values.append(text)
    out = op.outputs.add()
    out.name = name
    out.type.tensorType.dataType = 2
    out.type.tensorType.rank = 0
    return name


def _walk_blocks(spec):
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            yield from _walk_block(block)


def _walk_block(block):
    yield block
    for op in block.operations:
        for nested in op.blocks:
            yield from _walk_block(nested)
