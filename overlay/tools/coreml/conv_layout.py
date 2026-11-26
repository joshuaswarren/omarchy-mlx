# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Spell 1×1 valid conv as the H13 rank-4 corpus form.

H13 rejects encoder ``conv`` fp16 ``[1, 256, 750, 32]`` (and the
sibling ``[1, 256, 375, 16]``) as ``h13.conv-outside-envelope``.
``convParityPlan`` needs rank-4 batch-1, square ``[Cout, Cin, k, k]``
weight, unit dilations, zero ``pad``, ``pad_type`` ``same``/``valid``,
and an ``int32`` **scalar** ``groups``. The encoder spells
``pad_type`` as ``tensor<string, []>`` and ``groups`` as
``tensor<int32, []>``. Corpus probes use ``string`` / ``int32``.

k1 / stride-1 / valid is the same arithmetic as ``same`` (kernel 1
never pads). Nearby decoded CHW ``{256, 32, 32}`` compiles; CHW
``{256, 750, 32}`` is the remaining table miss
(``receipts/2026-09-13-h13-conv-envelope.md``). This rewrite does not
tile 750 into 32: 750/32 is not an integer.

A reshape of ``[1, 256, 750, 32]`` to ``[1, 750, 32, 256]`` keeps the
element count and changes packing. ``numpy_naive_nhwc_reshape`` is that
counterexample. The NCHW flatten through ``[N, C, H*W]`` is a view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class ConvLayoutError(RuntimeError):
    """A conv-layout rewrite is unsound here; the reason is named."""


STANDALONE_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 256, 32, 32]> x) {
    string pt = const()[name = string("pt"), val = string("valid")];
    tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];
    tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([0, 0, 0, 0])];
    tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];
    int32 gp = const()[name = string("gp"), val = int32(1)];
    tensor<fp16, [256, 256, 1, 1]> w = const()[name = string("w"), val = tensor<fp16, [256, 256, 1, 1]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [256]> b = const()[name = string("b"), val = tensor<fp16, [256]>(BLOBFILE(path = string("@model_path/bias.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 256, 32, 32]> y = conv(bias = b, dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string("y")];
  } -> (y);
}
"""

_DEF_LINE = re.compile(
    r"^(?P<indent>\s*)(?:tensor<(?P<dtype>[a-z0-9]+),\s*"
    r"(?P<shape>\[[^\]]*\])>|(?P<scalar>string|int32|bool))\s*"
    r"(?P<name>[A-Za-z_][\w]*) = "
    r"(?P<op>[A-Za-z_][\w]*)\("
)
_ARG = re.compile(r"(?P<key>[A-Za-z_][\w]*)\s*=\s*(?P<value>[A-Za-z_][\w]*)")
_PARAM = re.compile(
    r"tensor<(?P<dtype>[a-z0-9]+),\s*(?P<shape>\[[^\]]*\])>\s*"
    r"(?P<name>[A-Za-z_][\w]*)"
)
_INT_VAL = re.compile(
    r"val = (?:tensor<int32, \[[^\]]*\]>\()?(?:int32\()?(?P<body>[^)]*)"
)
_STR_VAL = re.compile(
    r'val = (?:tensor<string, \[[^\]]*\]>\()?(?:string\()?"(?P<body>[^"]*)"'
)


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one rank-4 1×1 valid/same groups-1 conv."""

    index: int
    output_name: str
    disposition: str  # REWRITTEN | REFUSED
    reason: str


@dataclass
class LayoutReport:
    rewritten: list[str] = field(default_factory=list)
    refused: list[LayoutEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": "mlx-omarchy.conv-layout-rewrite.v1",
            "rewritten": list(self.rewritten),
            "refused": [
                {
                    "index": entry.index,
                    "output": entry.output_name,
                    "reason": entry.reason,
                }
                for entry in self.refused
            ],
        }


def numpy_nchw_flatten_view(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[N, C, H, W]`` vs reshape through ``[N, C, H*W]``.

    Same bytes. Callers compare the pair; this is a reshape-family
    view, not a channel mix.
    """
    packed = np.asarray(source)
    if packed.ndim != 4:
        raise ConvLayoutError(
            f"flatten view expects rank-4 [N, C, H, W], got {packed.shape}"
        )
    batch, channels, height, width = packed.shape
    rewritten = packed.reshape(batch, channels, height * width).reshape(
        batch, channels, height, width
    )
    return packed, rewritten


def numpy_naive_nhwc_reshape(
    source: np.ndarray, weight: np.ndarray, bias: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """1×1 NCHW conv vs last-axis mix of reshape ``[N, H, W, C]``.

    Same element count, different packing. Callers compare the pair; a
    mismatch is a named counterexample, not a rounding tolerance.
    """
    packed, kernel, offset = _as_conv1x1(source, weight, bias)
    original = _fp16_conv1x1(packed, kernel, offset)
    batch, channels, height, width = packed.shape
    naive = packed.reshape(batch, height, width, channels)
    acc = naive.astype(np.float32) @ kernel[:, :, 0, 0].astype(np.float32).T
    if offset is not None:
        acc = acc + offset.astype(np.float32)
    rewritten = acc.astype(np.float16).reshape(
        batch, kernel.shape[0], height, width
    )
    return original, rewritten



def require_exact(original: np.ndarray, rewritten: np.ndarray, what: str) -> None:
    """Raise ``ConvLayoutError`` naming the first mismatched lane."""
    if original.shape != rewritten.shape or original.dtype != rewritten.dtype:
        raise ConvLayoutError(
            f"{what}: shape/dtype {original.shape}/{original.dtype} vs "
            f"{rewritten.shape}/{rewritten.dtype}"
        )
    left = original.view("<u2")
    right = rewritten.view("<u2")
    if np.array_equal(left, right):
        return
    mismatch = np.argwhere(left != right)
    index = tuple(int(v) for v in mismatch[0])
    raise ConvLayoutError(
        f"inexact {what} at {list(index)}: original="
        f"{original[index]!r} rewritten={rewritten[index]!r}"
    )


def _as_conv1x1(source, weight, bias):
    packed = np.asarray(source)
    kernel = np.asarray(weight)
    if packed.ndim != 4:
        raise ConvLayoutError(
            f"1×1 conv expects rank-4 [N, C, H, W], got {packed.shape}"
        )
    if kernel.ndim != 4 or kernel.shape[2:] != (1, 1):
        raise ConvLayoutError(
            f"1×1 weight expects [Cout, Cin, 1, 1], got {kernel.shape}"
        )
    if kernel.shape[1] != packed.shape[1]:
        raise ConvLayoutError(
            f"weight Cin {kernel.shape[1]} != input C {packed.shape[1]}"
        )
    offset = None if bias is None else np.asarray(bias)
    if offset is not None and offset.shape != (kernel.shape[0],):
        raise ConvLayoutError(
            f"bias shape {offset.shape} != ({kernel.shape[0]},)"
        )
    return packed, kernel, offset


def _fp16_conv1x1(source, weight, bias):
    acc = np.einsum(
        "nchw,oc->nohw",
        source.astype(np.float32),
        weight.astype(np.float32)[:, :, 0, 0],
    )
    if bias is not None:
        acc = acc + np.asarray(bias, np.float32).reshape(1, -1, 1, 1)
    return acc.astype(np.float16)


def _parse_shape(text: str) -> tuple[int, ...] | None:
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return ()
    try:
        return tuple(int(part.strip()) for part in inner.split(","))
    except ValueError:
        return None


def _code_of(line: str) -> str:
    cut = line.find("[name =")
    return line if cut < 0 else line[:cut]


def _parse_ints(line: str) -> list[int] | None:
    match = _INT_VAL.search(line)
    if match is None:
        return None
    body = match.group("body").strip()
    try:
        if body.startswith("[") and body.endswith("]"):
            inner = body[1:-1].strip()
            if not inner:
                return []
            return [int(part.strip()) for part in inner.split(",")]
        return [int(body)]
    except ValueError:
        return None


def _parse_string(line: str) -> str | None:
    match = _STR_VAL.search(line)
    if match is None:
        return None
    return match.group("body")


def _index_ops(lines: list[str]):
    table = {}
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line.rstrip("\n"))
        if not match:
            continue
        args = dict(_ARG.findall(_code_of(line).split("(", 1)[1]))
        dtype = match.group("dtype") or match.group("scalar")
        shape = match.group("shape") or "[]"
        table[match.group("name")] = (
            index,
            match.group("op"),
            dtype,
            shape,
            args,
            match.group("indent"),
            match.group("scalar") is not None,
        )
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("func ") and "(" in stripped:
            params = stripped.split("(", 1)[1].rsplit(")", 1)[0]
            for match in _PARAM.finditer(params):
                table.setdefault(
                    match.group("name"),
                    (
                        index,
                        "param",
                        match.group("dtype"),
                        match.group("shape"),
                        {},
                        "",
                        False,
                    ),
                )
    return table


def _const_ints(table, lines, name: str) -> list[int] | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_ints(lines[entry[0]])


def _const_string(table, lines, name: str) -> str | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_string(lines[entry[0]])


def _is_1x1_candidate(table, name) -> bool:
    entry = table.get(name)
    if entry is None or entry[1] != "conv" or entry[2] != "fp16":
        return False
    shape = _parse_shape(entry[3])
    if shape is None or len(shape) != 4 or shape[0] != 1:
        return False
    weight = table.get(entry[4].get("weight"))
    if weight is None:
        return False
    weight_shape = _parse_shape(weight[3])
    return (
        weight_shape is not None
        and len(weight_shape) == 4
        and weight_shape[2:] == (1, 1)
        and weight_shape[0] == shape[1]
        and weight_shape[1] == shape[1]
    )


def _classify(lines, table, index, name) -> LayoutEntry:
    _index, _op, _dtype, shape_text, args, _indent, _scalar = table[name]
    _ = _index
    shape = _parse_shape(shape_text)
    if shape is None or len(shape) != 4 or shape[0] != 1:
        return LayoutEntry(
            index, name, "REFUSED",
            f"conv output {shape_text} is not static rank-4 batch-1",
        )
    weight_name = args.get("weight")
    weight = table.get(weight_name) if weight_name else None
    weight_shape = _parse_shape(weight[3]) if weight is not None else None
    if (
        weight_shape is None
        or len(weight_shape) != 4
        or weight_shape[2:] != (1, 1)
    ):
        return LayoutEntry(
            index, name, "REFUSED",
            f"weight {None if weight is None else weight[3]} is not 1×1",
        )
    if weight_shape[0] != shape[1] or weight_shape[1] != shape[1]:
        return LayoutEntry(
            index, name, "REFUSED",
            f"weight {weight_shape} is not groups-1 [C, C, 1, 1] for {shape}",
        )
    strides = _const_ints(table, lines, args.get("strides"))
    dilations = _const_ints(table, lines, args.get("dilations"))
    pad = _const_ints(table, lines, args.get("pad"))
    groups = _const_ints(table, lines, args.get("groups"))
    pad_type = _const_string(table, lines, args.get("pad_type"))
    if strides != [1, 1]:
        return LayoutEntry(
            index, name, "REFUSED", f"strides {strides} are not [1, 1]"
        )
    if dilations != [1, 1]:
        return LayoutEntry(
            index, name, "REFUSED", f"dilations {dilations} are not [1, 1]"
        )
    if pad != [0, 0, 0, 0]:
        return LayoutEntry(
            index, name, "REFUSED", f"pad {pad} is not the valid/same zero vector"
        )
    if groups != [1]:
        return LayoutEntry(
            index, name, "REFUSED", f"groups {groups} is not scalar 1"
        )
    if pad_type not in ("valid", "same"):
        return LayoutEntry(
            index, name, "REFUSED",
            f"pad_type {pad_type!r} is not valid/same",
        )
    if args.get("bias") is None:
        return LayoutEntry(
            index, name, "REFUSED", "1×1 envelope neighbor is the bias form"
        )
    groups_entry = table[args["groups"]]
    pad_entry = table[args["pad_type"]]
    already = groups_entry[6] and pad_entry[6] and pad_type == "valid"
    if already:
        return LayoutEntry(
            index, name, "REFUSED",
            "already corpus string/int32 spelling; CHW is the table miss",
        )
    return LayoutEntry(
        index, name, "REWRITTEN",
        "corpus string pad_type / int32 groups; k1/st1/valid rank-4",
    )


def plan_conv_layout(mil_text: str) -> list[LayoutEntry]:
    """Classify every rank-4 1×1 conv."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    entries = []
    for name, (index, op, dtype, _shape, _args, _indent, _scalar) in table.items():
        if op != "conv" or dtype != "fp16":
            continue
        if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
            continue
        if not _is_1x1_candidate(table, name):
            continue
        entries.append(_classify(lines, table, index, name))
    return entries


def _scalar_string_line(indent: str, name: str, value: str) -> str:
    return (
        f"{indent}string {name} = const()"
        f'[name = string("{name}"), val = string("{value}")];'
    )


def _scalar_int_line(indent: str, name: str, value: int) -> str:
    return (
        f"{indent}int32 {name} = const()"
        f'[name = string("{name}"), val = int32({value})];'
    )


def rewrite_conv_layout(mil_text: str) -> tuple[str, LayoutReport]:
    """Replace leftover 1×1 pad_type/groups with corpus scalars.

    Leftover convs stay so the compiler names them.
    """
    lines = mil_text.splitlines()
    report = LayoutReport()
    table = _index_ops(lines)
    plan = plan_conv_layout(mil_text)
    replacements: dict[int, str] = {}
    for entry in plan:
        if entry.disposition != "REWRITTEN":
            report.refused.append(entry)
            continue
        args = table[entry.output_name][4]
        pad_name = args["pad_type"]
        groups_name = args["groups"]
        pad_index, _, _, _, _, pad_indent, _ = table[pad_name]
        groups_index, _, _, _, _, groups_indent, _ = table[groups_name]
        replacements[pad_index] = _scalar_string_line(
            pad_indent, pad_name, "valid"
        )
        replacements[groups_index] = _scalar_int_line(
            groups_indent, groups_name, 1
        )
        report.rewritten.append(entry.output_name)
    if not replacements:
        eol = "\n" if mil_text.endswith("\n") else ""
        return "\n".join(lines) + eol, report
    out = [
        replacements.get(index, line) for index, line in enumerate(lines)
    ]
    eol = "\n" if mil_text.endswith("\n") else ""
    return "\n".join(out) + eol, report
