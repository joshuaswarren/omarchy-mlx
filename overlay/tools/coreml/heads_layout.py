# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Birth Q/K/V as ``[B, H, T, D]`` so H13 never sees the middle-swap.

H13 rejects ``transpose(perm=[0,2,1,3], x=var_332)`` fp16
``[1, 8, 375, 128]`` as ``h13.nonfoldable-transpose``: the perm keeps
the fast axis but two ``add`` consumers need a materialized surface,
and there is no data-movement encoder. The producer is

    linear [B, T, H*D] → reshape [B, T, H, D] → transpose [B, H, T, D]

A reshape of that linear to ``[B, H, T, D]`` is a different packing
(token 0's ``H*D`` values would be read as head 0 across ``T``). A
weight-row permute cannot fix it: each linear row still belongs to one
token. ``numpy_naive_reshape`` / ``numpy_weight_permute_reshape``
return that counterexample.

The exact rewrite splits the projection into ``H`` linears of width
``D`` (contiguous fp16 row slices of the same BLOBFILE) and concats
the unit-expanded heads on axis 1. Each head's ``[B, T, D]`` is a
contiguous plane of ``[B, H, T, D]``, so concat is a permutation of
the same bytes as the transpose — not an approximation.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class HeadsLayoutError(RuntimeError):
    """A heads-layout rewrite is unsound here; the reason is named."""


HEADS_PERM = (0, 2, 1, 3)
_FP16_BYTES = 2

_DEF_LINE = re.compile(
    r"^(?P<indent>\s*)tensor<(?P<dtype>[a-z0-9]+),\s*"
    r"(?P<shape>\[[^\]]*\])>\s*(?P<name>[A-Za-z_][\w]*) = "
    r"(?P<op>[A-Za-z_][\w]*)\("
)
_FOOTER = re.compile(r"^\s*\} -> \((?P<returns>[^)]*)\)\s*;\s*$")
_ARG = re.compile(r"(?P<key>[A-Za-z_][\w]*)\s*=\s*(?P<value>[A-Za-z_][\w]*)")
_PARAM = re.compile(
    r"tensor<(?P<dtype>[a-z0-9]+),\s*(?P<shape>\[[^\]]*\])>\s*"
    r"(?P<name>[A-Za-z_][\w]*)"
)
_INT_VAL = re.compile(
    r"val = tensor<int32, \[[^\]]*\]>\((?P<body>[^)]*)\)"
)
_BLOB = re.compile(
    r'BLOBFILE\(path = string\("(?P<path>[^"]+)"\), '
    r"offset = uint64\((?P<offset>\d+)\)\)"
)


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one rank-4 heads middle-swap transpose."""

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
            "schema": "mlx-omarchy.heads-layout-rewrite.v1",
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


def numpy_naive_reshape(
    linear_out: np.ndarray, heads: int, head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """``transpose(reshape([B,T,H,D]))`` vs reshape ``[B,H,T,D]``.

    The second packing is wrong. Callers compare the pair; a mismatch
    is a named counterexample, not a rounding tolerance.
    """
    packed = _as_linear_out(linear_out, heads, head_dim)
    batch, time, _out = packed.shape
    original = np.transpose(
        packed.reshape(batch, time, heads, head_dim), (0, 2, 1, 3)
    )
    rewritten = packed.reshape(batch, heads, time, head_dim)
    return original, rewritten


def numpy_weight_permute_reshape(
    x: np.ndarray,
    weight: np.ndarray,
    row_perm: np.ndarray,
    heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Weight-row permute then reshape ``[B,H,T,D]`` vs the true transpose.

    Linear still emits one token per row, so no output-channel permute
    makes a ``[B, H, T, D]`` reshape equal the heads transpose.
    """
    x = np.asarray(x)
    weight = np.asarray(weight)
    row_perm = np.asarray(row_perm)
    if x.ndim != 3 or weight.ndim != 2:
        raise HeadsLayoutError(
            f"expected x [B,T,C] and weight [H*D,C], got {x.shape} {weight.shape}"
        )
    if weight.shape[0] != heads * head_dim:
        raise HeadsLayoutError(
            f"weight rows {weight.shape[0]} != heads*head_dim {heads * head_dim}"
        )
    if row_perm.shape != (weight.shape[0],):
        raise HeadsLayoutError(
            f"row_perm shape {row_perm.shape} != ({weight.shape[0]},)"
        )
    original_linear = _fp16_linear(x, weight, None)
    original = np.transpose(
        original_linear.reshape(x.shape[0], x.shape[1], heads, head_dim),
        (0, 2, 1, 3),
    )
    rewritten_linear = _fp16_linear(x, weight[row_perm], None)
    rewritten = rewritten_linear.reshape(
        x.shape[0], heads, x.shape[1], head_dim
    )
    return original, rewritten


def numpy_heads_stack(
    linear_out: np.ndarray, heads: int, head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Original transpose vs concat of per-head last-axis slices.

    ``linear_out`` is ``[B, T, H*D]``. Returns ``(original, rewritten)``
    in the same dtype. The stack is the definition of perm ``[0,2,1,3]``.
    """
    packed = _as_linear_out(linear_out, heads, head_dim)
    batch, time, _out = packed.shape
    original = np.transpose(
        packed.reshape(batch, time, heads, head_dim), (0, 2, 1, 3)
    )
    rewritten = np.stack(
        [
            packed[:, :, head * head_dim : (head + 1) * head_dim]
            for head in range(heads)
        ],
        axis=1,
    )
    return original, rewritten


def require_exact(original: np.ndarray, rewritten: np.ndarray, what: str) -> None:
    """Raise ``HeadsLayoutError`` naming the first mismatched lane."""
    if original.shape != rewritten.shape or original.dtype != rewritten.dtype:
        raise HeadsLayoutError(
            f"{what}: shape/dtype {original.shape}/{original.dtype} vs "
            f"{rewritten.shape}/{rewritten.dtype}"
        )
    left = original.view("<u2")
    right = rewritten.view("<u2")
    if np.array_equal(left, right):
        return
    mismatch = np.argwhere(left != right)
    index = tuple(int(v) for v in mismatch[0])
    raise HeadsLayoutError(
        f"inexact {what} at {list(index)}: original="
        f"{original[index]!r} rewritten={rewritten[index]!r}"
    )


def _as_linear_out(linear_out, heads, head_dim):
    packed = np.asarray(linear_out)
    if packed.ndim != 3:
        raise HeadsLayoutError(
            f"linear_out expects rank-3 [B, T, H*D], got {packed.shape}"
        )
    if packed.shape[2] != heads * head_dim:
        raise HeadsLayoutError(
            f"linear_out last dim {packed.shape[2]} != heads*head_dim "
            f"{heads * head_dim}"
        )
    return packed


def _fp16_linear(x, weight, bias):
    acc = x.astype(np.float32) @ np.asarray(weight, np.float32).T
    if bias is not None:
        acc = acc + np.asarray(bias, np.float32)
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


def _shape_text(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


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


def _normalize_perm(perm: list[int], rank: int) -> tuple[int, ...] | None:
    if len(perm) != rank:
        return None
    out = []
    for axis in perm:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        out.append(axis)
    return tuple(out)


def _index_ops(lines: list[str]):
    table = {}
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line.rstrip("\n"))
        if not match:
            continue
        args = dict(_ARG.findall(_code_of(line).split("(", 1)[1]))
        table[match.group("name")] = (
            index,
            match.group("op"),
            match.group("dtype"),
            match.group("shape"),
            args,
            match.group("indent"),
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
                    ),
                )
    return table


def _const_ints(table, lines, name: str) -> list[int] | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_ints(lines[entry[0]])


def _blob(table, lines, name: str):
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    match = _BLOB.search(lines[entry[0]])
    if match is None:
        return None
    return match.group("path"), int(match.group("offset"))


_BLOB_MAGIC = 0xDEADBEEF
_BLOB_HEADER = 24


def _resolve_blob_path(model_root: Path, mil_path: str) -> Path:
    prefix = "@model_path/"
    if mil_path.startswith(prefix):
        mil_path = mil_path[len(prefix):]
    return Path(model_root) / mil_path


def _read_blob_header(blob_path: Path, header_off: int) -> tuple[int, int]:
    with blob_path.open("rb") as handle:
        handle.seek(header_off)
        header = handle.read(_BLOB_HEADER)
    if len(header) != _BLOB_HEADER:
        raise HeadsLayoutError(
            f"BLOBFILE header at {header_off} is truncated in {blob_path}"
        )
    magic, payload_length, payload_offset = struct.unpack_from(
        "<I4xQQ", header
    )
    if magic != _BLOB_MAGIC:
        raise HeadsLayoutError(
            f"BLOBFILE header at {header_off} has magic {magic:#x}"
        )
    return int(payload_length), int(payload_offset)


def _append_blob_header(
    blob_path: Path, payload_offset: int, payload_length: int
) -> int:
    header = struct.pack(
        "<I4xQQ", _BLOB_MAGIC, payload_length, payload_offset
    )
    with blob_path.open("ab") as handle:
        offset = handle.tell()
        handle.write(header)
    return offset


def _head_blob_offset(
    mil_path: str,
    header_off: int,
    head: int,
    stride: int,
    payload_bytes: int,
    model_root: Path | None,
) -> int:
    """Header offset for one head slice of a BLOBFILE const.

    Without ``model_root`` the MIL keeps arithmetic offsets (host tests).
    With a root, each slice gets its own 24-byte DEADBEEF header pointing
    at the existing payload; BLOBFILE offsets are headers, not bytes.
    """
    if model_root is None:
        return header_off + head * stride
    blob_path = _resolve_blob_path(model_root, mil_path)
    _length, payload_offset = _read_blob_header(blob_path, header_off)
    return _append_blob_header(
        blob_path, payload_offset + head * stride, payload_bytes
    )


def _consumers_of(lines: list[str], name: str) -> list[int]:
    pattern = re.compile(r"(?<![\w])" + re.escape(name) + r"(?![\w])")
    found = []
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line.rstrip("\n"))
        if not match or match.group("name") == name:
            continue
        if pattern.search(_code_of(line)):
            found.append(index)
    return found


def _fresh(taken: set[str], prefix: str) -> str:
    if prefix not in taken:
        taken.add(prefix)
        return prefix
    counter = 1
    while f"{prefix}_{counter}" in taken:
        counter += 1
    name = f"{prefix}_{counter}"
    taken.add(name)
    return name


def _classify(lines, table, index, name) -> LayoutEntry:
    _index, op, dtype, shape_text, args, _indent = table[name]
    _ = _index
    if op != "transpose" or dtype != "fp16":
        raise HeadsLayoutError(f"not a heads-layout candidate: {name}")
    shape = _parse_shape(shape_text)
    if shape is None or len(shape) != 4 or any(dim <= 0 for dim in shape):
        return LayoutEntry(
            index, name, "REFUSED",
            f"transpose output shape {shape_text} is not static [B,H,T,D]",
        )
    batch, heads, time, head_dim = shape
    if heads <= 1:
        return LayoutEntry(
            index, name, "REFUSED",
            "H<=1 is a unit-axis move, not this rewrite",
        )
    perm_name = args.get("perm")
    perm = _const_ints(table, lines, perm_name) if perm_name else None
    if perm is None or _normalize_perm(perm, 4) != HEADS_PERM:
        return LayoutEntry(
            index, name, "REFUSED",
            f"perm is {perm}; only {list(HEADS_PERM)} (negatives ok) "
            "is this rewrite",
        )
    source_name = args.get("x")
    source = table.get(source_name) if source_name else None
    if source is None or source[1] != "reshape":
        return LayoutEntry(
            index, name, "REFUSED",
            "transpose input is not a reshape of a projection",
        )
    reshaped = _parse_shape(source[3])
    if reshaped != (batch, time, heads, head_dim):
        return LayoutEntry(
            index, name, "REFUSED",
            f"reshape {source[3]} is not [B,T,H,D] for transpose {shape_text}",
        )
    linear_name = source[4].get("x")
    linear = table.get(linear_name) if linear_name else None
    if linear is None or linear[1] != "linear":
        return LayoutEntry(
            index, name, "REFUSED",
            "reshape input is not a linear; a per-head projection cannot "
            "be emitted without inventing a packing",
        )
    linear_shape = _parse_shape(linear[3])
    if linear_shape != (batch, time, heads * head_dim):
        return LayoutEntry(
            index, name, "REFUSED",
            f"linear {linear[3]} is not [B,T,H*D] for {shape_text}",
        )
    weight_name = linear[4].get("weight")
    bias_name = linear[4].get("bias")
    input_name = linear[4].get("x")
    if not weight_name or not bias_name or not input_name:
        return LayoutEntry(
            index, name, "REFUSED",
            "linear is missing weight, bias, or x",
        )
    weight = table.get(weight_name)
    if weight is None:
        return LayoutEntry(
            index, name, "REFUSED", "linear weight has no defining line"
        )
    weight_shape = _parse_shape(weight[3])
    if (
        weight_shape is None
        or len(weight_shape) != 2
        or weight_shape[0] != heads * head_dim
    ):
        return LayoutEntry(
            index, name, "REFUSED",
            f"weight shape {weight[3]} is not [H*D, C]",
        )
    if _blob(table, lines, weight_name) is None:
        return LayoutEntry(
            index, name, "REFUSED",
            "weight is not a BLOBFILE; per-head row slices need a byte offset",
        )
    if _blob(table, lines, bias_name) is None:
        return LayoutEntry(
            index, name, "REFUSED",
            "bias is not a BLOBFILE; per-head slices need a byte offset",
        )
    return LayoutEntry(
        index, name, "REWRITTEN",
        "per-head linear + concat axis=1 emits [B,H,T,D]; "
        "values match transpose(reshape(linear))",
    )


def plan_heads_layout(mil_text: str) -> list[LayoutEntry]:
    """Classify every rank-4 heads middle-swap transpose."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    entries = []
    for name, (index, op, dtype, shape_text, args, _indent) in table.items():
        if op != "transpose" or dtype != "fp16":
            continue
        if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
            continue
        shape = _parse_shape(shape_text)
        if shape is None or len(shape) != 4:
            continue
        perm_name = args.get("perm")
        perm = _const_ints(table, lines, perm_name) if perm_name else None
        if perm is None or _normalize_perm(perm, 4) != HEADS_PERM:
            continue
        entries.append(_classify(lines, table, index, name))
    return entries


def _blob_const(indent, dtype, shape, name, path, offset) -> str:
    shape_text = _shape_text(shape)
    return (
        f"{indent}tensor<{dtype}, {shape_text}> {name} = const()"
        f"[val = tensor<{dtype}, {shape_text}>("
        f"BLOBFILE(path = string(\"{path}\"), offset = uint64({offset})))];"
    )


def _emit_rewrite(
    lines, table, taken, name, model_root: Path | None,
) -> tuple[list[str], str]:
    index, _op, dtype, shape_text, args, indent = table[name]
    shape = _parse_shape(shape_text)
    batch, heads, time, head_dim = shape
    reshape_name = args["x"]
    linear_name = table[reshape_name][4]["x"]
    linear_args = table[linear_name][4]
    weight_name = linear_args["weight"]
    bias_name = linear_args["bias"]
    input_name = linear_args["x"]
    weight_shape = _parse_shape(table[weight_name][3])
    channels = weight_shape[1]
    weight_path, weight_off = _blob(table, lines, weight_name)
    bias_path, bias_off = _blob(table, lines, bias_name)
    axes_name = _fresh(taken, f"{name}_heads_axes")
    axis_name = _fresh(taken, f"{name}_heads_axis")
    inserted = [
        (
            f"{indent}tensor<int32, [1]> {axes_name} = const()"
            f"[name = tensor<string, []>(\"{axes_name}\"), "
            f"val = tensor<int32, [1]>(1)];"
        ),
        (
            f"{indent}tensor<int32, []> {axis_name} = const()"
            f"[name = tensor<string, []>(\"{axis_name}\"), "
            f"val = tensor<int32, []>(1)];"
        ),
    ]
    head_exps = []
    weight_stride = head_dim * channels * _FP16_BYTES
    bias_stride = head_dim * _FP16_BYTES
    for head in range(heads):
        w_name = _fresh(taken, f"{name}_h{head}_w")
        b_name = _fresh(taken, f"{name}_h{head}_b")
        lin_name = _fresh(taken, f"{name}_h{head}_lin")
        exp_name = _fresh(taken, f"{name}_h{head}_exp")
        inserted.append(
            _blob_const(
                indent, "fp16", (head_dim, channels), w_name,
                weight_path,
                _head_blob_offset(
                    weight_path, weight_off, head, weight_stride,
                    weight_stride, model_root,
                ),
            )
        )
        inserted.append(
            _blob_const(
                indent, "fp16", (head_dim,), b_name,
                bias_path,
                _head_blob_offset(
                    bias_path, bias_off, head, bias_stride,
                    bias_stride, model_root,
                ),
            )
        )
        inserted.append(
            f"{indent}tensor<fp16, {_shape_text((batch, time, head_dim))}> "
            f"{lin_name} = linear(bias = {b_name}, weight = {w_name}, "
            f"x = {input_name})"
            f"[name = tensor<string, []>(\"{lin_name}\")];"
        )
        inserted.append(
            f"{indent}tensor<fp16, "
            f"{_shape_text((batch, 1, time, head_dim))}> {exp_name} = "
            f"expand_dims(axes = {axes_name}, x = {lin_name})"
            f"[name = tensor<string, []>(\"{exp_name}\")];"
        )
        head_exps.append(exp_name)
    args_text = ", ".join(
        [f"axis = {axis_name}"]
        + [f"x{head} = {exp}" for head, exp in enumerate(head_exps)]
    )
    replacement = (
        f"{indent}tensor<{dtype}, {shape_text}> {name} = "
        f"concat({args_text})"
        f"[name = tensor<string, []>(\"{name}_heads_concat\")];"
    )
    return inserted, replacement


def rewrite_heads_layout(
    mil_text: str, model_root: Path | None = None,
) -> tuple[str, LayoutReport]:
    """Replace each sound heads transpose with per-head linear + concat.

    Leftover transposes stay in the graph so the compiler names them.
    ``model_root`` clones BLOBFILE headers for head slices; omit it in
    host tests that only check MIL text.
    """
    lines = mil_text.splitlines()
    report = LayoutReport()
    table = _index_ops(lines)
    plan = plan_heads_layout(mil_text)
    insertions: dict[int, list[str]] = {}
    replacements: dict[int, str] = {}
    deletions: set[int] = set()
    taken = set(table)
    for entry in plan:
        if entry.disposition != "REWRITTEN":
            report.refused.append(entry)
            continue
        inserted, replacement = _emit_rewrite(
            lines, table, taken, entry.output_name, model_root
        )
        transpose_index = table[entry.output_name][0]
        insertions[transpose_index] = inserted
        replacements[transpose_index] = replacement
        reshape_name = table[entry.output_name][4]["x"]
        reshape_index = table[reshape_name][0]
        if _consumers_of(lines, reshape_name) == [transpose_index]:
            deletions.add(reshape_index)
            linear_name = table[reshape_name][4]["x"]
            linear_index = table[linear_name][0]
            if _consumers_of(lines, linear_name) == [reshape_index]:
                deletions.add(linear_index)
        report.rewritten.append(entry.output_name)
    if not replacements:
        eol = "\n" if mil_text.endswith("\n") else ""
        return "\n".join(lines) + eol, report
    out: list[str] = []
    for index, line in enumerate(lines):
        for text in insertions.get(index, []):
            out.append(text)
        if index in deletions:
            continue
        out.append(replacements.get(index, line))
    eol = "\n" if mil_text.endswith("\n") else ""
    return "\n".join(out) + eol, report
