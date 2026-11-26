# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Decompose a noncontiguous ``slice_by_index`` into unit chunks + concat.

H13 rejects ``slice_by_index`` on fp16 ``[1, 8, 749, 375]`` as
``h13.noncontiguous-slice``: eight chunks of 280875 elements spaced
281250 apart (``x[:, :, 1:, :]`` on ``[1, 8, 750, 375]``). It then
rejects ``matrix_bd_3`` fp16 ``[1, 8, 375, 375]``: 3000 chunks of 375
spaced 749 (``x[:, :, :, :375]`` on ``[1, 8, 375, 749]``).

H13 allows at most one sliced dimension per binding. The rewrite peels
every wrapping dim as a one-dim unit plane, applies the remaining
one-dim inner slice, and concats back. Two-dim cuts and drop-last /
suffix-prefix swaps are named counterexamples, not approximations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class SliceLayoutError(RuntimeError):
    """A slice-layout rewrite is unsound here; the reason is named."""


_DEF_LINE = re.compile(
    r"^(?P<indent>\s*)tensor<(?P<dtype>[a-z0-9]+),\s*"
    r"(?P<shape>\[[^\]]*\])>\s*(?P<name>[A-Za-z_][\w]*) = "
    r"(?P<op>[A-Za-z_][\w]*)\("
)
_ARG = re.compile(r"(?P<key>[A-Za-z_][\w]*)\s*=\s*(?P<value>[A-Za-z_][\w]*)")
_PARAM = re.compile(
    r"tensor<(?P<dtype>[a-z0-9]+),\s*(?P<shape>\[[^\]]*\])>\s*"
    r"(?P<name>[A-Za-z_][\w]*)"
)
_INT_VAL = re.compile(
    r"val = tensor<int32, \[[^\]]*\]>\((?P<body>[^)]*)\)"
)
_BOOL_VAL = re.compile(
    r"val = tensor<bool, \[[^\]]*\]>\((?P<body>[^)]*)\)"
)


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one noncontiguous slice_by_index."""

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
            "schema": "mlx-omarchy.slice-layout-rewrite.v1",
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


def numpy_relpos_drop_first(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Original ``x[:, :, 1:, :]`` vs concat of per-head unit slices.

    ``source`` is ``[B, H, T, D]``. Returns ``(original, rewritten)``
    in the same dtype. The stack is the definition of that slice.
    """
    packed = _as_relpos(source)
    original = packed[:, :, 1:, :]
    rewritten = np.concatenate(
        [packed[:, head : head + 1, 1:, :] for head in range(packed.shape[1])],
        axis=1,
    )
    return original, rewritten


def numpy_relpos_drop_last(source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``x[:, :, 1:, :]`` vs ``x[:, :, :T-1, :]``.

    Same output shape, different packing. Callers compare the pair; a
    mismatch is a named counterexample, not a rounding tolerance.
    """
    packed = _as_relpos(source)
    return packed[:, :, 1:, :], packed[:, :, :-1, :]

def numpy_last_dim_prefix(source: np.ndarray, keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Original ``x[:, :, :, :keep]`` vs nested per-head per-row concat.

    ``source`` is ``[B, H, T, D]``. Each ``[B, 1, 1, keep]`` piece is
    one contiguous chunk of ``keep``; concat axis 2 then 1 restores
    ``[B, H, T, keep]``.
    """
    packed = np.asarray(source)
    if packed.ndim != 4:
        raise SliceLayoutError(
            f"last-dim prefix expects rank-4 [B, H, T, D], got {packed.shape}"
        )
    if keep <= 0 or keep > packed.shape[3]:
        raise SliceLayoutError(
            f"keep={keep} is outside last dim {packed.shape[3]}"
        )
    original = packed[:, :, :, :keep]
    heads = []
    for head in range(packed.shape[1]):
        rows = [
            packed[:, head : head + 1, time : time + 1, :keep]
            for time in range(packed.shape[2])
        ]
        heads.append(np.concatenate(rows, axis=2))
    rewritten = np.concatenate(heads, axis=1)
    return original, rewritten


def numpy_last_dim_suffix(source: np.ndarray, keep: int) -> tuple[np.ndarray, np.ndarray]:
    """``x[:, :, :, :keep]`` vs ``x[:, :, :, -keep:]``.

    Same output shape, different packing when ``keep < D``.
    """
    packed = np.asarray(source)
    if packed.ndim != 4:
        raise SliceLayoutError(
            f"last-dim suffix expects rank-4 [B, H, T, D], got {packed.shape}"
        )
    return packed[:, :, :, :keep], packed[:, :, :, -keep:]


def require_exact(original: np.ndarray, rewritten: np.ndarray, what: str) -> None:
    """Raise ``SliceLayoutError`` naming the first mismatched lane."""
    if original.shape != rewritten.shape or original.dtype != rewritten.dtype:
        raise SliceLayoutError(
            f"{what}: shape/dtype {original.shape}/{original.dtype} vs "
            f"{rewritten.shape}/{rewritten.dtype}"
        )
    left = original.view("<u2")
    right = rewritten.view("<u2")
    if np.array_equal(left, right):
        return
    mismatch = np.argwhere(left != right)
    index = tuple(int(v) for v in mismatch[0])
    raise SliceLayoutError(
        f"inexact {what} at {list(index)}: original="
        f"{original[index]!r} rewritten={rewritten[index]!r}"
    )


def _as_relpos(source) -> np.ndarray:
    packed = np.asarray(source)
    if packed.ndim != 4:
        raise SliceLayoutError(
            f"rel-pos slice expects rank-4 [B, H, T, D], got {packed.shape}"
        )
    if packed.shape[2] < 2:
        raise SliceLayoutError(
            f"rel-pos T={packed.shape[2]} cannot drop the first row"
        )
    return packed


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


def _parse_bools(line: str) -> list[bool] | None:
    match = _BOOL_VAL.search(line)
    if match is None:
        return None
    body = match.group("body").strip()
    try:
        if body.startswith("[") and body.endswith("]"):
            inner = body[1:-1].strip()
            if not inner:
                return []
            return [_as_bool(part.strip()) for part in inner.split(",")]
        return [_as_bool(body)]
    except ValueError:
        return None


def _as_bool(text: str) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(text)


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


def _const_bools(table, lines, name: str) -> list[bool] | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_bools(lines[entry[0]])


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


def _resolve_index(value: int, dim: int) -> int | None:
    if value < 0:
        value += dim
    if value < 0 or value > dim:
        return None
    return value


def _resolved_ranges(source_shape, begin, end, begin_mask, end_mask, stride):
    rank = len(source_shape)
    if (
        len(begin) != rank
        or len(end) != rank
        or (begin_mask is not None and len(begin_mask) != rank)
        or (end_mask is not None and len(end_mask) != rank)
        or (stride is not None and len(stride) != rank)
    ):
        return None
    ranges = []
    for axis, dim in enumerate(source_shape):
        step = 1 if stride is None else stride[axis]
        if step != 1:
            return None
        if begin_mask is not None and begin_mask[axis]:
            start = 0
        else:
            start = _resolve_index(begin[axis], dim)
        if end_mask is not None and end_mask[axis]:
            stop = dim
        else:
            stop = _resolve_index(end[axis], dim)
        if start is None or stop is None or start > stop:
            return None
        out = stop - start
        full = start == 0 and stop == dim
        ranges.append((start, stop, out, full))
    return ranges


def _inner_partial(ranges) -> int | None:
    """Boundary dim of the innermost run, or None if the slice is full."""
    inner_full_from = len(ranges)
    for axis in range(len(ranges) - 1, -1, -1):
        if ranges[axis][3]:
            inner_full_from = axis
        else:
            break
    if inner_full_from == 0:
        return None
    return inner_full_from - 1


def _is_contiguous(source_shape, ranges) -> bool:
    partial = _inner_partial(ranges)
    if partial is None:
        return True
    n_chunks = 1
    for axis in range(partial):
        n_chunks *= ranges[axis][2]
    if n_chunks == 1:
        return True
    spacing = 1
    chunk_elems = 1
    for axis in range(partial, len(ranges)):
        spacing *= source_shape[axis]
        chunk_elems *= ranges[axis][2]
    return spacing == chunk_elems


def _wrapping_axes(ranges) -> list[int]:
    """Wrapping dims slower than the innermost partial run."""
    partial = _inner_partial(ranges)
    if partial is None:
        return []
    return [axis for axis in range(partial) if ranges[axis][2] > 1]



def _classify(lines, table, index, name) -> LayoutEntry | None:
    _index, op, dtype, shape_text, args, _indent = table[name]
    _ = _index
    if op != "slice_by_index" or dtype != "fp16":
        raise SliceLayoutError(f"not a slice-layout candidate: {name}")
    if "squeeze_mask" in args:
        return LayoutEntry(
            index, name, "REFUSED",
            "squeeze_mask changes rank; this rewrite keeps the unit axis",
        )
    shape = _parse_shape(shape_text)
    if shape is None or len(shape) != 4 or any(dim <= 0 for dim in shape):
        return LayoutEntry(
            index, name, "REFUSED",
            f"slice output shape {shape_text} is not static rank-4",
        )
    source_name = args.get("x")
    source = table.get(source_name) if source_name else None
    if source is None:
        return LayoutEntry(
            index, name, "REFUSED", "slice input has no defining line"
        )
    source_shape = _parse_shape(source[3])
    if source_shape is None or len(source_shape) != 4:
        return LayoutEntry(
            index, name, "REFUSED",
            f"slice input shape {source[3]} is not static rank-4",
        )
    begin_name = args.get("begin")
    end_name = args.get("end")
    begin = _const_ints(table, lines, begin_name) if begin_name else None
    end = _const_ints(table, lines, end_name) if end_name else None
    if begin is None or end is None:
        return LayoutEntry(
            index, name, "REFUSED", "begin/end are not const int32 vectors"
        )
    begin_mask = (
        _const_bools(table, lines, args["begin_mask"])
        if "begin_mask" in args
        else None
    )
    end_mask = (
        _const_bools(table, lines, args["end_mask"])
        if "end_mask" in args
        else None
    )
    if "begin_mask" in args and begin_mask is None:
        return LayoutEntry(
            index, name, "REFUSED", "begin_mask is not a const bool vector"
        )
    if "end_mask" in args and end_mask is None:
        return LayoutEntry(
            index, name, "REFUSED", "end_mask is not a const bool vector"
        )
    stride = (
        _const_ints(table, lines, args["stride"])
        if "stride" in args
        else None
    )
    if "stride" in args and stride is None:
        return LayoutEntry(
            index, name, "REFUSED", "stride is not a const int32 vector"
        )
    ranges = _resolved_ranges(
        source_shape, begin, end, begin_mask, end_mask, stride
    )
    if ranges is None:
        return LayoutEntry(
            index, name, "REFUSED",
            "slice bounds are not unit-stride static ranges"
        )
    out_shape = tuple(item[2] for item in ranges)
    if out_shape != shape:
        return LayoutEntry(
            index, name, "REFUSED",
            f"resolved slice shape {list(out_shape)} != declared {shape_text}",
        )
    if _is_contiguous(source_shape, ranges):
        return None
    wrapping = _wrapping_axes(ranges)
    if not wrapping:
        return LayoutEntry(
            index, name, "REFUSED",
            "noncontiguous slice has no wrapping dim to peel into "
            "one-dim unit planes",
        )
    n_chunks = 1
    for axis in wrapping:
        n_chunks *= ranges[axis][2]
    partial = _inner_partial(ranges)
    chunk_elems = 1
    spacing = 1
    for axis in range(partial, 4):
        chunk_elems *= ranges[axis][2]
        spacing *= source_shape[axis]
    return LayoutEntry(
        index, name, "REWRITTEN",
        f"per-chunk slice + nested concat axes={wrapping} emits "
        f"{list(shape)}; {n_chunks} chunks of {chunk_elems} spaced "
        f"{spacing} apart",
    )



def plan_slice_layout(mil_text: str) -> list[LayoutEntry]:
    """Classify every noncontiguous fp16 rank-4 slice_by_index."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    entries = []
    for name, (index, op, dtype, shape_text, _args, _indent) in table.items():
        if op != "slice_by_index" or dtype != "fp16":
            continue
        if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
            continue
        shape = _parse_shape(shape_text)
        if shape is None or len(shape) != 4:
            continue
        entry = _classify(lines, table, index, name)
        if entry is not None:
            entries.append(entry)
    return entries


def _int_vec(indent: str, name: str, values: list[int]) -> str:
    return (
        f"{indent}tensor<int32, [4]> {name} = const()"
        f"[name = tensor<string, []>(\"{name}\"), "
        f"val = tensor<int32, [4]>({_shape_text(tuple(values))})];"
    )


def _int_scalar(indent: str, name: str, value: int) -> str:
    return (
        f"{indent}tensor<int32, []> {name} = const()"
        f"[name = tensor<string, []>(\"{name}\"), "
        f"val = tensor<int32, []>({value})];"
    )


def _slice_line(indent, dtype, shape, name, begin, end, source) -> str:
    return (
        f"{indent}tensor<{dtype}, {_shape_text(tuple(shape))}> {name} = "
        f"slice_by_index(begin = {begin}, end = {end}, x = {source})"
        f"[name = tensor<string, []>(\"{name}\")];"
    )


def _concat_line(indent, dtype, shape, name, axis_name, pieces) -> str:
    args = ", ".join(
        [f"axis = {axis_name}"]
        + [f"x{offset} = {piece}" for offset, piece in enumerate(pieces)]
    )
    return (
        f"{indent}tensor<{dtype}, {_shape_text(tuple(shape))}> {name} = "
        f"concat({args})"
        f"[name = tensor<string, []>(\"{name}_slice_concat\")];"
    )

_MAX_CONCAT = 8  # 8-way heads concat already compiled; 375-way did not


def _concat_tree(
    indent, taken, prefix, dtype, axis, axis_name, pieces, piece_shape, out_name
):
    """Concat ``pieces`` along ``axis`` in groups of at most ``_MAX_CONCAT``."""
    inserted = []
    names = list(pieces)
    sizes = [piece_shape[axis]] * len(pieces)
    base_shape = list(piece_shape)
    round_i = 0
    while len(names) > 1:
        new_names = []
        new_sizes = []
        last_round = len(names) <= _MAX_CONCAT
        for i in range(0, len(names), _MAX_CONCAT):
            group = names[i : i + _MAX_CONCAT]
            group_sizes = sizes[i : i + _MAX_CONCAT]
            if len(group) == 1:
                new_names.append(group[0])
                new_sizes.append(group_sizes[0])
                continue
            out_s = list(base_shape)
            out_s[axis] = sum(group_sizes)
            n_name = (
                out_name
                if last_round and i == 0
                else _fresh(taken, f"{prefix}_c{round_i}_{i}")
            )
            inserted.append(
                _concat_line(indent, dtype, out_s, n_name, axis_name, group)
            )
            new_names.append(n_name)
            new_sizes.append(out_s[axis])
        names = new_names
        sizes = new_sizes
        round_i += 1
    return names[0], inserted



def _emit_rewrite(lines, table, taken, name) -> tuple[list[str], str]:
    index, _op, dtype, shape_text, args, indent = table[name]
    _ = index
    source_name = args["x"]
    source_shape = _parse_shape(table[source_name][3])
    begin = _const_ints(table, lines, args["begin"])
    end = _const_ints(table, lines, args["end"])
    begin_mask = (
        _const_bools(table, lines, args["begin_mask"])
        if "begin_mask" in args
        else None
    )
    end_mask = (
        _const_bools(table, lines, args["end_mask"])
        if "end_mask" in args
        else None
    )
    stride = (
        _const_ints(table, lines, args["stride"])
        if "stride" in args
        else None
    )
    ranges = _resolved_ranges(
        source_shape, begin, end, begin_mask, end_mask, stride
    )

    def peel(prefix, tensor_name, tensor_shape, tensor_ranges, out_name):
        wrapping = _wrapping_axes(tensor_ranges)
        if not wrapping:
            begin_vals = [item[0] for item in tensor_ranges]
            end_vals = [item[1] for item in tensor_ranges]
            out_shape = [item[2] for item in tensor_ranges]
            result = out_name or _fresh(taken, f"{prefix}_inner")
            b_name = _fresh(taken, f"{prefix}_inner_begin")
            e_name = _fresh(taken, f"{prefix}_inner_end")
            inserted = [
                _int_vec(indent, b_name, begin_vals),
                _int_vec(indent, e_name, end_vals),
                _slice_line(
                    indent, dtype, out_shape, result,
                    b_name, e_name, tensor_name,
                ),
            ]
            return result, inserted
        axis = wrapping[0]
        count = tensor_ranges[axis][2]
        start = tensor_ranges[axis][0]
        axis_name = _fresh(taken, f"{prefix}_axis")
        inserted = [_int_scalar(indent, axis_name, axis)]
        pieces = []
        for offset in range(count):
            head = start + offset
            plane_begin = [
                head if dim == axis else 0 for dim in range(4)
            ]
            plane_end = [
                head + 1 if dim == axis else tensor_shape[dim]
                for dim in range(4)
            ]
            plane_shape = list(tensor_shape)
            plane_shape[axis] = 1
            b_name = _fresh(taken, f"{prefix}_h{offset}_begin")
            e_name = _fresh(taken, f"{prefix}_h{offset}_end")
            p_name = _fresh(taken, f"{prefix}_h{offset}_plane")
            inserted.append(_int_vec(indent, b_name, plane_begin))
            inserted.append(_int_vec(indent, e_name, plane_end))
            inserted.append(
                _slice_line(
                    indent, dtype, plane_shape, p_name,
                    b_name, e_name, tensor_name,
                )
            )
            child_ranges = [
                (0, 1, 1, True) if dim == axis else item
                for dim, item in enumerate(tensor_ranges)
            ]
            child_name, child_ins = peel(
                f"{prefix}_h{offset}",
                p_name,
                tuple(plane_shape),
                child_ranges,
                None,
            )
            inserted.extend(child_ins)
            pieces.append(child_name)
        piece_shape = [item[2] for item in child_ranges]
        result = out_name or _fresh(taken, f"{prefix}_concat")
        tree_name, tree_ins = _concat_tree(
            indent, taken, prefix, dtype, axis, axis_name,
            pieces, piece_shape, result,
        )
        _ = tree_name
        inserted.extend(tree_ins)
        return result, inserted

    result, inserted = peel(
        name, source_name, source_shape, ranges, name
    )
    _ = result
    replacement = inserted[-1]
    return inserted[:-1], replacement


def rewrite_slice_layout(mil_text: str) -> tuple[str, LayoutReport]:
    """Replace each sound noncontiguous slice with unit slices + concat.

    Leftover slices stay in the graph so the compiler names them.
    """
    lines = mil_text.splitlines()
    report = LayoutReport()
    table = _index_ops(lines)
    plan = plan_slice_layout(mil_text)
    insertions: dict[int, list[str]] = {}
    replacements: dict[int, str] = {}
    taken = set(table)
    for entry in plan:
        if entry.disposition != "REWRITTEN":
            report.refused.append(entry)
            continue
        inserted, replacement = _emit_rewrite(
            lines, table, taken, entry.output_name
        )
        slice_index = table[entry.output_name][0]
        insertions[slice_index] = inserted
        replacements[slice_index] = replacement
        report.rewritten.append(entry.output_name)
    if not replacements:
        eol = "\n" if mil_text.endswith("\n") else ""
        return "\n".join(lines) + eol, report
    out: list[str] = []
    for index, line in enumerate(lines):
        for text in insertions.get(index, []):
            out.append(text)
        out.append(replacements.get(index, line))
    eol = "\n" if mil_text.endswith("\n") else ""
    return "\n".join(out) + eol, report
