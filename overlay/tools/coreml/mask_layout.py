# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Emit the encoder mask already in the mul's layout.

H13 rejects ``transpose(perm=[0,2,1], x=attention_mask_3)`` bool
``[1, N, N]`` as ``h13.nonfoldable-transpose``: a tail-swap moves the
storage-fastest axis, and the consumer is ``mul`` (or ``logical_and``,
which the compiler folds to ``mul``) with no transpose flag.

The producer is a vector mask ``[B, N]`` expanded to ``[B, 1, N]`` and
tiled ``[1, N, 1]``. The tail-swap of that tile is the same tensor as
expanding the vector to ``[B, N, 1]`` and tiling ``[1, 1, N]``. Both
``expand_dims`` and ``tile`` on bool already compile on this graph
(attempt 3), so the rewrite deletes the transpose and H13 never sees
it. Elementwise ``mul``/``logical_and`` of the two tiles equals the
original ``mul(tile, transpose(tile))`` at every position — a
permutation of operands, not an approximation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class MaskLayoutError(RuntimeError):
    """A mask-layout rewrite is unsound here; the reason is named."""


TAIL_SWAP_RANK3 = (0, 2, 1)
ELEMENTWISE_CONSUMERS = frozenset({"mul", "logical_and"})

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


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one rank-3 tail-swap transpose."""

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
            "schema": "mlx-omarchy.mask-layout-rewrite.v1",
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


def numpy_tail_swap_mul(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Original ``mul(tile, transpose(tile))`` vs complementary-tile mul.

    ``mask`` is ``[B, N]``. Returns ``(original, rewritten)`` with the
    same dtype. Callers compare them; a mismatch is a named
    counterexample, not a rounding tolerance.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise MaskLayoutError(
            f"numpy_tail_swap_mul expects rank-2 [B, N], got {mask.shape}"
        )
    length = mask.shape[1]
    col = np.tile(np.expand_dims(mask, 1), (1, length, 1))
    original = col * np.transpose(col, (0, 2, 1))
    row = np.tile(np.expand_dims(mask, 2), (1, 1, length))
    rewritten = col * row
    return original, rewritten


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


def _returns_of(lines: list[str]) -> set[str]:
    for line in lines:
        footer = _FOOTER.match(line.rstrip("\n"))
        if footer:
            return {
                name.strip()
                for name in footer.group("returns").split(",")
                if name.strip()
            }
    return set()


def _const_ints(table, lines, name: str) -> list[int] | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_ints(lines[entry[0]])


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


def _complement(expanded: tuple[int, ...], tiled: tuple[int, ...]):
    """Complementary expand shape + tile reps for a tail-swap of ``tiled``.

    ``expanded`` is the expand_dims output, ``tiled`` the tile output.
    Returns ``(axes, expand_shape, reps)`` or None.
    """
    if len(expanded) != 3 or len(tiled) != 3:
        return None
    batch, rows, cols = tiled
    if rows != cols or batch != expanded[0]:
        return None
    if expanded == (batch, 1, cols):
        return (2,), (batch, rows, 1), (1, 1, cols)
    if expanded == (batch, rows, 1):
        return (1,), (batch, 1, cols), (1, rows, 1)
    return None


def _classify(lines, table, returns, index, name) -> LayoutEntry:
    _, op, dtype, shape_text, args, _indent = table[name]
    if op != "transpose" or dtype not in ("bool", "fp16"):
        raise MaskLayoutError(f"not a mask-layout candidate: {name}")
    shape = _parse_shape(shape_text)
    if shape is None or len(shape) != 3 or shape[1] != shape[2] or shape[1] <= 0:
        return LayoutEntry(
            index, name, "REFUSED",
            f"transpose output shape {shape_text} is not static [B, N, N]",
        )
    perm_name = args.get("perm")
    perm = _const_ints(table, lines, perm_name) if perm_name else None
    if perm != list(TAIL_SWAP_RANK3):
        return LayoutEntry(
            index, name, "REFUSED",
            f"perm is {perm}; only rank-3 tail-swap {list(TAIL_SWAP_RANK3)} "
            "is this rewrite",
        )
    if name in returns:
        return LayoutEntry(
            index, name, "REFUSED",
            "transpose output is a graph return; no consumer to emit into",
        )
    consumers = _consumers_of(lines, name)
    if not consumers:
        return LayoutEntry(
            index, name, "REFUSED",
            "transpose has no consumer; deleting it would drop a value",
        )
    for consumer_index in consumers:
        match = _DEF_LINE.match(lines[consumer_index].rstrip("\n"))
        if match.group("op") not in ELEMENTWISE_CONSUMERS:
            return LayoutEntry(
                index, name, "REFUSED",
                f"consumer '{match.group('name')}' is {match.group('op')}, "
                "not mul/logical_and (matmul transpose flags are compiler-"
                "side; this rewrite only emits the mul operand layout)",
            )
    source_name = args.get("x")
    source = table.get(source_name) if source_name else None
    if source is None or source[1] != "tile":
        return LayoutEntry(
            index, name, "REFUSED",
            "transpose input is not a tile; a complementary tile cannot "
            "be emitted without materializing the permutation",
        )
    tile_shape = _parse_shape(source[3])
    if tile_shape != shape:
        return LayoutEntry(
            index, name, "REFUSED",
            f"tile shape {source[3]} != transpose shape {shape_text}",
        )
    expand_name = source[4].get("x")
    expand = table.get(expand_name) if expand_name else None
    if expand is None or expand[1] != "expand_dims":
        return LayoutEntry(
            index, name, "REFUSED",
            "tile input is not expand_dims of a vector mask",
        )
    expanded_shape = _parse_shape(expand[3])
    complement = (
        _complement(expanded_shape, tile_shape)
        if expanded_shape is not None
        else None
    )
    if complement is None:
        return LayoutEntry(
            index, name, "REFUSED",
            f"expand shape {expand[3]} is not [B,1,N] or [B,N,1] over a "
            f"square tile {source[3]}",
        )
    mask_name = expand[4].get("x")
    mask = table.get(mask_name) if mask_name else None
    if mask is None:
        return LayoutEntry(
            index, name, "REFUSED",
            "expand_dims input has no defining line",
        )
    mask_shape = _parse_shape(mask[3])
    axes, expand_shape, reps = complement
    if mask_shape != (expand_shape[0], expand_shape[1] * expand_shape[2]):
        # [B, N] with the unit axis removed from expand_shape.
        unitless = tuple(dim for dim in expand_shape if dim != 1)
        if mask_shape != unitless and mask_shape != expand_shape[:2]:
            return LayoutEntry(
                index, name, "REFUSED",
                f"mask shape {mask[3]} does not match expand {expand[3]}",
            )
    _ = axes, reps
    return LayoutEntry(
        index, name, "REWRITTEN",
        "complementary expand_dims+tile emits the tail-swap layout; "
        "mul/logical_and result is unchanged",
    )


def plan_mask_layout(mil_text: str) -> list[LayoutEntry]:
    """Classify every rank-3 square tail-swap transpose."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    returns = _returns_of(lines)
    entries = []
    for name, (index, op, dtype, shape_text, args, _indent) in table.items():
        if op != "transpose" or dtype not in ("bool", "fp16"):
            continue
        if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
            continue
        shape = _parse_shape(shape_text)
        if shape is None or len(shape) != 3 or shape[1] != shape[2]:
            continue
        perm_name = args.get("perm")
        perm = _const_ints(table, lines, perm_name) if perm_name else None
        if perm != list(TAIL_SWAP_RANK3):
            continue
        entries.append(_classify(lines, table, returns, index, name))
    return entries


def rewrite_mask_layout(mil_text: str) -> tuple[str, LayoutReport]:
    """Replace each sound tail-swap with a complementary tile.

    Leftover transposes stay in the graph so the compiler names them.
    """
    lines = mil_text.splitlines()
    report = LayoutReport()
    table = _index_ops(lines)
    returns = _returns_of(lines)
    plan = plan_mask_layout(mil_text)
    insertions: dict[int, list[str]] = {}
    replacements: dict[int, str] = {}
    taken = set(table)
    for entry in plan:
        if entry.disposition != "REWRITTEN":
            report.refused.append(entry)
            continue
        name = entry.output_name
        index, _op, dtype, shape_text, args, indent = table[name]
        tile_name = args["x"]
        tile_entry = table[tile_name]
        expand_name = tile_entry[4]["x"]
        expand_entry = table[expand_name]
        mask_name = expand_entry[4]["x"]
        expanded_shape = _parse_shape(expand_entry[3])
        tiled_shape = _parse_shape(tile_entry[3])
        axes, expand_shape, reps = _complement(expanded_shape, tiled_shape)
        axes_name = _fresh(taken, f"{name}_layout_axes")
        exp_name = _fresh(taken, f"{name}_layout_exp")
        reps_name = _fresh(taken, f"{name}_layout_reps")
        insertions[index] = [
            (
                f"{indent}tensor<int32, [1]> {axes_name} = const()"
                f"[name = tensor<string, []>(\"{axes_name}\"), "
                f"val = tensor<int32, [1]>({axes[0]})];"
            ),
            (
                f"{indent}tensor<{dtype}, {_shape_text(expand_shape)}> "
                f"{exp_name} = expand_dims(axes = {axes_name}, "
                f"x = {mask_name})"
                f"[name = tensor<string, []>(\"{exp_name}\")];"
            ),
            (
                f"{indent}tensor<int32, [3]> {reps_name} = const()"
                f"[name = tensor<string, []>(\"{reps_name}\"), "
                f"val = tensor<int32, [3]>([{reps[0]}, {reps[1]}, "
                f"{reps[2]}])];"
            ),
        ]
        replacements[index] = (
            f"{indent}tensor<{dtype}, {shape_text}> {name} = "
            f"tile(reps = {reps_name}, x = {exp_name})"
            f"[name = tensor<string, []>(\"{name}_layout_tile\")];"
        )
        report.rewritten.append(name)
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
