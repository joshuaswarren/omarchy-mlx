# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Fold non-fp16/non-bool shape aliases in the frontend.

The H13 shape-alias (free view) path requires fp16
(`invalid-shape-alias` otherwise), and bool views belong to the
compiler lane's bool-surface work. So any remaining view op —
``expand_dims``, ``reshape``, ``squeeze`` — whose output dtype is
neither fp16 nor bool must be eliminated in the frontend, or H13 sees
an int32 (or other) view it cannot lower.

Two sound rewrites, both value-preserving by construction:

P1 — const-input relabel. A view whose value input is a const is
replaced by a const with identical bytes and the view's output shape.
Reshape-family ops never reorder elements, so the flat payload is
unchanged; only shape metadata moves.

P2 — unit-expand absorption. An ``expand_dims`` (which by definition
only inserts unit axes) feeding elementwise/broadcast consumers is
deleted: const siblings are duplicated with leading unit axes
prepended (same bytes, same flat order), runtime operands keep their
shapes, and each consumer is re-verified to produce its originally
declared output shape under numpy broadcasting. If any consumer's
output shape would change, or a consumer is not
elementwise-positional, the fold is refused with a named reason.

Leftover views are reported, not raised: the compiler names the next
reject, which is the fail-fast loop working as designed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class UnitFoldError(RuntimeError):
    """A shape-alias fold is unsound here; the reason is named."""


VIEW_OPS = ("expand_dims", "reshape", "squeeze")

# Ops whose output element at each position depends only on the
# broadcast-aligned input elements at that position. Moving unit axes
# between their operands preserves every value as long as the output
# shape is unchanged (checked per consumer before rewriting).
ELEMENTWISE_CONSUMERS = frozenset({
    "add",
    "sub",
    "mul",
    "div",
    "real_div",
    "maximum",
    "minimum",
    "less",
    "greater",
    "equal",
    "not_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "select",
})

_SKIP_DTYPES = ("fp16", "bool")

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


@dataclass
class FoldEntry:
    """Classification of one non-fp16/non-bool view op."""

    index: int
    output_name: str
    disposition: str  # FOLDED | REFUSED
    reason: str


@dataclass
class FoldReport:
    folded: list[str] = field(default_factory=list)
    refused: list[FoldEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": "mlx-omarchy.unit-view-fold.v1",
            "folded": list(self.folded),
            "refused": [
                {
                    "index": entry.index,
                    "output": entry.output_name,
                    "reason": entry.reason,
                }
                for entry in self.refused
            ],
        }


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


def _element_count(shape: tuple[int, ...]) -> int | None:
    total = 1
    for dim in shape:
        if dim < 0:
            return None
        total *= dim
    return total


def _broadcast(shapes: list[tuple[int, ...]]) -> tuple[int, ...] | None:
    """Numpy-style broadcast of static shapes; None if incompatible."""
    rank = max((len(shape) for shape in shapes), default=0)
    result: list[int] = []
    for axis in range(1, rank + 1):
        dims = {
            shape[-axis] if axis <= len(shape) else 1 for shape in shapes
        }
        dims.discard(1)
        if len(dims) > 1:
            return None
        result.append(dims.pop() if dims else 1)
    return tuple(reversed(result))


def _code_of(line: str) -> str:
    """The executable part of a line (drops the [name = ...] metadata)."""
    cut = line.find("[name =")
    return line if cut < 0 else line[:cut]


def _index_ops(lines: list[str]):
    """Map output name -> (line index, op, dtype, shape, args)."""
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
        )
    # Function parameters have no defining line; index them too.
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("func ") and "(" in stripped:
            params = stripped.split("(", 1)[1].rsplit(")", 1)[0]
            # finditer, not comma-split: shape tuples contain commas.
            for match in _PARAM.finditer(params):
                table.setdefault(
                    match.group("name"),
                    (index, "param", match.group("dtype"),
                     match.group("shape"), {}),
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


def plan_unit_folds(mil_text: str) -> list[FoldEntry]:
    """Classify every non-fp16/non-bool view op without rewriting."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    returns: set[str] = set()
    for line in lines:
        footer = _FOOTER.match(line.rstrip("\n"))
        if footer:
            returns = {
                name.strip()
                for name in footer.group("returns").split(",")
            }
    entries = []
    for name, (index, op, dtype, _shape, _args) in table.items():
        if not isinstance(index, int):
            continue
        line = lines[index]
        if _DEF_LINE.match(line.rstrip("\n")) is None:
            continue  # function parameter, not an op
        if op not in VIEW_OPS or dtype in _SKIP_DTYPES:
            continue
        entries.append(_classify(lines, table, returns, index, name))
    return entries


def _target_shape(table, name: str) -> tuple[int, ...] | None:
    entry = table.get(name)
    if entry is None:
        return None
    shape = _parse_shape(entry[3])
    if shape is None or any(dim < 0 for dim in shape):
        return None
    return shape


def _classify(lines, table, returns, index, name) -> FoldEntry:
    _, op, dtype, shape_text, args = table[name]
    shape = _parse_shape(shape_text)
    if shape is None or any(dim < 0 for dim in shape):
        return FoldEntry(
            index, name, "REFUSED",
            f"view output shape {shape_text} is not static",
        )
    if name in returns:
        return FoldEntry(
            index, name, "REFUSED",
            "view output is a graph return; there is no consumer to "
            "absorb it into (boundary views belong to output_peel)",
        )
    value_name = args.get("x")
    if value_name is None:
        return FoldEntry(
            index, name, "REFUSED", f"{op} has no readable x input"
        )
    value = table.get(value_name)
    if value is not None and value[1] == "const":
        return FoldEntry(
            index, name, "FOLDED",
            f"const-input relabel to {_shape_text(shape)}",
        )
    if op in ("reshape", "squeeze"):
        return FoldEntry(
            index, name, "REFUSED",
            f"{op} input is runtime data; only const-fed relabels and "
            "expand_dims absorption are implemented",
        )
    # expand_dims with a runtime input: unit-expand absorption.
    consumers = _consumers_of(lines, name)
    if not consumers:
        return FoldEntry(index, name, "FOLDED", "dead view; deleted")
    problems = []
    for consumer_index in consumers:
        problem = _check_consumer(lines, table, name, consumer_index)
        if problem is not None:
            problems.append(problem)
    if problems:
        return FoldEntry(index, name, "REFUSED", "; ".join(problems))
    return FoldEntry(
        index, name, "FOLDED",
        f"unit-expand absorption into {len(consumers)} consumer(s)",
    )


def _check_consumer(lines, table, view_name, consumer_index) -> str | None:
    """Verify one consumer survives absorption with an identical output."""
    line = lines[consumer_index]
    match = _DEF_LINE.match(line.rstrip("\n"))
    consumer_op = match.group("op")
    if consumer_op not in ELEMENTWISE_CONSUMERS:
        return (
            f"consumer '{match.group('name')}' is {consumer_op}, not "
            "elementwise-positional"
        )
    declared = _parse_shape(match.group("shape"))
    if declared is None or any(dim < 0 for dim in declared):
        return (
            f"consumer '{match.group('name')}' output shape is dynamic"
        )
    view_index, _, _, _, view_args = table[view_name]
    _ = view_index
    input_name = view_args.get("x")
    input_shape = _target_shape(table, input_name) if input_name else None
    if input_shape is None:
        return (
            f"view '{view_name}' input shape is unknown or dynamic"
        )
    args = dict(_ARG.findall(_code_of(line).split("(", 1)[1]))
    new_shapes: list[tuple[int, ...]] = []
    for key, bound in sorted(args.items()):
        if bound == view_name:
            new_shapes.append(input_shape)
            continue
        sibling = _target_shape(table, bound)
        if sibling is None:
            return (
                f"consumer '{match.group('name')}' input '{bound}' "
                "has unknown shape"
            )
        entry = table.get(bound)
        if entry is not None and entry[1] == "const":
            want = max(len(declared), len(sibling))
            sibling = (1,) * (want - len(sibling)) + sibling
        new_shapes.append(sibling)
    if _broadcast(new_shapes) != declared:
        return (
            f"consumer '{match.group('name')}' output would change "
            f"{_shape_text(declared)} -> "
            f"{_broadcast(new_shapes)}; refusing"
        )
    return None


def fold_unit_views(mil_text: str) -> tuple[str, FoldReport]:
    """Apply every sound fold; report leftovers without raising."""
    lines = mil_text.splitlines()
    report = FoldReport()
    for _ in range(10):
        table = _index_ops(lines)
        returns: set[str] = set()
        for line in lines:
            footer = _FOOTER.match(line.rstrip("\n"))
            if footer:
                returns = {
                    name.strip()
                    for name in footer.group("returns").split(",")
                }
        progressed = False
        for name in list(table):
            entry = table.get(name)
            if entry is None:
                continue
            index, op, dtype, _shape, _args = entry
            if not isinstance(index, int):
                continue
            if index >= len(lines):
                continue
            if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
                continue
            if op not in VIEW_OPS or dtype in _SKIP_DTYPES:
                continue
            decision = _classify(lines, table, returns, index, name)
            if decision.disposition != "FOLDED":
                if not any(
                    e.output_name == name for e in report.refused
                ):
                    report.refused.append(decision)
                continue
            lines = _apply_fold(lines, table, returns, index, name)
            table = _index_ops(lines)
            report.folded.append(name)
            progressed = True
        if not progressed:
            break
    eol = "\n" if mil_text.endswith("\n") else ""
    return "\n".join(lines) + eol, report


def _apply_fold(lines, table, returns, index, name) -> list[str]:
    _, op, dtype, shape_text, args = table[name]
    shape = _parse_shape(shape_text)
    value_name = args.get("x")
    value = table.get(value_name, None) if value_name else None
    if value is not None and value[1] == "const":
        return _apply_relabel(lines, table, index, name, value_name)
    if value_name is None:
        raise UnitFoldError(f"view '{name}' has no x input to rewire to")
    # Unit-expand absorption. All edits are computed against the
    # original indices first, then applied in one rebuild, so helper
    # insertions can never shift a pending edit.
    consumers = _consumers_of(lines, name)
    insertions: dict[int, list[str]] = {}
    dup_map: dict[str, str] = {}
    rewritten: dict[int, str] = {}
    taken_names = set(table)
    for consumer_index in consumers:
        line = lines[consumer_index]
        match = _DEF_LINE.match(line.rstrip("\n"))
        declared = _parse_shape(match.group("shape"))
        cargs = dict(_ARG.findall(_code_of(line).split("(", 1)[1]))
        for key in sorted(cargs):
            bound = cargs[key]
            if bound == name:
                continue
            entry = table.get(bound)
            if entry is None or entry[1] != "const":
                continue
            sibling_shape = _target_shape(table, bound)
            if sibling_shape is None:
                raise UnitFoldError(
                    f"const sibling '{bound}' has an unreadable shape"
                )
            want = max(len(declared), len(sibling_shape))
            if len(sibling_shape) >= want or bound in dup_map:
                continue
            source_index = entry[0]
            if not isinstance(source_index, int):
                raise UnitFoldError(
                    f"const sibling '{bound}' is a function parameter; "
                    "cannot relabel an input"
                )
            if source_index > consumer_index:
                raise UnitFoldError(
                    f"const sibling '{bound}' is defined after its "
                    "consumer; refusing out-of-order rewrite"
                )
            fresh, text = _dup_const_text(
                lines, table, taken_names, bound,
                (1,) * (want - len(sibling_shape)) + sibling_shape,
            )
            taken_names.add(fresh)
            insertions.setdefault(source_index + 1, []).append(text)
            dup_map[bound] = fresh
        code, sep, meta = line.partition("[name =")
        code = _rename(code, name, value_name)
        for original, dup in dup_map.items():
            code = _rename(code, original, dup)
        rewritten[consumer_index] = code + (sep + meta if sep else "")
    out: list[str] = []
    for i, line in enumerate(lines):
        for text in insertions.get(i, []):
            out.append(text)
        if i == index:
            continue  # delete the absorbed view
        out.append(rewritten.get(i, line))
    return out


def _rename(code: str, old: str, new: str) -> str:
    return re.compile(r"(?<![\w])" + re.escape(old) + r"(?![\w])").sub(
        new, code
    )


def _apply_relabel(lines, table, index, name, value_name) -> list[str]:
    line = lines[index]
    match = _DEF_LINE.match(line.rstrip("\n"))
    value_index, _, _, _, _ = table[value_name]
    val_text = _const_val_text(lines[value_index])
    if val_text is None:
        raise UnitFoldError(
            f"const '{value_name}' has no readable value payload"
        )
    indent = match.group("indent")
    dtype = match.group("dtype")
    shape = match.group("shape")
    lines[index] = (
        f"{indent}tensor<{dtype}, {shape}> {name} = "
        f"const()[name = tensor<string, []>(\"{name}_folded_const\"), "
        f"val = {val_text}];"
    )
    return lines


def _dup_const_text(lines, table, taken, bound: str, shape: tuple[int, ...]):
    """A duplicated const line (new shape, identical bytes)."""
    source_index, _, _, _, _ = table[bound]
    val_text = _const_val_text(lines[source_index])
    if val_text is None:
        raise UnitFoldError(
            f"const '{bound}' has no readable value payload"
        )
    candidate, counter = f"{bound}_fold_const", 0
    while candidate in taken:
        counter += 1
        candidate = f"{bound}_fold_const_{counter}"
    indent = _DEF_LINE.match(lines[source_index].rstrip("\n")).group(
        "indent"
    )
    dtype = table[bound][2]
    text = (
        f"{indent}tensor<{dtype}, {_shape_text(shape)}> {candidate} = "
        f"const()[name = tensor<string, []>(\"{candidate}_folded_const\"), "
        f"val = {val_text}];"
    )
    return candidate, text


def _const_val_text(line: str) -> str | None:
    start = line.find("val = ")
    if start < 0:
        return None
    tail = line[start + len("val = "):].rstrip()
    if tail.endswith(";"):
        tail = tail[:-1]
    # The payload is a balanced tensor<...>(...) expression.
    depth = 0
    in_string = False
    for pos, char in enumerate(tail):
        if char == '"' and (pos == 0 or tail[pos - 1] != "\\"):
            in_string = not in_string
        elif not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return tail[: pos + 1]
    return None
