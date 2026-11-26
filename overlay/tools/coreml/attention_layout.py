# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Birth encoder K as ``[B, H, D, T]`` so scores are the one-program GEMM.

H13's batched envelope covers ``[1, 8, 375, 128] × [1, 8, 128, 375]``
with ``transpose_x=false``, ``transpose_y=false`` as one program
(``apple-parity-batched-matmul``, 208 TDs). The encoder leftover is
the transposed form ``Q[B,H,T,D] @ K[B,H,T,D]^T``.

A trailing-two-dim transpose of K folds back into ``transpose_y=true``,
which is the leftover. A reshape of ``[B,H,T,D]`` to ``[B,H,D,T]`` is a
different packing. The exact rewrite stacks each token's contiguous
``[B, D]`` row as a ``[B, D, 1]`` plane and concats on the last axis,
then expands the unit head axis and concats heads. That is the
definition of perm ``[0, 1, 3, 2]``, not an approximation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class AttentionLayoutError(RuntimeError):
    """An attention-layout rewrite is unsound here; the reason is named."""


STANDALONE_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 128]> x, tensor<fp16, [1, 8, 128, 375]> w) {
    bool tx = const()[name = string("tx"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(false)];
    tensor<fp16, [1, 8, 375, 375]> product = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("product")];
  } -> (product);
}
"""

_MAX_CONCAT = 8

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
    r"val = (?:tensor<bool, \[[^\]]*\]>\()?(?:bool\()?(?P<body>true|false)"
)


@dataclass(frozen=True)
class LayoutEntry:
    """Classification of one QK^T score matmul."""

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
            "schema": "mlx-omarchy.attention-layout-rewrite.v1",
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


def numpy_k_token_stack(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``transpose(K, (0,1,3,2))`` vs concat of per-token last-axis rows.

    ``keys`` is ``[B, H, T, D]``. Each ``keys[:, h, t, :]`` is one
    contiguous ``D``-vector. Stacking those as ``[B, 1, D, 1]`` and
    concating on axis 3 then 1 is the last-two-dim transpose.
    """
    packed = _as_keys(keys)
    original = np.transpose(packed, (0, 1, 3, 2))
    heads = []
    for head in range(packed.shape[1]):
        tokens = [
            packed[:, head : head + 1, time : time + 1, :].reshape(
                packed.shape[0], 1, packed.shape[3], 1
            )
            for time in range(packed.shape[2])
        ]
        heads.append(np.concatenate(tokens, axis=3))
    return original, np.concatenate(heads, axis=1)


def numpy_k_naive_reshape(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``transpose(K, (0,1,3,2))`` vs reshape ``[B, H, D, T]``.

    Same element count, different packing. Callers compare the pair; a
    mismatch is a named counterexample, not a rounding tolerance.
    """
    packed = _as_keys(keys)
    original = np.transpose(packed, (0, 1, 3, 2))
    rewritten = packed.reshape(
        packed.shape[0], packed.shape[1], packed.shape[3], packed.shape[2]
    )
    return original, rewritten


def numpy_qk_contraction(
    query: np.ndarray, keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``Q @ K^T`` vs ``Q @ token_stack(K)``. Same fp16 values."""
    query = np.asarray(query)
    transposed, stacked = numpy_k_token_stack(keys)
    original = _fp16_matmul(query, transposed)
    rewritten = _fp16_matmul(query, stacked)
    return original, rewritten


def require_exact(original: np.ndarray, rewritten: np.ndarray, what: str) -> None:
    """Raise ``AttentionLayoutError`` naming the first mismatched lane."""
    if original.shape != rewritten.shape or original.dtype != rewritten.dtype:
        raise AttentionLayoutError(
            f"{what}: shape/dtype {original.shape}/{original.dtype} vs "
            f"{rewritten.shape}/{rewritten.dtype}"
        )
    left = original.view("<u2")
    right = rewritten.view("<u2")
    if np.array_equal(left, right):
        return
    mismatch = np.argwhere(left != right)
    index = tuple(int(v) for v in mismatch[0])
    raise AttentionLayoutError(
        f"inexact {what} at {list(index)}: original="
        f"{original[index]!r} rewritten={rewritten[index]!r}"
    )


def _as_keys(keys) -> np.ndarray:
    packed = np.asarray(keys)
    if packed.ndim != 4:
        raise AttentionLayoutError(
            f"keys expect rank-4 [B, H, T, D], got {packed.shape}"
        )
    return packed


def _fp16_matmul(left, right):
    acc = np.matmul(left.astype(np.float32), right.astype(np.float32))
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


def _parse_bool(line: str) -> bool | None:
    match = _BOOL_VAL.search(line)
    if match is None:
        return None
    return match.group("body") == "true"


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


def _const_bool(table, lines, name: str) -> bool | None:
    entry = table.get(name)
    if entry is None or entry[1] != "const":
        return None
    return _parse_bool(lines[entry[0]])


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


def _int_vec(indent: str, name: str, values: list[int]) -> str:
    rank = len(values)
    return (
        f"{indent}tensor<int32, [{rank}]> {name} = const()"
        f"[name = tensor<string, []>(\"{name}\"), "
        f"val = tensor<int32, [{rank}]>({_shape_text(tuple(values))})];"
    )


def _concat_line(indent, dtype, shape, name, axis_name, pieces) -> str:
    args = ", ".join(
        [f"axis = {axis_name}"]
        + [f"x{offset} = {piece}" for offset, piece in enumerate(pieces)]
    )
    return (
        f"{indent}tensor<{dtype}, {_shape_text(tuple(shape))}> {name} = "
        f"concat({args})"
        f"[name = tensor<string, []>(\"{name}_attn_concat\")];"
    )


def _concat_tree(
    indent, taken, prefix, dtype, axis, axis_name, pieces, piece_shape, out_name
):
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
    if not inserted:
        return names[0], inserted
    return names[0], inserted


def _head_linears(table, lines, y_name):
    """Return ``(axis_name, [(exp_name, lin_name), ...])`` or None."""
    entry = table.get(y_name)
    if entry is None or entry[1] != "concat":
        return None
    args = entry[4]
    axis_name = args.get("axis")
    axis = _const_ints(table, lines, axis_name) if axis_name else None
    if axis != [1] and axis != 1:
        return None
    heads = []
    offset = 0
    while f"x{offset}" in args:
        exp_name = args[f"x{offset}"]
        exp = table.get(exp_name)
        if exp is None or exp[1] != "expand_dims":
            return None
        lin_name = exp[4].get("x")
        lin = table.get(lin_name) if lin_name else None
        if lin is None or lin[1] != "linear":
            return None
        heads.append((exp_name, lin_name, exp[4].get("axes")))
        offset += 1
    if len(heads) < 2:
        return None
    return axis_name, heads


def _classify(lines, table, index, name) -> LayoutEntry:
    _index, op, dtype, shape_text, args, _indent = table[name]
    _ = _index
    if op != "matmul" or dtype != "fp16":
        raise AttentionLayoutError(f"not an attention-layout candidate: {name}")
    out_shape = _parse_shape(shape_text)
    x_name = args.get("x")
    y_name = args.get("y")
    x = table.get(x_name) if x_name else None
    y = table.get(y_name) if y_name else None
    if x is None or y is None:
        return LayoutEntry(
            index, name, "REFUSED", "matmul is missing x or y"
        )
    if y[1] == "const":
        return LayoutEntry(
            index, name, "REFUSED",
            "constant K; host-side transpose is a compiler packing path, "
            "not this rewrite",
        )
    x_shape = _parse_shape(x[3])
    y_shape = _parse_shape(y[3])
    if (
        x_shape is None
        or y_shape is None
        or out_shape is None
        or len(x_shape) != 4
        or len(y_shape) != 4
        or len(out_shape) != 4
    ):
        return LayoutEntry(
            index, name, "REFUSED",
            f"score matmul wants rank-4 Q/K/out, got {x[3]} {y[3]} {shape_text}",
        )
    batch, heads, time, dim = x_shape
    if y_shape != (batch, heads, time, dim):
        return LayoutEntry(
            index, name, "REFUSED",
            f"K shape {y[3]} is not Q's [B,H,T,D] {x[3]}; "
            "this rewrite is Q @ K^T, not PV/rel-pos",
        )
    if out_shape != (batch, heads, time, time):
        return LayoutEntry(
            index, name, "REFUSED",
            f"output {shape_text} is not [B,H,T,T] for Q {x[3]}",
        )
    if heads <= 1:
        return LayoutEntry(
            index, name, "REFUSED",
            "H<=1 is a unit-axis move, not this rewrite",
        )
    if any(dim_i <= 0 for dim_i in x_shape):
        return LayoutEntry(
            index, name, "REFUSED",
            f"Q shape {x[3]} is not a static [B,H,T,D]",
        )
    tx = _const_bool(table, lines, args["transpose_x"]) if "transpose_x" in args else None
    ty = _const_bool(table, lines, args["transpose_y"]) if "transpose_y" in args else None
    if tx is not False:
        return LayoutEntry(
            index, name, "REFUSED",
            f"transpose_x is {tx}; only explicit false is this rewrite",
        )
    if ty is not True:
        return LayoutEntry(
            index, name, "REFUSED",
            f"transpose_y is {ty}; leftover scores are the ty=true form",
        )
    return LayoutEntry(
        index, name, "REWRITTEN",
        "K born [B,H,D,T] via token-row concat; matmul ty=false is "
        "[B,H,T,D] x [B,H,D,T]",
    )


def plan_attention_layout(mil_text: str) -> list[LayoutEntry]:
    """Classify every fp16 QK^T score matmul."""
    lines = mil_text.splitlines()
    table = _index_ops(lines)
    entries = []
    for name, (index, op, dtype, shape_text, args, _indent) in table.items():
        if op != "matmul" or dtype != "fp16":
            continue
        if _DEF_LINE.match(lines[index].rstrip("\n")) is None:
            continue
        if "transpose_y" not in args:
            continue
        ty = _const_bool(table, lines, args["transpose_y"])
        if ty is not True:
            continue
        entries.append(_classify(lines, table, index, name))
    return entries


def _emit_from_linear(
    indent, taken, prefix, lin_name, batch, time, dim, out_name, axes_name
):
    """``[B, T, D]`` linear → ``[B, 1, D, T]`` token-row concat + expand."""
    inserted = []
    if axes_name is None:
        axes_name = _fresh(taken, f"{prefix}_axes")
        inserted.append(
            f"{indent}tensor<int32, [1]> {axes_name} = const()"
            f"[name = tensor<string, []>(\"{axes_name}\"), "
            f"val = tensor<int32, [1]>(1)];"
        )
    shape_name = _fresh(taken, f"{prefix}_row_shape")
    inserted.append(_int_vec(indent, shape_name, [batch, dim, 1]))
    axis_name = _fresh(taken, f"{prefix}_token_axis")
    inserted.append(
        f"{indent}tensor<int32, []> {axis_name} = const()"
        f"[name = tensor<string, []>(\"{axis_name}\"), "
        f"val = tensor<int32, []>(2)];"
    )
    pieces = []
    for token in range(time):
        b_name = _fresh(taken, f"{prefix}_t{token}_begin")
        e_name = _fresh(taken, f"{prefix}_t{token}_end")
        s_name = _fresh(taken, f"{prefix}_t{token}_sl")
        r_name = _fresh(taken, f"{prefix}_t{token}_row")
        inserted.append(_int_vec(indent, b_name, [0, token, 0]))
        inserted.append(_int_vec(indent, e_name, [batch, token + 1, dim]))
        inserted.append(
            f"{indent}tensor<fp16, {_shape_text((batch, 1, dim))}> {s_name} = "
            f"slice_by_index(begin = {b_name}, end = {e_name}, x = {lin_name})"
            f"[name = tensor<string, []>(\"{s_name}\")];"
        )
        inserted.append(
            f"{indent}tensor<fp16, {_shape_text((batch, dim, 1))}> {r_name} = "
            f"reshape(shape = {shape_name}, x = {s_name})"
            f"[name = tensor<string, []>(\"{r_name}\")];"
        )
        pieces.append(r_name)
    stacked_name = _fresh(taken, f"{prefix}_dt") if time > 1 else pieces[0]
    if time > 1:
        tree_name, tree_ins = _concat_tree(
            indent, taken, prefix, "fp16", 2, axis_name,
            pieces, [batch, dim, 1], stacked_name,
        )
        inserted.extend(tree_ins)
        stacked_name = tree_name
    inserted.append(
        f"{indent}tensor<fp16, {_shape_text((batch, 1, dim, time))}> {out_name} = "
        f"expand_dims(axes = {axes_name}, x = {stacked_name})"
        f"[name = tensor<string, []>(\"{out_name}_attn_exp\")];"
    )
    return inserted


def _emit_from_keys(
    indent, taken, prefix, keys_name, batch, heads, time, dim, out_name
):
    """``[B, H, T, D]`` → ``[B, H, D, T]`` via per-head token-row concat."""
    inserted = []
    axis_name = _fresh(taken, f"{prefix}_heads_axis")
    inserted.append(
        f"{indent}tensor<int32, []> {axis_name} = const()"
        f"[name = tensor<string, []>(\"{axis_name}\"), "
        f"val = tensor<int32, []>(1)];"
    )
    axes_name = _fresh(taken, f"{prefix}_axes")
    inserted.append(
        f"{indent}tensor<int32, [1]> {axes_name} = const()"
        f"[name = tensor<string, []>(\"{axes_name}\"), "
        f"val = tensor<int32, [1]>(1)];"
    )
    head_names = []
    for head in range(heads):
        b_name = _fresh(taken, f"{prefix}_h{head}_begin")
        e_name = _fresh(taken, f"{prefix}_h{head}_end")
        p_name = _fresh(taken, f"{prefix}_h{head}_plane")
        sq_shape = _fresh(taken, f"{prefix}_h{head}_sq_shape")
        sq_name = _fresh(taken, f"{prefix}_h{head}_lin")
        inserted.append(_int_vec(indent, b_name, [0, head, 0, 0]))
        inserted.append(_int_vec(indent, e_name, [batch, head + 1, time, dim]))
        inserted.append(
            f"{indent}tensor<fp16, "
            f"{_shape_text((batch, 1, time, dim))}> {p_name} = "
            f"slice_by_index(begin = {b_name}, end = {e_name}, x = {keys_name})"
            f"[name = tensor<string, []>(\"{p_name}\")];"
        )
        inserted.append(_int_vec(indent, sq_shape, [batch, time, dim]))
        inserted.append(
            f"{indent}tensor<fp16, {_shape_text((batch, time, dim))}> {sq_name} = "
            f"reshape(shape = {sq_shape}, x = {p_name})"
            f"[name = tensor<string, []>(\"{sq_name}\")];"
        )
        exp_name = _fresh(taken, f"{prefix}_h{head}_exp")
        inserted.extend(
            _emit_from_linear(
                indent, taken, f"{prefix}_h{head}", sq_name,
                batch, time, dim, exp_name, axes_name,
            )
        )
        head_names.append(exp_name)
    tree_name, tree_ins = _concat_tree(
        indent, taken, prefix, "fp16", 1, axis_name,
        head_names, [batch, 1, dim, time], out_name,
    )
    _ = tree_name
    inserted.extend(tree_ins)
    return inserted


def _emit_rewrite(lines, table, taken, name) -> tuple[dict[int, list[str]], dict[int, str]]:
    index, _op, _dtype, _out_shape, args, indent = table[name]
    x_name = args["x"]
    y_name = args["y"]
    x_shape = _parse_shape(table[x_name][3])
    batch, heads, time, dim = x_shape
    ty_name = _fresh(taken, f"{name}_ty0")
    ty_line = (
        f"{indent}tensor<bool, []> {ty_name} = const()"
        f"[name = tensor<string, []>(\"{ty_name}\"), "
        f"val = tensor<bool, []>(false)];"
    )
    insertions: dict[int, list[str]] = {}
    replacements: dict[int, str] = {}
    y_consumers = _consumers_of(lines, y_name)
    in_place = y_name != x_name and y_consumers == [index]
    born = _head_linears(table, lines, y_name) if in_place else None
    if born is not None:
        y_index = table[y_name][0]
        _axis_name, head_pairs = born
        if any(
            _consumers_of(lines, exp_name) != [y_index]
            for exp_name, _lin_name, _axes in head_pairs
        ):
            born = None
    if born is not None:
        _axis_name, head_pairs = born
        for exp_name, lin_name, axes_name in head_pairs:
            exp_index = table[exp_name][0]
            emitted = _emit_from_linear(
                indent, taken, exp_name, lin_name,
                batch, time, dim, exp_name, axes_name,
            )
            insertions[exp_index] = emitted[:-1]
            replacements[exp_index] = emitted[-1]
        y_index = table[y_name][0]
        y_line = lines[y_index]
        old = f"tensor<fp16, {_shape_text((batch, heads, time, dim))}> {y_name}"
        new = f"tensor<fp16, {_shape_text((batch, heads, dim, time))}> {y_name}"
        if old not in y_line:
            raise AttentionLayoutError(
                f"concat line for {y_name} does not start with {old}"
            )
        replacements[y_index] = y_line.replace(old, new, 1)
        new_y = y_name
    else:
        new_y = _fresh(taken, f"{name}_k_dt")
        emitted = _emit_from_keys(
            indent, taken, new_y, y_name,
            batch, heads, time, dim, new_y,
        )
        insertions[index] = emitted
    mm_line = lines[index]
    mm_line = re.sub(
        r"transpose_y\s*=\s*" + re.escape(args["transpose_y"]),
        f"transpose_y = {ty_name}",
        mm_line,
        count=1,
    )
    if new_y != y_name:
        mm_line = re.sub(
            r"y\s*=\s*" + re.escape(y_name) + r"(?![\w])",
            f"y = {new_y}",
            mm_line,
            count=1,
        )
    replacements[index] = mm_line
    insertions.setdefault(index, [])
    insertions[index] = [ty_line] + insertions[index]
    return insertions, replacements


def rewrite_attention_layout(mil_text: str) -> tuple[str, LayoutReport]:
    """Replace each sound QK^T with ``Q @ K[B,H,D,T]`` and ``ty=false``.

    Leftover matmuls stay in the graph so the compiler names them.
    """
    lines = mil_text.splitlines()
    report = LayoutReport()
    table = _index_ops(lines)
    plan = plan_attention_layout(mil_text)
    insertions: dict[int, list[str]] = {}
    replacements: dict[int, str] = {}
    taken = set(table)
    for entry in plan:
        if entry.disposition != "REWRITTEN":
            report.refused.append(entry)
            continue
        ins, repl = _emit_rewrite(lines, table, taken, entry.output_name)
        for key, value in ins.items():
            insertions.setdefault(key, []).extend(value)
        replacements.update(repl)
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
