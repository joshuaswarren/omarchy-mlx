#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Numpy evaluator for textual MIL from the adapter or the original encoder.

Host-only. Evaluates the pinned Parakeet encoder from MIL plus blob-v2 weight
files, including original scalar constructors and fp16/uint4 palette constants.
Encoder tensors can be staged, diffed and validated without Core ML, the ANE,
or a GPU. Anything outside the implemented operations is a hard error.

Numeric contract (the Core ML fp16 execution model): every op computes in
fp32 and rounds once to the dtype the MIL declares for its result. Elementwise
ops on two fp16 operands therefore round exactly as numpy fp16 arithmetic
would; conv, linear, matmul, layer_norm and softmax accumulate in fp32.

This is the one canonical copy. The receipt directories
``receipts/2026-09-14-encoder-islands-exec-*`` and
``receipts/2026-09-14-encoder-parity-ane`` keep their original forks as
evidence; see ``receipts/2026-09-14-mil-evaluator-collapse.md``.

Usage::

    python3 mil_numpy.py --source DIR --capture DIR [--out validation.json]

``--source`` is a ``mil_adapter.py emit`` output (``model.mil`` next to
``model-root/``); ``--capture`` holds ``encoder_input_features.npy``,
``encoder_input_mask.npy``, ``encoder_hidden.npy`` and ``encoder_mask.npy``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coreml.depalettize import _gather_dense, _read_blob, read_blob_metadata

BLOB_MAGIC = 0xDEADBEEF
BLOB_STORAGE = {1: np.float16, 2: np.float32, 3: np.uint8, 4: np.int8, 14: np.int32}
DTYPES = {
    "fp16": np.float16,
    "fp32": np.float32,
    "int32": np.int32,
    "int16": np.int16,
    "uint16": np.uint16,
    "uint8": np.uint8,
    "int8": np.int8,
    "bool": np.bool_,
}

STMT = re.compile(
    r"^\s*(?P<type>tensor<[^>]*>|string|int32|bool|fp16|fp32)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_@]*)\s*=\s*"
    r"(?P<op>[a-z_][a-z_0-9]*)\((?P<args>.*)\)\s*(?:\[(?P<attrs>.*)\])?;\s*$"
)
TUPLE_STMT = re.compile(
    r"^\s*\((?P<results>tensor<[^)]*)\)\s*=\s*"
    r"(?P<op>[a-z_][a-z_0-9]*)\((?P<args>.*)\)\s*(?:\[(?P<attrs>.*)\])?;\s*$"
)
TYPE = re.compile(r"^tensor<\s*(?P<dtype>\w+)\s*,\s*\[(?P<shape>[^\]]*)\]\s*>$")
LITERAL = re.compile(r"^tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\((.*)\)$", re.S)
BLOBFILE = re.compile(
    r'BLOBFILE\(path = string\("(?P<path>[^"]+)"\), offset = uint64\((?P<offset>\d+)\)\)'
)
FUNC_INPUT = re.compile(r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\s+([A-Za-z_]\w*)")


class MilError(RuntimeError):
    pass


def split_top(text: str) -> list[str]:
    """Split on commas outside brackets, parentheses, angle brackets and quotes."""
    out, depth, start, quoted = [], 0, 0, False
    for i, ch in enumerate(text):
        if quoted:
            if ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return [item.strip() for item in out]


def parse_kwargs(text: str) -> dict[str, str]:
    result = {}
    for item in split_top(text):
        if item:
            key, _, value = item.partition("=")
            result[key.strip()] = value.strip()
    return result


def parse_type(text: str) -> tuple[str, tuple[int, ...]]:
    match = TYPE.match(text.strip())
    if match is None:
        return text.strip().lower(), ()
    shape_text = match.group("shape").strip()
    shape = tuple(int(i) for i in shape_text.split(",")) if shape_text else ()
    return match.group("dtype").lower(), shape


def _scalar(text: str, dtype):
    text = text.strip()
    if dtype is np.bool_:
        if text not in ("true", "false"):
            raise MilError(f"invalid bool literal {text!r}")
        return text == "true"
    if dtype in (np.float16, np.float32):
        try:
            return float(text)
        except ValueError:
            return float.fromhex(text)
    return int(text, 0)


class Blobs:
    """Reader for the blob-v2 files ``mil_adapter.py`` emits."""

    def __init__(self, model_root: Path):
        self.root = Path(model_root)
        self._maps: dict[str, np.memmap] = {}

    def _map(self, path: str) -> np.memmap:
        name = path.replace("@model_path/", "")
        if name not in self._maps:
            self._maps[name] = np.memmap(self.root / name, dtype=np.uint8, mode="r")
        return self._maps[name]

    def read(self, path: str, offset: int, dtype, shape: tuple[int, ...]) -> np.ndarray:
        raw = self._map(path)
        magic, storage, length, payload = struct.unpack_from(
            "<IIQQ", raw[offset : offset + 24].tobytes(), 0
        )
        if magic != BLOB_MAGIC:
            raise MilError(f"blob magic {magic:#x} at {path}:{offset}")
        stored = BLOB_STORAGE.get(storage)
        if stored is None:
            raise MilError(f"unsupported blob storage code {storage} at {path}:{offset}")
        count = int(np.prod(shape)) if shape else 1
        want = count * np.dtype(stored).itemsize
        if want > length:
            raise MilError(f"blob at {path}:{offset} holds {length} bytes, want {want}")
        data = np.frombuffer(raw[payload : payload + want].tobytes(), dtype=stored, count=count)
        return data.astype(dtype, copy=False).reshape(shape)


class Statement:
    __slots__ = ("index", "names", "op", "kwargs", "attrs", "dtypes", "shapes", "done")

    def __init__(self, index, names, op, kwargs, attrs, dtypes, shapes):
        self.index = index
        self.names = names
        self.op = op
        self.kwargs = kwargs
        self.attrs = attrs
        self.dtypes = dtypes
        self.shapes = shapes
        self.done = False


class Program:
    """One MIL ``main`` function, evaluated on demand or front to back."""

    def __init__(self, mil_path: Path, model_root: Path):
        self.blobs = Blobs(model_root)
        self.values: dict[str, object] = {}
        self.inputs: dict[str, tuple[str, tuple[int, ...]]] = {}
        self.statements: list[Statement] = []
        self.producer: dict[str, Statement] = {}
        self.executed = 0
        self._parse(Path(mil_path).read_text())
        self._index_last_use()

    # ---------------------------------------------------------------- parse

    def _parse(self, text: str) -> None:
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("func main"):
                for dtype, shape, name in FUNC_INPUT.findall(stripped):
                    self.inputs[name] = (dtype, parse_type(f"tensor<{dtype}, [{shape}]>")[1])
                continue
            tup = TUPLE_STMT.match(line)
            if tup is not None:
                names, dtypes, shapes = [], [], []
                for item in split_top(tup.group("results")):
                    decl, name = item.rsplit(" ", 1)
                    dtype, shape = parse_type(decl)
                    names.append(name)
                    dtypes.append(dtype)
                    shapes.append(shape)
                op, args, attrs = tup.group("op"), tup.group("args"), tup.group("attrs")
            else:
                match = STMT.match(line)
                if match is None:
                    continue
                dtype, shape = parse_type(match.group("type"))
                names, dtypes, shapes = [match.group("name")], [dtype], [shape]
                op, args, attrs = match.group("op"), match.group("args"), match.group("attrs")
            stmt = Statement(
                len(self.statements), names, op, parse_kwargs(args), attrs or "", dtypes, shapes
            )
            self.statements.append(stmt)
            for name in names:
                self.producer[name] = stmt

    def _operand_names(self, token: str) -> list[str]:
        token = token.strip()
        if token.startswith("("):
            return [t for t in split_top(token.strip("() ")) if t in self.producer]
        return [token] if token in self.producer else []

    def _index_last_use(self) -> None:
        """Names to release after each statement, so ``run`` frees tensors past
        their final reader. The encoder's constants alone are 1.2 GB of fp16."""
        last_use: dict[str, int] = {}
        for stmt in self.statements:
            for token in stmt.kwargs.values():
                for name in self._operand_names(token):
                    last_use[name] = stmt.index
        self.release_at: dict[int, list[str]] = {}
        for name, index in last_use.items():
            self.release_at.setdefault(index, []).append(name)

    # ------------------------------------------------------------- operands

    def ensure(self, name: str) -> None:
        if name in self.values:
            return
        stmt = self.producer.get(name)
        if stmt is None:
            raise MilError(f"unresolved operand {name!r}")
        if stmt.done:
            raise MilError(f"{name!r} was produced and already released")
        for token in stmt.kwargs.values():
            for operand in self._operand_names(token):
                self.ensure(operand)
        self.execute(stmt)

    def get(self, name: str):
        self.ensure(name)
        return self.values[name]

    def arg(self, stmt: Statement, key: str, default=None):
        """Operand ``key``: a named value, a parenthesised list, an inline
        literal, or ``default`` when the MIL omits it."""
        token = stmt.kwargs.get(key)
        if token is None:
            return default
        if token.startswith("("):
            return [self.get(t) for t in split_top(token.strip("() "))]
        if token.startswith("tensor<"):
            return self._literal(token)
        return self.get(token)

    def scalar(self, stmt: Statement, key: str, default=None):
        value = self.arg(stmt, key, default)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return np.asarray(value).reshape(-1)[0].item()

    def ints(self, stmt: Statement, key: str, default=None) -> list[int] | None:
        value = self.arg(stmt, key, default)
        if value is None:
            return None
        return [int(v) for v in np.atleast_1d(np.asarray(value)).ravel()]

    def bools(self, stmt: Statement, key: str) -> list[bool] | None:
        value = self.arg(stmt, key)
        if value is None:
            return None
        return [bool(v) for v in np.atleast_1d(np.asarray(value)).ravel()]

    # ---------------------------------------------------------------- const

    def _literal(self, text: str):
        scalar = re.fullmatch(r"(string|bool|int32|fp16|fp32)\((.*)\)", text.strip())
        if scalar is not None:
            kind, payload = scalar.groups()
            if kind == "string":
                value = json.loads(payload)
                if not isinstance(value, str):
                    raise MilError("string literal must hold a string")
                return value
            dtype = DTYPES[kind]
            return np.asarray(_scalar(payload, dtype), dtype=dtype)
        match = LITERAL.match(text.strip())
        if match is None:
            raise MilError(f"unparsed literal {text[:80]}")
        dtype_name, shape_text, payload = match.groups()
        _, shape = parse_type(f"tensor<{dtype_name}, [{shape_text}]>")
        payload = payload.strip()
        if dtype_name == "string":
            items = [x.strip().strip('"') for x in split_top(payload.strip("[]"))]
            return items if shape else items[0]
        dtype = DTYPES[dtype_name]
        blob = BLOBFILE.search(payload)
        if blob is not None:
            return self.blobs.read(blob.group("path"), int(blob.group("offset")), dtype, shape)
        items = [_scalar(x, dtype) for x in split_top(payload.strip("[]"))]
        return np.array(items, dtype=dtype).reshape(shape)

    def _palette_blob(self, token: str, dtype: str, storage: int):
        match = LITERAL.fullmatch(token)
        if match is None or match.group(1) != dtype:
            raise MilError(f"palette operand must be a blob-backed {dtype} tensor")
        _, shape = parse_type(f"tensor<{dtype}, [{match.group(2)}]>")
        if not shape or any(size <= 0 for size in shape):
            raise MilError("palette tensor dimensions must be positive")
        blob = BLOBFILE.fullmatch(match.group(3))
        if blob is None:
            raise MilError("palette operand must use BLOBFILE storage")
        path, offset = blob.group("path"), int(blob.group("offset"))
        weights = self.blobs.root / "weights"
        payload = _read_blob(weights, path, offset, "palette")
        with (weights / path.split("/")[-1]).open("rb") as stream:
            metadata = read_blob_metadata(stream, offset)
        if metadata.dtype_code != storage:
            raise MilError(f"palette {dtype} blob has storage code {metadata.dtype_code}, expected {storage}")
        return shape, payload

    def _palette(self, stmt: Statement) -> np.ndarray:
        if set(stmt.kwargs) != {"indices", "lut"}:
            raise MilError("palette requires exactly indices and lut")
        shape, indices = self._palette_blob(stmt.kwargs["indices"], "uint4", 11)
        lut_shape, lut = self._palette_blob(stmt.kwargs["lut"], "fp16", 1)
        if len(lut_shape) != len(shape) + 2:
            raise MilError("palette LUT rank must be indices rank plus two")
        if any(size % blocks for size, blocks in zip(shape, lut_shape)):
            raise MilError("palette blocks must divide each indices dimension")
        if stmt.dtypes != ["fp16"] or stmt.shapes != [shape]:
            raise MilError("palette output must match the fp16 LUT dtype and indices shape")
        dense = _gather_dense(indices, 35, shape, lut, 10, lut_shape)
        return np.frombuffer(dense, dtype=np.float16).reshape(shape)

    # -------------------------------------------------------------- execute

    def execute(self, stmt: Statement) -> None:
        if stmt.done:
            return
        stmt.done = True
        self.executed += 1
        if stmt.op == "const":
            self.values[stmt.names[0]] = self._literal(parse_kwargs(stmt.attrs)["val"])
            return
        results = self.apply(stmt)
        if not isinstance(results, list):
            results = [results]
        if len(results) != len(stmt.names):
            raise MilError(f"{stmt.op} produced {len(results)} of {len(stmt.names)}")
        for name, dtype, shape, value in zip(stmt.names, stmt.dtypes, stmt.shapes, results):
            out = np.asarray(value).astype(DTYPES[dtype])
            if out.shape != shape:
                out = out.reshape(shape)
            self.values[name] = out

    def run(
        self, inputs: dict[str, np.ndarray], wanted: set[str], stop_after: str | None = None
    ) -> dict[str, np.ndarray]:
        """Evaluate front to back, releasing tensors after their last reader.
        Returns the ``wanted`` tensors; stops after ``stop_after`` is produced."""
        for name, value in inputs.items():
            if name not in self.inputs:
                raise MilError(f"{name!r} is not a function input")
            dtype, shape = self.inputs[name]
            array = np.asarray(value, dtype=DTYPES[dtype])
            if tuple(array.shape) != shape:
                raise MilError(f"input {name} is {array.shape}, expected {shape}")
            self.values[name] = array
        keep: dict[str, np.ndarray] = {}
        protected = set(wanted) | {stop_after}
        for stmt in self.statements:
            self.execute(stmt)
            for name in stmt.names:
                if name in wanted:
                    keep[name] = self.values[name]
            for name in self.release_at.get(stmt.index, ()):
                if name not in protected:
                    self.values.pop(name, None)
            if stop_after is not None and stop_after in stmt.names:
                break
        else:
            if stop_after is not None:
                raise MilError(f"never reached {stop_after}")
        missing = set(wanted) - keep.keys()
        if missing:
            raise MilError(f"never produced {sorted(missing)}")
        return keep

    # ------------------------------------------------------------------ ops

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        positive = x >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        return out

    @staticmethod
    def _f32(value) -> np.ndarray:
        return np.asarray(value).astype(np.float32)

    def apply(self, stmt: Statement):
        op = stmt.op
        arg = lambda key, default=None: self.arg(stmt, key, default)
        ints = lambda key, default=None: self.ints(stmt, key, default)

        if op == "constexpr_lut_to_dense":
            return self._palette(stmt)
        if op == "identity":
            return arg("x")
        if op == "cast":
            return np.asarray(arg("x")).astype(DTYPES[self.scalar(stmt, "dtype")])
        if op in ("add", "sub", "mul"):
            x, y = np.asarray(arg("x")), np.asarray(arg("y"))
            if x.dtype.kind == "f" or y.dtype.kind == "f":
                x, y = self._f32(x), self._f32(y)
            return {"add": x + y, "sub": x - y, "mul": x * y}[op]
        if op == "floor_div":
            x, y = np.asarray(arg("x")), np.asarray(arg("y"))
            if x.dtype.kind == "f" or y.dtype.kind == "f":
                return np.floor(self._f32(x) / self._f32(y))
            return np.floor_divide(x, y)
        if op == "floor":
            return np.floor(self._f32(arg("x")))
        if op == "less":
            return np.less(arg("x"), arg("y"))
        if op == "greater":
            return np.greater(np.asarray(arg("x")), np.asarray(arg("y")))
        if op == "logical_not":
            return np.logical_not(arg("x"))
        if op == "logical_and":
            return np.logical_and(arg("x"), arg("y"))
        if op == "relu":
            return np.maximum(self._f32(arg("x")), 0.0)
        if op == "sigmoid":
            return self._sigmoid(self._f32(arg("x")))
        if op == "silu":
            x = self._f32(arg("x"))
            return x * self._sigmoid(x)
        if op == "select":
            return np.where(arg("cond"), arg("a"), arg("b"))

        if op == "reshape":
            return np.asarray(arg("x")).reshape(ints("shape"))
        if op == "transpose":
            x = np.asarray(arg("x"))
            return np.transpose(x, [p % x.ndim for p in ints("perm")])
        if op == "expand_dims":
            x = np.asarray(arg("x"))
            axes = ints("axes")
            for axis in sorted(a % (x.ndim + len(axes)) for a in axes):
                x = np.expand_dims(x, axis)
            return x
        if op == "squeeze":
            x = np.asarray(arg("x"))
            axes = ints("axes")
            return np.squeeze(x) if axes is None else np.squeeze(x, tuple(a % x.ndim for a in axes))
        if op == "tile":
            return np.tile(arg("x"), ints("reps"))
        if op == "concat":
            values = arg("values")
            if values is None:
                keys = sorted((k for k in stmt.kwargs if re.fullmatch(r"x\d+", k)), key=lambda k: int(k[1:]))
                values = [arg(k) for k in keys]
            return np.concatenate([np.asarray(v) for v in values], axis=int(self.scalar(stmt, "axis")))
        if op == "split":
            x = np.asarray(arg("x"))
            axis = int(self.scalar(stmt, "axis"))
            sizes = ints("split_sizes")
            if sizes is not None:
                return list(np.split(x, np.cumsum(sizes)[:-1], axis=axis))
            return list(np.split(x, int(self.scalar(stmt, "num_splits", len(stmt.names))), axis=axis))
        if op == "slice_by_index":
            return self._slice(stmt)
        if op == "pad":
            mode = self.scalar(stmt, "mode", "constant")
            if mode != "constant":
                raise MilError(f"pad mode {mode!r} is not implemented")
            x = self._f32(arg("x"))
            pad = ints("pad")
            pairs = [(pad[i], pad[i + 1]) for i in range(0, len(pad), 2)][-x.ndim :]
            widths = [(0, 0)] * (x.ndim - len(pairs)) + pairs
            value = self.scalar(stmt, "constant_val", 0.0)
            return np.pad(x, widths, mode="constant", constant_values=float(value))

        if op in ("reduce_sum", "reduce_min", "reduce_max"):
            x = np.asarray(arg("x"))
            if op == "reduce_sum" and x.dtype.kind == "f":
                x = self._f32(x)
            fn = {"reduce_sum": np.sum, "reduce_min": np.min, "reduce_max": np.max}[op]
            keep = bool(self.scalar(stmt, "keep_dims", False))
            return fn(x, axis=tuple(ints("axes")), keepdims=keep)
        if op == "softmax":
            x = self._f32(arg("x"))
            axis = int(self.scalar(stmt, "axis"))
            exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return exp / np.sum(exp, axis=axis, keepdims=True)
        if op == "layer_norm":
            x = self._f32(arg("x"))
            axes = tuple(a % x.ndim for a in ints("axes"))
            eps = float(self.scalar(stmt, "epsilon", 1e-5))
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.mean((x - mean) ** 2, axis=axes, keepdims=True)
            out = (x - mean) / np.sqrt(var + eps)
            gamma, beta = arg("gamma"), arg("beta")
            if gamma is not None:
                out = out * self._f32(gamma)
            if beta is not None:
                out = out + self._f32(beta)
            return out

        if op == "linear":
            out = self._f32(arg("x")) @ self._f32(arg("weight")).T
            bias = arg("bias")
            return out if bias is None else out + self._f32(bias)
        if op == "matmul":
            a, b = self._f32(arg("x")), self._f32(arg("y"))
            if self.scalar(stmt, "transpose_x", False):
                a = np.swapaxes(a, -1, -2)
            if self.scalar(stmt, "transpose_y", False):
                b = np.swapaxes(b, -1, -2)
            return a @ b
        if op == "conv":
            return self._conv(stmt)
        raise MilError(f"unimplemented MIL op {op!r} ({stmt.names[0]})")

    def _slice(self, stmt: Statement) -> np.ndarray:
        x = np.asarray(self.arg(stmt, "x"))
        begin, end = self.ints(stmt, "begin"), self.ints(stmt, "end")
        strides = self.ints(stmt, "stride") or self.ints(stmt, "strides")
        begin_mask, end_mask = self.bools(stmt, "begin_mask"), self.bools(stmt, "end_mask")
        squeeze_mask = self.bools(stmt, "squeeze_mask")
        index = []
        for axis in range(x.ndim):
            if squeeze_mask and squeeze_mask[axis]:
                index.append(begin[axis])
                continue
            lo = None if (begin_mask and begin_mask[axis]) else begin[axis]
            hi = None if (end_mask and end_mask[axis]) else end[axis]
            index.append(slice(lo, hi, strides[axis] if strides else 1))
        return x[tuple(index)]

    def _conv(self, stmt: Statement) -> np.ndarray:
        """MIL conv (NCHW) for spatial rank 1 and 2; rank 1 lifts to a unit H axis.
        im2col through stride tricks, one group at a time; depthwise in one pass."""
        x = self._f32(self.arg(stmt, "x"))
        weight = self._f32(self.arg(stmt, "weight"))
        bias = self.arg(stmt, "bias")
        rank = x.ndim - 2
        strides = self.ints(stmt, "strides", [1] * rank)
        dilations = self.ints(stmt, "dilations", [1] * rank)
        pad_type = self.scalar(stmt, "pad_type", "valid")
        pad = self.ints(stmt, "pad", [])
        groups = int(self.scalar(stmt, "groups", 1))
        if rank == 1:
            x, weight = x[:, :, None, :], weight[:, :, None, :]
            strides, dilations, pad = [1, strides[0]], [1, dilations[0]], [0, 0, *pad]
        elif rank != 2:
            raise MilError(f"conv spatial rank {rank} is not implemented")
        if tuple(dilations) != (1, 1):
            raise MilError(f"dilated conv {dilations} is not implemented")
        n, cin, h, w = x.shape
        cout, cin_g, kh, kw = weight.shape
        sh, sw = strides
        if pad_type == "valid":
            pad = [0, 0, 0, 0]
        elif pad_type == "same":
            pad = []
            for size, kernel, stride in ((h, kh, sh), (w, kw, sw)):
                total = max(0, (size - 1) * stride + kernel - size)
                pad += [total // 2, total - total // 2]
        elif pad_type != "custom":
            raise MilError(f"conv pad_type {pad_type!r} is not implemented")
        if cin_g * groups != cin:
            raise MilError(f"conv channel mismatch {x.shape} {weight.shape} groups={groups}")
        top, bottom, left, right = pad
        xp = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))
        oh = (h + top + bottom - kh) // sh + 1
        ow = (w + left + right - kw) // sw + 1
        s = xp.strides
        cols = np.lib.stride_tricks.as_strided(
            xp,
            shape=(n, cin, kh, kw, oh, ow),
            strides=(s[0], s[1], s[2], s[3], s[2] * sh, s[3] * sw),
            writeable=False,
        )
        per = cout // groups
        if cin_g == 1 and per == 1:
            out = np.einsum("nckhij,ckh->ncij", cols, weight[:, 0], optimize=True)
        else:
            out = np.empty((n, cout, oh, ow), dtype=np.float32)
            for g in range(groups):
                xg = cols[:, g * cin_g : (g + 1) * cin_g].reshape(n, cin_g * kh * kw, oh * ow)
                wg = weight[g * per : (g + 1) * per].reshape(per, cin_g * kh * kw)
                out[:, g * per : (g + 1) * per] = (wg @ xg).reshape(n, per, oh, ow)
        if bias is not None:
            out = out + self._f32(bias).reshape(1, cout, 1, 1)
        return out[:, :, 0, :] if rank == 1 else out


# ---------------------------------------------------------------- validation

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(source: Path, capture: Path, contract: dict) -> dict:
    """Run the whole encoder on the capture inputs and score it against the
    captured golden under the reference lock's frozen encoder contract."""
    source, capture = Path(source), Path(capture)
    program = Program(source / "model.mil", source / "model-root")
    started = time.monotonic()
    keep = program.run(
        inputs={
            "input_features": np.load(capture / "encoder_input_features.npy").reshape(1, 3000, 128),
            "attention_mask": np.load(capture / "encoder_input_mask.npy").reshape(1, 3000),
        },
        wanted={"encoder_hidden", "encoder_mask"},
        stop_after="encoder_mask",
    )
    elapsed = time.monotonic() - started
    hidden = keep["encoder_hidden"].astype(np.float32)
    golden = np.load(capture / "encoder_hidden.npy").astype(np.float32)
    golden_mask = np.load(capture / "encoder_mask.npy")
    mask = keep["encoder_mask"].astype(golden_mask.dtype).reshape(golden_mask.shape)
    diff = np.abs(hidden - golden)
    measured = {
        "encoder_max_abs_err": float(diff.max()),
        "encoder_mean_abs_err": float(diff.mean()),
        "encoder_rel_l2_err": float(np.linalg.norm(hidden - golden) / np.linalg.norm(golden)),
        "nan_count": int(np.isnan(hidden).sum()),
        "inf_count": int(np.isinf(hidden).sum()),
    }
    bounds = {
        "encoder_max_abs_err": contract["encoder_max_abs_err"],
        "encoder_mean_abs_err": contract["encoder_mean_abs_err"],
        "encoder_rel_l2_err": contract["encoder_rel_l2_err"],
        "nan_count": contract.get("nan_count_allowed", 0),
        "inf_count": contract.get("inf_count_allowed", 0),
    }
    return {
        "schema": "mlx-omarchy.mil-numpy-validation.v2",
        "mil": str(source / "model.mil"),
        "mil_sha256": sha256(source / "model.mil"),
        "capture": str(capture),
        "capture_sha256": {
            name: sha256(capture / name)
            for name in ("encoder_input_features.npy", "encoder_input_mask.npy", "encoder_hidden.npy")
        },
        "ops_executed": program.executed,
        "elapsed_s": round(elapsed, 1),
        "encoder_hidden_shape": list(hidden.shape),
        "encoder_mask_equal": bool(np.array_equal(mask, golden_mask)),
        "measured": measured,
        "bounds": bounds,
        "within_frozen_contract": all(measured[k] <= bounds[k] for k in bounds),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--reference-lock",
        type=Path,
        default=Path(__file__).with_name("parakeet-reference.lock"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    contract = json.loads(args.reference_lock.read_text())["numerical_contract"]
    report = validate(args.source, args.capture, contract)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(text)
    print(text, end="")
    return 0 if report["within_frozen_contract"] and report["encoder_mask_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
