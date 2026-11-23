"""torch.compile backend 'mlx_vulkan': lowers dynamo FX graphs to mlx (omarchy-mlx, Vulkan/Apple GPU).

dynamo hands us torch-level graphs: call_function targets are torch.* /
torch.nn.functional.* builtins (and, for AOT-style graphs, aten.* OpOverloads),
call_method targets are Tensor method names. Three dispatch outcomes:

  LOWERED   -> executed through mlx.core on the GPU
  GLUE      -> python/shape bookkeeping, no kernel on either side
  FALLBACK  -> executed through torch on CPU (mx<->torch boundary conversion)

Every decision is logged to the 'mlx_vulkan' logger; that log is the
op-coverage evidence for the milestone.
"""
from __future__ import annotations

import logging
import math
import operator
import os

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import torch._dynamo

log = logging.getLogger("mlx_vulkan")

# ---------------- conversions ----------------

def _mx(v):
    if isinstance(v, mx.array):
        return v
    if isinstance(v, torch.Tensor):
        if v.dtype == torch.bfloat16:
            return mx.view(mx.array(v.detach().cpu().view(torch.uint16).numpy()), mx.bfloat16)
        return mx.array(v.detach().cpu().numpy())
    return v

def _torch(v):
    if isinstance(v, mx.array):
        return torch.from_numpy(np.array(v, copy=True))
    if isinstance(v, (tuple, list)):
        return type(v)(_torch(x) for x in v)
    return v

def _opname(node):
    """aten.mm.default -> mm ; aten::mm -> mm ; else the raw target string."""
    t = str(node.target)
    if t.startswith("aten."):
        return t[5:].split(".")[0]
    if t.startswith("aten::"):
        return t[6:]
    return t

# ---------------- helpers ----------------

def _perm_swap(a, d0, d1):
    n = a.ndim
    d0, d1 = d0 % n, d1 % n
    p = list(range(n))
    p[d0], p[d1] = p[d1], p[d0]
    return tuple(p)

def _wt(w):
    return mx.transpose(w, axes=_perm_swap(w, -2, -1))

def _linear(x, w, b=None):
    y = mx.matmul(x, _wt(w))
    return y + _mx(b) if b is not None else y

def _reshape(a, *s):
    if len(s) == 1 and isinstance(s[0], (list, tuple)):
        s = tuple(s[0])
    if len(s) == 1 and isinstance(s[0], torch.Size):
        s = tuple(s[0])
    return mx.reshape(a, tuple(s))

def _slice(a, dim, start, end, step):
    dim = dim % a.ndim
    idx = [slice(None)] * a.ndim
    idx[dim] = slice(start, end, step)
    return a[tuple(idx)]

def _split_sizes(a, sizes, dim):
    idx = list(np.cumsum(sizes))[:-1]
    return tuple(mx.split(a, idx, axis=dim))

def _split_even(a, split_size, dim):
    nsec = -(-a.shape[dim] // split_size)
    return tuple(mx.split(a, nsec, axis=dim))

def _chunk(a, chunks, dim):
    sz = a.shape[dim]
    step = -(-sz // chunks)
    return _split_sizes(a, [min(step, sz - i * step) for i in range(chunks) if i * step < sz], dim)

def _unbind(a, dim):
    dim = dim % a.ndim
    idx = [slice(None)] * a.ndim
    out = []
    for j in range(a.shape[dim]):
        idx[dim] = j
        out.append(a[tuple(idx)])
    return tuple(out)

def _ln_native(a, normalized_shape, weight, bias, eps):
    """native_layer_norm with real mean/rstd.

    aot_autograd saves the returned mean and rstd and feeds them to
    aten.native_layer_norm_backward; dummies (the old lowering) make that
    backward divide by rstd = 0 and every gradient upstream of the norm
    turns NaN at the first training step (PT-T1 bisect). The stats reduce
    over the trailing len(normalized_shape) axes and keep them singleton,
    matching aten's saved shapes for any normalized_shape.
    """
    w = _mx(weight) if weight is not None else None
    b = _mx(bias) if bias is not None else None
    axes = tuple(range(a.ndim - len(normalized_shape), a.ndim))
    mu = a.mean(axis=axes, keepdims=True)
    rstd = mx.rsqrt(
        a.square().mean(axis=axes, keepdims=True) - mu.square() + eps)
    out = (a - mu) * rstd
    out = out * w if w is not None else out
    out = out + b if b is not None else out
    return (out, mu, rstd)

def _expand(a, *s):
    if len(s) == 1 and isinstance(s[0], (list, tuple)):
        s = tuple(s[0])
    shape = tuple(a.shape[i] if x == -1 else x for i, x in enumerate(s))
    return mx.broadcast_to(a, shape)

def _pad_width(a, p):
    pairs = [(0, 0)] * a.ndim
    dims = len(p) // 2
    for i in range(dims):
        pairs[a.ndim - 1 - i] = (int(p[2 * i]), int(p[2 * i + 1]))
    return tuple(pairs)

def _shape_of(a, k):
    if len(a) == 1:
        return tuple(a[0]) if isinstance(a[0], (list, tuple)) else (a[0],)
    return tuple(a) if a else tuple(k.get("size", ()))

def _cat_impl(t, dim=0):
    xs = [_mx(x) for x in t]
    nd = max(x.ndim for x in xs)
    keep = [x for x in xs if x.size != 0 or x.ndim == nd]
    if not keep:
        keep = xs
    if len(keep) == 1:
        return keep[0]
    return mx.concatenate(keep, axis=dim % nd)

def _mean(a, dim=None, keepdim=False, **k):
    return mx.mean(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim)

def _sum(a, dim=None, keepdim=False, **k):
    return mx.sum(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim)

def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    kw = {}
    # mlx 0.32's fast binding requires scale explicitly (no default).
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    kw["scale"] = scale
    if attn_mask is not None:
        kw["mask"] = _mx(attn_mask)
    elif is_causal:
        kw["mask"] = "causal"
    return mx.fast.scaled_dot_product_attention(q, k, v, **kw)

def _conv2d(a, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    # torch: a NCHW, w OIHW ; mlx.conv2d: x NHWC, w (H, W, I/groups, O)
    if groups != 1:
        return _raise("grouped conv2d")
    x = _mx(a).transpose(0, 2, 3, 1)
    wt = _mx(w).transpose(0, 2, 3, 1)  # OIHW -> O,H,W,I (mlx conv2d weight layout)
    p = tuple(padding) if isinstance(padding, (list, tuple)) else (padding, padding)
    st = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride)
    dl = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, dilation)
    y = mx.conv2d(x, wt, stride=st, padding=p, dilation=dl)
    if b is not None:
        y = y + _mx(b).reshape(1, 1, 1, -1)
    return y.transpose(0, 3, 1, 2)

def _group_norm(a, num_groups, w=None, b=None, eps=1e-5):
    # torch NCHW; mlx 0.32 has no fast.group_norm - normalize per group.
    x = _mx(a)
    n, c = x.shape[0], x.shape[1]
    xg = x.reshape(n, num_groups, -1)
    mean = xg.mean(axis=-1, keepdims=True)
    var = ((xg - mean) ** 2).mean(axis=-1, keepdims=True)
    xn = ((xg - mean) / mx.sqrt(var + eps)).reshape(x.shape)
    if w is not None:
        xn = xn * _mx(w).reshape(1, -1, 1, 1)
    if b is not None:
        xn = xn + _mx(b).reshape(1, -1, 1, 1)
    return xn

def _interpolate(a, size=None, scale_factor=None, mode="nearest",
                 align_corners=None, recompute_scale_factor=None, **k):
    # torch NCHW; nearest-neighbor upsample (the SD UNet upsampler mode).
    if mode != "nearest":
        return _raise("interpolate mode " + str(mode))
    x = _mx(a).transpose(0, 2, 3, 1)  # NHWC
    n, h, w, c = x.shape
    if size is not None:
        th, tw = (size if isinstance(size, (list, tuple)) else (size, size))
    else:
        sf = scale_factor if isinstance(scale_factor, (list, tuple)) else (scale_factor, scale_factor)
        th, tw = int(h * sf[0]), int(w * sf[1])
    yi = (mx.arange(th) * (h / th)).astype(mx.int32)
    xi = (mx.arange(tw) * (w / tw)).astype(mx.int32)
    return x[:, yi][:, :, xi].transpose(0, 3, 1, 2)

# ---------------- lowering tables ----------------

# torch-level function objects -> lowering. Built lazily below so missing
# attributes on any given torch build can't break the import.
def _embedding(weight, idx):
    """aten.embedding(weight, indices) order."""
    return _mx(weight)[_mx(idx).astype(mx.int32)]

def _embedding_f(input, weight, *a_, **k_):
    """F.embedding(input, weight) order."""
    return _embedding(weight, input)

_FUNC_SPECS = {
    "matmul": mx.matmul, "bmm": mx.matmul, "mm": mx.matmul,
    "addmm": lambda b, m1, m2, beta=1, alpha=1: beta * _mx(b) + alpha * mx.matmul(m1, m2),
    "baddbmm": lambda b, m1, m2, beta=1, alpha=1: beta * _mx(b) + alpha * mx.matmul(m1, m2),
    "add": lambda a, b, *k: _mx(a) + _mx(b),
    "sub": lambda a, b, *k: _mx(a) - _mx(b),
    "mul": lambda a, b, *k: _mx(a) * _mx(b),
    "div": lambda a, b, *k: _mx(a) / _mx(b),
    "truediv": lambda a, b, *k: _mx(a) / _mx(b),
    "neg": lambda a: -a,
    "pow": lambda a, b: mx.power(_mx(a), _mx(b)),
    "sqrt": mx.sqrt, "rsqrt": mx.rsqrt, "exp": mx.exp, "log": mx.log,
    "sigmoid": mx.sigmoid, "tanh": mx.tanh, "erf": mx.erf,
    "transpose": lambda a, d0, d1: mx.transpose(a, axes=_perm_swap(a, d0, d1)),
    "permute": lambda a, *d: mx.transpose(a, axes=tuple(d[0]) if len(d) == 1 else tuple(d)),
    "reshape": _reshape, "view": _reshape,
    "flatten": lambda a, start=0, end=-1: mx.reshape(a, (*a.shape[:start], int(np.prod(a.shape[start:end + 1])), *a.shape[end + 1:]) if end != -1 else (*a.shape[:start], int(np.prod(a.shape[start:])))),
    "unsqueeze": lambda a, d: mx.expand_dims(a, axis=d),
    "squeeze": lambda a, d=None: mx.squeeze(a, axis=d) if d is not None else mx.squeeze(a),
    "expand": _expand, "expand_as": lambda a, o: mx.broadcast_to(a, o.shape),
    "cat": _cat_impl,
    "concat": _cat_impl,
    "stack": lambda t, dim=0: mx.stack([_mx(x) for x in t], axis=dim),
    "split": lambda a, ss, dim=0: _split_even(a, ss, dim) if isinstance(ss, int) else _split_sizes(a, ss, dim),
    "split_with_sizes": lambda a, ss, dim=0: _split_sizes(a, ss, dim),
    "chunk": _chunk, "unbind": lambda a, dim=0: _unbind(a, dim),
    "select": lambda a, dim, i: _slice(a, dim, i, i + 1, 1)[tuple(0 if d == dim % a.ndim else slice(None) for d in range(a.ndim))],
    "slice": _slice, "narrow": lambda a, dim, s, n: _slice(a, dim, s, s + n, 1),
    "mean": _mean, "sum": _sum,
    "max": lambda a, d=None, keepdim=False, *k: mx.max(a, axis=d, keepdims=keepdim) if d is not None else mx.max(a),
    "min": lambda a, d=None, keepdim=False, *k: mx.min(a, axis=d, keepdims=keepdim) if d is not None else mx.min(a),
    "where": lambda c, a, b: mx.where(_mx(c), _mx(a), _mx(b)),
    "zeros": lambda *a, **k: mx.zeros(_shape_of(a, k)),
    "zeros_like": lambda a: mx.zeros(a.shape),
    "ones": lambda *a, **k: mx.ones(_shape_of(a, k)),
    "ones_like": lambda a: mx.ones(a.shape),
    "full": lambda s, v, **k: mx.full(tuple(s), v),
    "full_like": lambda a, v, **k: mx.full(a.shape, v),
    "tril": lambda a, d=0: mx.tril(a, d) if hasattr(mx, "tril") else _raise("tril"),
    "getitem": lambda t, i: t[i],
    "detach": lambda a: a, "clone": lambda a: a, "contiguous": lambda a: a,
    "alias": lambda a: a, "lift_fresh": lambda a: a, "to": lambda a, *a_, **k_: a,
    "_to_copy": lambda a, **k_: a, "size": lambda a, d=None: a.shape[d] if d is not None else a.shape,
    "native_layer_norm": _ln_native,
    "dropout": lambda a, p=0.5, training=False, *k: a,
    "conv2d": _conv2d,
    "group_norm": _group_norm,
    "interpolate": _interpolate,
    "softmax": lambda a, dim=-1, dtype=None, **k: mx.softmax(a, axis=dim),
    "_softmax": lambda a, dim, half=False: mx.softmax(a, axis=dim),
    "gelu": lambda a, approximate="none": a * mx.sigmoid(1.702 * a) if approximate == "tanh" else 0.5 * a * (1.0 + mx.erf(a / math.sqrt(2.0))),
    "silu": lambda a, inplace=False: a * mx.sigmoid(a),
    "relu": lambda a, inplace=False: mx.maximum(a, 0.0),
    "scaled_dot_product_attention": _sdpa,
    "eq": lambda a, b: mx.equal(_mx(a), _mx(b)),
    "ne": lambda a, b: mx.not_equal(_mx(a), _mx(b)),
    "gt": lambda a, b: mx.greater(_mx(a), _mx(b)),
    "lt": lambda a, b: mx.less(_mx(a), _mx(b)),
    "cos": mx.cos, "sin": mx.sin, "log": mx.log, "log1p": mx.log1p,
    "outer": mx.outer, "einsum": lambda *a: mx.einsum(*a),
    "argmax": lambda a, dim=None, keepdim=False: mx.argmax(a, axis=dim, keepdims=keepdim),
    "argmin": lambda a, dim=None, keepdim=False: mx.argmin(a, axis=dim, keepdims=keepdim),
    "take": lambda a, i, dim=None: mx.take(a, _mx(i).astype(mx.int32), axis=dim),
    "index_select": lambda a, dim, i: mx.take(a, _mx(i).astype(mx.int32), axis=dim),
    "gather": lambda a, dim, i: mx.take_along_axis(a, _mx(i).astype(mx.int32), axis=dim),
    "take_along_dim": lambda a, i, dim: mx.take_along_axis(a, _mx(i).astype(mx.int32), axis=dim),
    "index": lambda a, idx: a[tuple(_mx(i) if isinstance(i, (torch.Tensor, list)) else i for i in (idx if isinstance(idx, (list, tuple)) else (idx,)))],
    "tril": lambda a, k=0: mx.tril(a, k), "triu": lambda a, k=0: mx.triu(a, k),
    "pad": lambda a, p, mode="constant", value=0: mx.pad(a, _pad_width(a, p)),
    "tile": lambda a, d: mx.tile(a, tuple(d)),
    "repeat": lambda a, *d: mx.tile(a, tuple(d[0]) if len(d) == 1 and isinstance(d[0], (list, tuple)) else tuple(d)),
    "movedim": lambda a, src, dst: mx.moveaxis(a, src, dst),
    "moveaxis": lambda a, src, dst: mx.moveaxis(a, src, dst),
    "swapaxes": lambda a, d0, d1: mx.swapaxes(a, d0, d1),
    "abs": mx.abs, "sign": mx.sign,
    "maximum": mx.maximum, "minimum": mx.minimum, "fmax": mx.maximum, "fmin": mx.minimum,
    "floor": mx.floor, "ceil": mx.ceil, "round": mx.round, "floor_divide": mx.floor_divide,
    "negative": mx.negative, "isnan": mx.isnan, "isinf": mx.isinf,
    "isfinite": lambda a: mx.logical_not(mx.isinf(a)),
    "linspace": lambda s, e, n, **k: mx.linspace(s, e, n),
    "arange": lambda *a, **k: mx.arange(*a),
    "eye": lambda n, m=None, **k: mx.eye(n, m) if m else mx.eye(n),
    "ge": lambda a, b: mx.greater_equal(_mx(a), _mx(b)),
    "le": lambda a, b: mx.less_equal(_mx(a), _mx(b)),
    "greater": lambda a, b: mx.greater(_mx(a), _mx(b)),
    "less": lambda a, b: mx.less(_mx(a), _mx(b)),
    "logical_and": lambda a, b: mx.logical_and(_mx(a), _mx(b)),
    "logical_or": lambda a, b: mx.logical_or(_mx(a), _mx(b)),
    "logical_not": lambda a: mx.logical_not(a),
    "invert": lambda a: mx.logical_not(a),
    "clamp": lambda a, min=None, max=None: mx.clip(a, min if min is not None else -float("inf"), max if max is not None else float("inf")),
    "clip": lambda a, min=None, max=None: mx.clip(a, min if min is not None else -float("inf"), max if max is not None else float("inf")),
    "rsub": lambda a, b: _mx(b) - _mx(a),
    "masked_fill": lambda a, m, v: mx.where(_mx(m).astype(mx.bool_), v, a),
    "masked_fill_": lambda a, m, v: mx.where(_mx(m).astype(mx.bool_), v, a),
    "cumsum": lambda a, dim, dtype=None: mx.cumsum(a, axis=dim),
    "std": lambda a, dim=None, correction=None, keepdim=False: mx.std(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "roll": lambda a, shifts, dims=None: mx.roll(a, shifts, axis=dims),
    "log_softmax": lambda a, dim=-1, **k: a - mx.log(mx.sum(mx.exp(a), axis=dim, keepdims=True)),
    "normalize": lambda x, p=2, dim=1, eps=1e-12: x / mx.maximum(mx.sum(mx.power(mx.abs(x), p), axis=dim, keepdims=True), eps),
    "norm": lambda a, p=2, dim=None, keepdim=False: mx.power(mx.sum(mx.power(mx.abs(a), float(p) if p not in ("inf", "-inf") else 2.0), axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim), 1.0 / (float(p) if p not in ("inf", "-inf") else 2.0)),
    "rms_norm": lambda a, ns, w=None, eps=None: mx.fast.rms_norm(a, _mx(w) if w is not None else None, eps if eps is not None else 1e-6),
    "embedding": _embedding_f,
    "all": lambda a, dim=None, keepdim=False: mx.all(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "any": lambda a, dim=None, keepdim=False: mx.any(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "count_nonzero": lambda a, dim=None: mx.sum((a != 0).astype(mx.int32), axis=dim),
    "empty": lambda *a, **k: mx.zeros(_shape_of(a, k)),
    "empty_like": lambda a, **k: mx.zeros(a.shape),
    "tensor": lambda d, **k: mx.array(np.array(_torch(d)) if isinstance(d, (torch.Tensor, list, tuple)) else d),
    "as_tensor": lambda d, **k: _mx(d),
    "einsum_fn": mx.einsum,
}

def _raise(name):
    raise NotImplementedError(name)

FUNC_TABLE = {}
_FN_SOURCES = [torch, F, torch.special, operator]

def _resolve(dotted):
    obj = _FN_SOURCES[0]
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj

def _build_func_table():
    candidates = []
    for fn_name, lowering in _FUNC_SPECS.items():
        for src in (torch, F, torch.special):
            fn = getattr(src, fn_name, None)
            if fn is not None:
                candidates.append((fn, lowering))
    # explicit functional mappings
    explicit = [
        ("torch.nn.functional.linear", _linear),
        ("torch.nn.functional.layer_norm", lambda a, ns, w=None, b=None, eps=1e-5: mx.fast.layer_norm(a, _mx(w) if w is not None else None, _mx(b) if b is not None else None, eps)),
        ("torch.nn.functional.gelu", _FUNC_SPECS["gelu"]),
        ("torch.nn.functional.silu", _FUNC_SPECS["silu"]),
        ("torch.nn.functional.relu", _FUNC_SPECS["relu"]),
        ("torch.nn.functional.softmax", _FUNC_SPECS["softmax"]),
        ("torch.nn.functional.scaled_dot_product_attention", _sdpa),
        ("torch.nn.functional.dropout", _FUNC_SPECS["dropout"]),
        ("torch.nn.functional.embedding", _embedding_f),
        ("operator.getitem", lambda t, i: t[i]),
        ("operator.matmul", mx.matmul),
        ("operator.add", _FUNC_SPECS["add"]),
        ("operator.sub", _FUNC_SPECS["sub"]),
        ("operator.mul", _FUNC_SPECS["mul"]),
        ("operator.truediv", _FUNC_SPECS["div"]),
        ("operator.floordiv", _FUNC_SPECS["div"]),
        ("operator.neg", _FUNC_SPECS["neg"]),
    ]
    for dotted, lowering in explicit:
        try:
            candidates.append((_resolve(dotted), lowering))
        except AttributeError:
            pass
    for fn, lowering in candidates:
        FUNC_TABLE[fn] = lowering

_build_func_table()

# name-keyed fallback: dynamo may reference e.g. torch._C._VariableFunctions.matmul
# which is a different object than torch.matmul
FUNC_TABLE_BY_NAME = {}
for _fn, _low in FUNC_TABLE.items():
    FUNC_TABLE_BY_NAME.setdefault(getattr(_fn, "__name__", ""), _low)

# ops that produce arrays from scalars/shapes — lower even with no mx inputs
_PRODUCERS = {"arange", "zeros", "ones", "full", "empty", "eye", "linspace",
              "tensor", "zeros_like", "ones_like", "full_like", "empty_like",
              "rand", "randn", "randint", "randperm", "new_zeros", "new_ones"}

# aten.* OpOverload name table (same lowerings, keyed by opname)
ATEN_TABLE = {name: f for name, f in _FUNC_SPECS.items()}
# aten.embedding is (weight, indices): F.embedding is (input, weight).
ATEN_TABLE["embedding"] = _embedding

# zero-kernel bookkeeping targets: run on python values, logged GLUE
GLUE_FNS = {len, tuple, list, dict, int, float, str, bool, slice, range, zip, torch.finfo, torch.iinfo,
            enumerate, getattr, sorted, max, min, sum, abs, round,
            torch._C._set_grad_enabled, torch._C._log_api_usage_once,
            operator.index, operator.length_hint, operator.not_, operator.truth,
            math.sqrt, math.ceil, math.floor, math.log, math.log2, math.prod}
try:
    GLUE_FNS.add(torch.sym_int)
except AttributeError:
    pass

_METHOD_DT = {"float": mx.float32, "double": mx.float32, "half": mx.float16,
              "bfloat16": mx.bfloat16, "int": mx.int32, "long": mx.int64,
              "short": mx.int16, "byte": mx.uint8, "char": mx.int8,
              "bool": getattr(mx, "bool_", mx.uint8)}
_MX_DT = {torch.float32: mx.float32, torch.float64: mx.float32, torch.float16: mx.float16,
          torch.bfloat16: mx.bfloat16, torch.int32: mx.int32, torch.int64: mx.int64,
          torch.int16: mx.int16, torch.uint8: mx.uint8, torch.int8: mx.int8,
          torch.bool: getattr(mx, "bool_", mx.uint8)}

_METHODS_ON_MX = set(_FUNC_SPECS) | {"size", "dim", "ndim", "shape", "dtype",
                                   "cpu", "to", "cuda", "numpy", "numel",
                                   "type", "item", "tolist"}

def _run_method(obj, name, args, kwargs):
    """Returns (value, on_mlx). Raises NotImplementedError for unknown methods on mx arrays."""
    if isinstance(obj, mx.array):
        if name in ("size",):
            return tuple(obj.shape)[args[0]] if args else tuple(obj.shape), True
        if name in ("dim", "ndim"):
            return obj.ndim, True
        if name in ("shape",):
            return tuple(obj.shape), True
        if name == "dtype":
            return obj.dtype, True
        if name == "numel":
            return int(np.prod(obj.shape)), True
        if name == "new_empty":
            return mx.zeros(tuple(args[0])), True
        if name == "new_zeros":
            return mx.zeros(tuple(args[0])), True
        if name in ("cpu", "detach", "type", "cuda"):
            return obj, True
        if name == "to":
            dt = next((a for a in args if isinstance(a, torch.dtype)), kwargs.get("dtype"))
            return obj.astype(_MX_DT.get(dt, obj.dtype)) if dt else obj, True
        if name in ("float", "half", "bfloat16", "int", "long", "double", "bool", "short", "byte", "char"):
            return obj.astype(_METHOD_DT[name]), True
        if name == "item":
            return float(np.array(obj).item()), True
        if name == "tolist":
            return np.array(obj).tolist(), True
        if name == "numpy":
            return np.array(obj), True
        key = name if name in _FUNC_SPECS else None
        if key is None:
            raise NotImplementedError(f"mx method {name}")
        return _FUNC_SPECS[key](obj, *args, **kwargs), True
    return getattr(obj, name)(*args, **kwargs), False

# ---------------- executor ----------------

def make_executor(gm):
    stats = {"mlx": [], "cpu": [], "glue": [], "crossings": 0}

    def resolve(a, env):
        if isinstance(a, torch.fx.Node):
            return env[a]
        if isinstance(a, (tuple, list, torch.fx.immutable_collections.immutable_list)):
            return type(a)(resolve(x, env) for x in a)
        return a

    def _deep_mx(v):
        if isinstance(v, torch.Tensor):
            return mx.array(v.detach().cpu().numpy())
        if isinstance(v, (tuple, list)):
            return type(v)(_deep_mx(x) for x in v)
        return v

    def has_mx(v):
        if isinstance(v, mx.array):
            return True
        return isinstance(v, (tuple, list)) and any(has_mx(x) for x in v)

    def run(*runtime_inputs):
        env = {}
        it = iter(runtime_inputs)
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                env[node] = _mx(next(it))
                continue
            args = tuple(resolve(a, env) for a in node.args)
            kwargs = {k: resolve(v, env) for k, v in node.kwargs.items()}
            name = _opname(node)

            if node.op == "output":
                out = args[0]
                return type(out)(_torch(x) for x in out) if isinstance(out, (tuple, list)) else _torch(out)
            if node.op == "get_attr":
                t = gm
                for part in str(node.target).split("."):
                    t = getattr(t, part)
                env[node] = _mx(t)
                continue
            if node.target is operator.getitem or name == "getitem":
                env[node] = args[0][args[1]]
                stats["glue"].append("getitem")
                continue

            val = None
            on_mlx = False
            if node.op == "call_function":
                lowering = (FUNC_TABLE.get(node.target)
                            or FUNC_TABLE_BY_NAME.get(getattr(node.target, "__name__", ""))
                            or ATEN_TABLE.get(name))
                is_producer = (getattr(node.target, "__name__", "") in _PRODUCERS
                               or name in _PRODUCERS)
                if lowering is not None and (has_mx(args) or is_producer):
                    try:
                        val = lowering(*_deep_mx(args), **{k: _deep_mx(v) for k, v in kwargs.items()})
                        on_mlx = True
                    except Exception as e:
                        def _sh(v):
                            if isinstance(v, (mx.array, torch.Tensor)):
                                return f"{type(v).__name__}{tuple(v.shape)}"
                            if isinstance(v, (tuple, list)):
                                return "(" + ",".join(_sh(x) for x in v) + ")"
                            return repr(v)[:24]
                        log.info("LOWER-FAIL\t%s\t%s\targs=%s", name, e, [_sh(a) for a in args])
                if not on_mlx and node.target in GLUE_FNS:
                    val = node.target(*args, **kwargs)
                    stats["glue"].append(getattr(node.target, "__name__", str(node.target)))
                    env[node] = val
                    continue
            elif node.op == "call_method":
                try:
                    val, on_mlx = _run_method(args[0], str(node.target), args[1:], kwargs)
                except NotImplementedError:
                    val, on_mlx = None, False
                except Exception as e:
                    log.info("LOWER-FAIL\t%s\t%s", node.target, e)
                    val, on_mlx = None, False

            if not on_mlx:
                if has_mx(args) or has_mx(kwargs.values()):
                    stats["crossings"] += 1
                targs = type(args)(_torch(a) for a in args)
                tkwargs = {k: _torch(v) for k, v in kwargs.items()}
                if node.op == "call_function":
                    val = node.target(*targs, **tkwargs)
                elif node.op == "call_method":
                    val = getattr(targs[0], str(node.target))(*targs[1:], **tkwargs)
                elif node.op == "call_module":
                    val = gm.get_submodule(str(node.target))(*targs, **tkwargs)
                else:
                    raise RuntimeError(f"unhandled {node.op} {node.target}")
                stats["cpu"].append(name)
                log.info("FALLBACK\t%s\t%s", name, node.target)
            else:
                stats["mlx"].append(name)
                log.info("LOWERED\t%s\t%s", name, node.target)
            env[node] = val
            if node.op == "output":
                return val
        return None

    return run, stats


def _eager_run(gm, return_all_outputs=False):
    run, stats = make_executor(gm)
    run._mlx_stats = stats
    return run


def _boxed(runner):
    """aot_autograd's runtime passes the inputs flat (some paths boxed as
    one list). Normalize to *args for the executor, which consumes one
    positional per placeholder."""
    def call(*inputs):
        flat = inputs
        if (len(inputs) == 1 and isinstance(inputs[0], (list, tuple))
                and not isinstance(inputs[0], mx.array)
                and all(isinstance(i, (torch.Tensor, mx.array))
                        for i in inputs[0])):
            flat = tuple(inputs[0])
        return runner(*flat)

    call._boxed_call = True
    return call


@torch._dynamo.register_backend(name="mlx_vulkan")
def mlx_vulkan_backend(gm: torch.fx.GraphModule, example_inputs,
                       return_all_outputs=False):
    try:
        run_c = make_compiled_executor(gm, return_all_outputs)
    except Exception as e:
        log.warning("compiled executor setup failed: %s", e)
        return _eager_run(gm, return_all_outputs)
    holder = {"run": None}

    def guarded(*args):
        if holder["run"] is None:
            try:
                out = run_c(*args)
            except Exception as e:
                import traceback as _tb; log.warning("compiled executor failed (%s); falling back to eager\n%s", e, _tb.format_exc())
                holder["run"] = _eager_run(gm)
            else:
                holder["run"] = run_c
                return out
        return holder["run"](*args)

    guarded._mlx_stats = run_c._mlx_stats
    guarded.return_all_outputs = return_all_outputs
    return guarded


# ================= compiled executor (v2) =================
# Whole-graph lazy lowering wrapped in mx.compile. Cache mutations
# (index_copy_/add_ on StaticCache buffers) are modeled functionally: updated
# tensors become extra outputs fed back as inputs next call, so HF's buffers
# only seed state once. RMSNorm chains fuse to mx.fast.rms_norm; the sdpa FX
# node maps to mx.fast.scaled_dot_product_attention; GEMM/sdpa inputs run bf16
# while the residual/elementwise spine stays fp32 -> ids match CPU fp32.

_DT = mx.bfloat16

# Optional weight quantization (MV_QUANT=4): static params whose every
# consumer chain ends at a matmul-family op are stored packed via mx.quantize
# and matmul lowerings emit mx.quantized_matmul. Transparent ops propagate a
# transpose flag; any other consumer dequantizes on the fly.
_Q_BITS = int(os.environ.get("MV_QUANT", "0") or 0)
_Q_GS = int(os.environ.get("MV_QUANT_GS", "64"))
_Q_MM = {"linear", "matmul", "mm", "bmm", "addmm"}
_Q_TR = {"t", "transpose", "permute"}
_Q_PASS = {"to", "detach", "clone", "contiguous", "float", "half", "type"}


class _QW:
    __slots__ = ("w", "s", "b", "tr")

    def __init__(self, w, s, b, tr=False):
        self.w, self.s, self.b, self.tr = w, s, b, tr

    def dequant(self):
        return mx.dequantize(self.w, self.s, self.b, group_size=_Q_GS, bits=_Q_BITS)


def _dqall(v):
    if isinstance(v, _QW):
        return v.dequant()
    if isinstance(v, (tuple, list)):
        return type(v)(_dqall(x) for x in v)
    return v


def _node_name(n):
    if n.op == "call_method":
        return str(n.target)
    return getattr(n.target, "__name__", None) or _opname(n)


def _quantizable_phs(phs):
    """Static 2-D params used only by matmul-family ops (through transparent ops)."""
    ok = set()
    for p in phs:
        if not str(p.target).startswith("G_"):
            continue
        good, seen, stack = True, set(), list(p.users)
        while stack and good:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            nm = _node_name(u)
            if nm in _Q_MM:
                continue
            if nm in _Q_TR or nm in _Q_PASS:
                stack.extend(u.users)
                continue
            good = False
        if good:
            ok.add(str(p.target))
    return ok


def _fuse_rmsnorm(nodes):
    """x -> [to] -> pow(2) -> mean -> add(eps) -> rsqrt -> mul(x, r) -> [to] -> mul(w)"""
    fused = {}
    skip = set()
    consumers = {}
    nm = {}
    for n in nodes:
        nm[n] = getattr(n.target, "__name__", None) or _opname(n)
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                consumers.setdefault(a, []).append(n)
    for r in nodes:
        if nm[r] != "rsqrt":
            continue
        a = r.args[0]
        if not (isinstance(a, torch.fx.Node) and nm[a] == "add" and len(a.args) == 2):
            continue
        eps = a.args[1]
        if not isinstance(eps, (int, float)):
            continue
        m, p = a.args[0], None
        if not (isinstance(m, torch.fx.Node) and nm[m] == "mean"):
            continue
        p = m.args[0]
        if not (isinstance(p, torch.fx.Node) and nm[p] == "pow"):
            continue
        x_c = p.args[0]
        mm = next((c for c in consumers.get(r, [])
                   if nm[c] == "mul" and c.args[0] is x_c), None)
        if mm is None:
            continue
        tail = mm
        cand = consumers.get(tail, [])
        if len(cand) == 1 and nm[cand[0]] == "to":
            tail = cand[0]
            cand = consumers.get(tail, [])
        out = next((c for c in cand if nm[c] == "mul"), None)
        if out is None:
            continue
        w_node = out.args[1] if out.args[0] is tail else out.args[0]
        fused[out] = (x_c, w_node, float(eps), {r, a, m, p, mm, tail})
    inputs_used = set()
    for x_c, w_node, _, _c in fused.values():
        inputs_used.update({x_c, w_node})
    final = {}
    for out, (x_c, w_node, eps, chain) in fused.items():
        # skip a member only if every consumer stays inside the fused set and
        # it is not an input to another fused chain
        ok = all(c in chain or c in fused
                 for x in chain - inputs_used
                 for c in consumers.get(x, []))
        if ok:
            final[out] = (x_c, w_node, eps)
            skip.update(chain - inputs_used)
    return skip, final


def make_compiled_executor(gm, return_all_outputs=False):
    nodes = list(gm.graph.nodes)
    phs = [n for n in nodes if n.op == "placeholder"]
    skip, fused = _fuse_rmsnorm(nodes)
    fp32_params = {str(w.target) for _, w, _ in fused.values()}
    state_names = {str(n.target) for n in phs if "_cache_layers_" in str(n.target)}
    static_names = {str(n.target) for n in phs if str(n.target).startswith("G_")}
    mutated = [n for n in phs if str(n.target) in state_names]
    quant_names = _quantizable_phs(phs) if _Q_BITS else set()
    conv_static = {}
    state = {}
    holder = {"ar": None}

    def _cast(x, dt=_DT):
        return x.astype(dt) if isinstance(x, mx.array) and x.dtype != dt else x

    def build(*flat):
        env = {}
        for n, v in zip(phs, flat):
            env[n] = _QW(*v) if str(n.target) in quant_names and isinstance(v, (tuple, list)) else v

        def res(a):
            if isinstance(a, torch.fx.Node):
                if a not in env:
                    raise KeyError("node %s (%s) not in env" % (a, a.op))
                return env[a]
            if isinstance(a, (tuple, list)):
                return type(a)(res(x) for x in a)
            return a

        for n in nodes:
            if n.op == "placeholder" or n in skip:
                continue
            name = _opname(n)
            if n.op == "output":
                out = res(n.args[0])
                tail = [env[m] for m in mutated]
                return (*out, *tail) if isinstance(out, (tuple, list)) else (out, *tail)
            if n.op == "get_attr":
                t = gm
                for part in str(n.target).split("."):
                    t = getattr(t, part)
                env[n] = _mx(t)
                continue
            if n in fused:
                xn, wn, eps = fused[n]
                env[n] = mx.fast.rms_norm(env[xn], env[wn], eps)
                continue
            args = tuple(res(a) for a in n.args)
            kwargs = {k: res(v) for k, v in n.kwargs.items()}
            if any(isinstance(a, _QW) for a in args) or any(
                    isinstance(v, _QW) for v in kwargs.values()):
                done = False
                if isinstance(args[0], _QW):
                    if name in ("t", "transpose"):
                        env[n] = _QW(args[0].w, args[0].s, args[0].b, not args[0].tr)
                        done = True
                    elif name == "permute":
                        dims = args[1] if len(args) > 1 else tuple(args[1:])
                        if tuple(dims) == (1, 0):
                            env[n] = _QW(args[0].w, args[0].s, args[0].b, not args[0].tr)
                            done = True
                        elif tuple(dims) == (0, 1):
                            env[n] = args[0]
                            done = True
                    elif name in _Q_PASS:
                        env[n] = args[0]
                        done = True
                if not done and len(args) > 1 and isinstance(args[1], _QW):
                    q = args[1]
                    if name == "linear":
                        y = mx.quantized_matmul(_cast(args[0]), q.w, q.s, q.b,
                                                transpose=not q.tr,
                                                group_size=_Q_GS, bits=_Q_BITS)
                        if len(args) > 2 and args[2] is not None:
                            y = y + _cast(args[2])
                        env[n] = y.astype(mx.float32)
                        done = True
                    elif name in ("matmul", "mm", "bmm"):
                        env[n] = mx.quantized_matmul(_cast(args[0]), q.w, q.s, q.b,
                                                     transpose=q.tr,
                                                     group_size=_Q_GS, bits=_Q_BITS).astype(mx.float32)
                        done = True
                if done:
                    continue
                args = _dqall(args)
                kwargs = {k: _dqall(v) for k, v in kwargs.items()}
            if name == "index_copy_":
                t, dim, idx, src = args[0], args[1], args[2], args[3]
                smax = t.shape[dim]
                if holder["ar"] is None or holder["ar"].shape[0] != smax:
                    holder["ar"] = mx.arange(smax)
                ar = holder["ar"]
                inrange = ((ar >= idx.min()) & (ar <= idx.max())).reshape(
                    [1] * dim + [smax] + [1] * (t.ndim - dim - 1))
                shift = mx.clip(ar - idx.min(), 0, max(0, int(src.shape[dim]) - 1))
                upd = mx.where(inrange, mx.take(_cast(src), shift, axis=dim), t)
                env[n.args[0]] = upd      # model in-place mutation
                env[n] = upd
                continue
            if name == "add_":
                env[n.args[0]] = env[n.args[0]] + args[1]
                env[n] = env[n.args[0]]
                continue
            # The aot backward graph: these ops must stay on mx or the whole
            # training step crosses to torch per node (PT-T1 speed gate).
            if name == "t" and n.op == "call_function":
                env[n] = mx.swapaxes(args[0], -1, -2)
                continue
            if name in ("transpose", "transpose.int") and n.op == "call_function":
                a0 = args[0]
                d0, d1 = int(args[1]), int(args[2])
                env[n] = mx.swapaxes(a0, d0 % a0.ndim, d1 % a0.ndim)
                continue
            if name in ("t", "transpose", "transpose.int", "view",
                        "_unsafe_view", "detach", "gelu_backward",
                        "native_layer_norm_backward", "split") \
                    and n.op == "call_function":
                shape = args[1]
                if isinstance(shape, torch.fx.Node):
                    shape = env[shape]
                if isinstance(shape, int):
                    shape = (shape,)
                env[n] = mx.reshape(args[0], tuple(int(s) for s in shape))
                continue
            if name == "detach" and n.op == "call_function":
                env[n] = args[0]
                continue
            if name == "gelu_backward" and n.op == "call_function":
                go, x = _cast(args[0]), _cast(args[1])
                approx = args[2] if len(args) > 2 else "none"
                if approx == "tanh":
                    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
                    phi = 0.5 * (1.0 + mx.tanh(inner))
                    dphi = math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * x * x) \
                        * (1.0 - mx.tanh(inner) * mx.tanh(inner))
                    env[n] = go * (phi + x * dphi)
                else:
                    env[n] = go * (0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))
                                   + x * mx.exp(-(x * x) / 2.0)
                                   / math.sqrt(2.0 * math.pi))
                continue
            if name == "native_layer_norm_backward" and n.op == "call_function":
                go, inp = _cast(args[0]), _cast(args[1])
                nshape = args[2]
                mean, rstd, weight = _cast(args[3]), _cast(args[4]), args[5]
                out_mask = args[7] if len(args) > 7 else [True, True, True]
                w = _cast(weight) if weight is not None else None
                inp_nd = inp.ndim
                red = tuple(range(inp_nd - len(nshape), inp_nd))
                xhat = (inp - mean) * rstd
                dw = (go * xhat).sum(axis=red) if 1 in out_mask else None
                db = go.sum(axis=red) if 2 in out_mask else None
                dx = rstd * (go - go.mean(axis=red, keepdims=True)
                             - xhat * (go * xhat).mean(axis=red,
                                                       keepdims=True))
                if w is not None:
                    dx = dx * w
                env[n] = (dx, dw, db)
                continue
            if name == "scaled_dot_product_attention":
                # GQA: HF expands KV to q-heads via expand+reshape before sdpa;
                # mx.fast sdpa broadcasts fewer kv heads natively -> look through.
                def _unexpand(narg):
                    if getattr(narg, "op", None) == "call_method" and _opname(narg) in ("reshape", "view"):
                        narg = narg.args[0]
                    if getattr(narg, "op", None) == "call_method" and _opname(narg) == "expand":
                        return env.get(narg.args[0])
                    return None
                uk, uv = _unexpand(n.args[1]), _unexpand(n.args[2])
                if uk is not None and uk.ndim == 5 and uk.shape[2] == 1:
                    uk = uk[:, :, 0]
                if uv is not None and uv.ndim == 5 and uv.shape[2] == 1:
                    uv = uv[:, :, 0]
                q = _cast(args[0])
                k = _cast(uk) if uk is not None else _cast(args[1])
                v = _cast(uv) if uv is not None else _cast(args[2])
                mask = kwargs.get("attn_mask")
                env[n] = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=kwargs.get("scale"),
                    mask=_cast(mask) if isinstance(mask, mx.array) else mask).astype(mx.float32)
                continue
            if name in ("matmul", "mm", "bmm") and n.op == "call_function":
                env[n] = mx.matmul(_cast(args[0]), _cast(args[1])).astype(mx.float32)
                continue
            if name == "linear" and n.op == "call_function":
                y = mx.matmul(_cast(args[0]), mx.transpose(_cast(args[1]), axes=(-1, -2)))
                if len(args) > 2 and args[2] is not None:
                    y = y + _cast(args[2])
                env[n] = y.astype(mx.float32)
                continue
            if n.op == "call_method":
                val, on_mx = _run_method(args[0], name, args[1:], kwargs)
                if on_mx:
                    env[n] = val
                    continue
                env[n] = getattr(args[0], name)(*args[1:], **kwargs)
                continue
            if n.target is operator.getitem:
                env[n] = args[0][args[1]]
                continue
            if n.op == "call_function":
                low = (FUNC_TABLE.get(n.target)
                       or FUNC_TABLE_BY_NAME.get(getattr(n.target, "__name__", ""))
                       or ATEN_TABLE.get(name))
                if low is not None:
                    kw2 = {k: v for k, v in kwargs.items() if k != "device"}
                    env[n] = low(*args, **kw2)
                    continue
                if n.target in GLUE_FNS:
                    env[n] = n.target(*args, **kwargs)
                    continue
            raise RuntimeError(f"unlowerable node {name} {n.target}")
        raise RuntimeError("no output node")

    cfn = mx.compile(build)
    stats = {"fused_rms": len(fused), "state": len(state_names), "quant": len(quant_names)}
    log.warning("compiled executor: fused_rms=%d state=%d quant=%d",
                stats["fused_rms"], stats["state"], stats["quant"])

    import time as _t
    _profile = os.environ.get("MV_PROF")
    _acc = {"prep": 0.0, "cfn": 0.0, "conv": 0.0, "n": 0}

    def run(*runtime_inputs):
        t0 = _t.perf_counter()
        flat = []
        for n, v in zip(phs, runtime_inputs):
            name = str(n.target)
            if name in state:
                flat.append(state[name])
            elif name in conv_static:
                flat.append(conv_static[name])
            elif name in state_names:
                arr = _mx(v)
                if arr.ndim >= 2:
                    arr = _cast(arr)
                state[name] = arr
                flat.append(arr)
            else:
                arr = _mx(v)
                if name in static_names:
                    if name in quant_names:
                        try:
                            arr = mx.quantize(arr, group_size=_Q_GS, bits=_Q_BITS)
                            mx.eval(*arr)  # force now: lazy graphs would hold every bf16 source
                        except Exception:
                            quant_names.discard(name)
                            arr = _cast(arr)
                    elif name not in fp32_params:
                        arr = _cast(arr)
                    conv_static[name] = arr
                flat.append(arr)
        t1 = _t.perf_counter()
        outs = cfn(*flat)
        mx.eval(*outs)
        t2 = _t.perf_counter()
        logits, extra = outs[0], outs[1:]
        for m, arr in zip(mutated, extra):
            state[str(m.target)] = arr
        out = _torch(logits)
        t3 = _t.perf_counter()
        if _profile:
            _acc["prep"] += t1 - t0; _acc["cfn"] += t2 - t1; _acc["conv"] += t3 - t2; _acc["n"] += 1
            if _acc["n"] % 10 == 0:
                print(f"MV_PROF n={_acc['n']} prep={_acc['prep']/_acc['n']*1e3:.1f}ms cfn={_acc['cfn']/_acc['n']*1e3:.1f}ms conv={_acc['conv']/_acc['n']*1e3:.1f}ms", flush=True)
        if return_all_outputs:
            return tuple(_torch(t) for t in outs)
        return out

    run._mlx_stats = stats
    return run
