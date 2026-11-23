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
    w = _mx(weight) if weight is not None else None
    b = _mx(bias) if bias is not None else None
    dummy = mx.array(np.zeros(1, np.float32))
    return (mx.fast.layer_norm(a, w, b, eps), dummy, dummy)

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
    if scale is not None:
        kw["scale"] = scale
    if attn_mask is not None:
        kw["mask"] = _mx(attn_mask)
    elif is_causal:
        kw["mask"] = "causal"
    return mx.fast.scaled_dot_product_attention(q, k, v, **kw)

# ---------------- lowering tables ----------------

# torch-level function objects -> lowering. Built lazily below so missing
# attributes on any given torch build can't break the import.
_FUNC_SPECS = {
    "matmul": mx.matmul, "bmm": mx.matmul, "mm": mx.matmul,
    "addmm": lambda b, m1, m2, beta=1, alpha=1: beta * _mx(b) + alpha * mx.matmul(m1, m2),
    "baddbmm": lambda b, m1, m2, beta=1, alpha=1: beta * _mx(b) + alpha * mx.matmul(m1, m2),
    "add": lambda a, b, *k: _mx(a) + _mx(b),
    "sub": lambda a, b, *k: _mx(a) - _mx(b),
    "mul": lambda a, b, *k: _mx(a) * _mx(b),
    "div": lambda a, b, *k: _mx(a) / _mx(b),
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
    "embedding": lambda idx, w, *a_, **k_: _mx(w)[_mx(idx).astype(mx.int32)],
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
        ("torch.nn.functional.embedding", lambda idx, w, *a_, **k_: _mx(w)[_mx(idx).astype(mx.int32)]),
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
        return None

    return run, stats


@torch._dynamo.register_backend(name="mlx_vulkan")
def mlx_vulkan_backend(gm: torch.fx.GraphModule, example_inputs):
    run, stats = make_executor(gm)
    run._mlx_stats = stats
    return run
