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

import dataclasses
import importlib
import logging
import math
import operator
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import torch._dynamo

log = logging.getLogger("mlx_vulkan")
_DBG_ENV = None

# Cross-process cache: when the user opts in (MLX_VK_DISK_CACHE=1 -> dynamo
# caching_precompile + save_cache_artifacts at exit), dynamo stores each frame's
# CompilePackage under <inductor-cache>/dynamo keyed on the wrapped fn. A package
# is only loadable if every __compiled_fn_<id> it references has a registered
# BackendCacheArtifact - we serialize the traced GraphModule (all our executors
# need) and rebuild the backend callable in after_deserialization, so a warm
# process skips the whole dynamo trace AND our lowering walk before first call.
try:
    from torch._dynamo.precompile_context import (
        BackendCacheArtifact as _BCA,
        PrecompileContext as _PCtx,
    )
except Exception:  # older torch without the precompile API
    _BCA = None
    _PCtx = None

if _BCA is not None:

    @dataclasses.dataclass
    class _MlxVkArtifact(_BCA):
        def __getstate__(self):
            # torch re-binds this artifact's backend as
            # module.__dict__[backend_id] for the frame's recorded
            # python_module - which comes from fn.__module__ and is spoofed by
            # @functools.wraps wrappers (HF deprecate_kwarg): the compiled code
            # then runs under the *wrapper* module's globals and LOAD_GLOBAL
            # misses. At save time the name is bound in the real dicts, so
            # snapshot those module names and restore them at load.
            mods = [
                m.__name__
                for m in list(sys.modules.values())
                if isinstance(getattr(m, "__dict__", None), dict)
                and self.key in m.__dict__
            ]
            # The same spoofed __module__ makes torch install() plant the
            # entry's other dynamo globals in the wrong dict; snapshot the
            # module-valued ones (__import_* aliases, ___module_* handles)
            # from the dicts that really ran the code.
            imports = {}
            for m in list(sys.modules.values()):
                d = getattr(m, "__dict__", None)
                if isinstance(d, dict) and self.key in d:
                    for k, v in d.items():
                        if isinstance(v, types.ModuleType) and (
                                k.startswith("__import_")
                                or k.startswith("___module_")):
                            imports[k] = v.__name__
            return {"key": self.key, "content": self.content, "mods": mods,
                    "imports": imports}

        def __setstate__(self, state):
            self.key = state["key"]
            self.content = state["content"]
            self._mods = state.get("mods", [])
            self._imports = state.get("imports", {})

        def after_deserialization(self):
            backend = _restore_backend(self.content)
            for name in getattr(self, "_mods", ()):
                mod = sys.modules.get(name)
                if mod is not None:
                    mod.__dict__[self.key] = backend
                    for alias, modname in self._imports.items():
                        if alias not in mod.__dict__:
                            mod.__dict__[alias] = importlib.import_module(modname)
            return backend


def _restore_backend(gm):
    return mlx_vulkan_backend(gm, [])


def _record_artifact(gm):
    if _PCtx is None:
        return
    bid = getattr(gm, "meta", {}).get("backend_id")
    if not bid:
        return
    try:
        _PCtx.record_artifact(_MlxVkArtifact(key=str(bid), content=gm))
    except Exception as e:
        log.debug("precompile artifact record failed: %s", e)


if os.environ.get("MLX_VK_DISK_CACHE") and _PCtx is not None:
    torch._dynamo.config.caching_precompile = True

    import atexit

    def _save_artifacts():
        try:
            torch.compiler.save_cache_artifacts()
        except Exception as e:
            log.debug("cache artifact save failed: %s", e)

    atexit.register(_save_artifacts)

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
        if v.dtype == mx.bfloat16:
            # numpy has no bf16 buffer descriptor: round the bits through uint16
            return torch.from_numpy(np.array(mx.view(v, mx.uint16))).view(torch.bfloat16)
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

def _expand(a, *s, **k):
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

def _tup(v, n):
    return tuple(v) if isinstance(v, (list, tuple)) else (v,) * n


def _nhwc(a):
    return _mx(a).transpose(0, *range(2, a.ndim), 1)


def _nchw(a):
    return a.transpose(0, a.ndim - 1, *range(1, a.ndim - 1))


def _nll_loss_forward(a, target, weight=None, reduction=1, ignore_index=-100):
    if weight is not None:
        return _raise("nll_loss weight")
    a, t = _mx(a), _mx(target).astype(mx.int32)
    n = a.shape[0]
    lp = a[mx.arange(n), t]
    m = t != ignore_index
    tw = mx.sum(m.astype(mx.float32))
    if reduction == 0:
        return (mx.where(m, -lp, 0.0), tw)
    return (-mx.sum(mx.where(m, lp, 0.0)) / (tw if reduction == 1 else 1.0), tw)


def _nll_loss_backward(gout, a, target, weight, reduction, ignore_index, tw):
    if weight is not None:
        return _raise("nll_loss_backward weight")
    a, t = _mx(a), _mx(target).astype(mx.int32)
    n = a.shape[0]
    m = (t != ignore_index).astype(a.dtype)
    if reduction == 1:
        g = -_mx(gout) * m / mx.maximum(_mx(tw).astype(a.dtype), 1e-12)
    else:
        g = -_mx(gout).reshape(-1)[:n] * m
    onehot = mx.equal(t.reshape(n, 1), mx.arange(a.shape[1]).reshape(1, -1))
    return mx.where(onehot, g.reshape(n, 1), mx.zeros_like(a))


def _convolution(x, w, b=None, stride=1, padding=0, dilation=1,
                 transposed=False, output_padding=0, groups=1):
    # aten.convolution: torch layouts NCHW / OIHW; mx weight (O,H,W,I), x NHWC.
    if transposed:
        return _raise("transposed convolution")
    if isinstance(output_padding, (list, tuple)) and any(output_padding):
        return _raise("convolution output_padding")
    nd = x.ndim - 2
    y = mx.conv_general(_nhwc(x), _nhwc(w), stride=_tup(stride, nd),
                        padding=_tup(padding, nd),
                        kernel_dilation=_tup(dilation, nd), groups=groups)
    if b is not None:
        y = y + _mx(b).reshape((1,) * (y.ndim - 1) + (-1,))
    return _nchw(y)


def _convolution_backward(gout, x, w, bias_sizes, stride, padding, dilation,
                          transposed, output_padding, groups, output_mask):
    # grad_input: transposed conv = conv_general(input_dilation=stride, flip=True)
    # with padding d(k-1)-p, plus output_padding folded into the hi pad.
    # grad_weight: batch->groups trick - conv(x^T [batch=Ci, ch=N],
    # gout^T [out=Co, in=N, kernel=(Ho,Wo)], stride=d, kernel_dilation=s),
    # then slice to (kh,kw). Verified vs torch to ~1e-6 (stride 2, pad 1).
    if transposed or groups != 1:
        return _raise("convolution_backward transposed/groups")
    if isinstance(output_padding, (list, tuple)) and any(output_padding):
        return _raise("convolution_backward output_padding")
    nd = x.ndim - 2
    st, pd, dl = _tup(stride, nd), _tup(padding, nd), _tup(dilation, nd)
    gm = _nhwc(gout)
    mask = output_mask if output_mask is not None else (True, True, True)
    gi = gw = gb = None
    if mask[0]:
        wt_t = _mx(w).transpose(1, *range(2, w.ndim), 0)     # (Ci,K...,Co)
        pe = [dl[i] * (w.shape[2 + i] - 1) - pd[i] for i in range(nd)]
        if min(pe) < 0:
            return _raise("convolution_backward neg pad")
        op = [x.shape[2 + i] - ((gout.shape[2 + i] - 1) * st[i]
                                + dl[i] * (w.shape[2 + i] - 1) - 2 * pd[i] + 1)
              for i in range(nd)]
        if min(op) < 0:
            return _raise("convolution_backward neg op")
        gi = _nchw(mx.conv_general(gm, wt_t, input_dilation=st,
                                   kernel_dilation=dl,
                                   padding=(pe, [pe[i] + op[i] for i in range(nd)]),
                                   flip=True))
    if mask[1]:
        xt = _mx(x).transpose(1, *range(2, x.ndim), 0)       # (Ci,S...,N)
        gt = _mx(gout).transpose(1, *range(2, gout.ndim), 0)  # (Co,S'...,N)
        c = mx.conv_general(xt, gt, stride=dl, kernel_dilation=st, padding=pd)
        c = c[tuple([slice(None)] + [slice(0, w.shape[2 + i]) for i in range(nd)]
                    + [slice(None)])]
        gw = c.transpose(c.ndim - 1, 0, *range(1, c.ndim - 1))  # (Co,Ci,K...)
    if mask[2] and bias_sizes:
        gb = mx.sum(gm, axis=(0,) + tuple(range(1, gm.ndim - 1)))
    return (gi, gw, gb)


def _conv1d(a, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    # torch: a NCL, w OIK ; mlx.conv1d: x NLC, w (K, I/groups, O)
    if groups != 1:
        return _raise("grouped conv1d")
    x = _mx(a).transpose(0, 2, 1)
    wt = _mx(w).transpose(0, 2, 1)  # torch O,I,K -> mlx O,K,I
    st = stride[0] if isinstance(stride, (list, tuple)) else stride
    p = padding[0] if isinstance(padding, (list, tuple)) else padding
    dl = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
    y = mx.conv1d(x, wt, stride=st, padding=p, dilation=dl)
    if b is not None:
        y = y + _mx(b).reshape(1, 1, -1)
    return y.transpose(0, 2, 1)

def _diff(a, n=1, dim=-1, prepend=None, append=None):
    x = _mx(a)
    d = dim % x.ndim
    if prepend is not None:
        x = mx.concatenate([_mx(prepend), x], axis=d)
    if append is not None:
        x = mx.concatenate([x, _mx(append)], axis=d)

    def _s(t, lo, hi):
        idx = [slice(None)] * t.ndim
        idx[d] = slice(lo, hi)
        return t[tuple(idx)]

    for _ in range(n):
        x = _s(x, 1, x.shape[d]) - _s(x, 0, x.shape[d] - 1)
    return x

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


def _topk(a, k, dim=-1, largest=True, sorted=True):
    """torch.topk parity: returns (values, indices) sorted along dim."""
    a = _mx(a)
    dim = (dim if dim is not None else -1) % a.ndim
    perm = [i for i in range(a.ndim) if i != dim] + [dim]
    at = mx.transpose(a, axes=perm)
    idx = mx.argsort(-at if largest else at, axis=-1)[..., :k]
    vals = mx.take_along_axis(at, idx, axis=-1)
    inv = [perm.index(i) for i in range(a.ndim)]
    return (mx.transpose(vals, axes=inv), mx.transpose(idx, axes=inv).astype(mx.int64))

def _multinomial(probs, num_samples, replacement=False, **kw):
    """torch.multinomial parity: draws from probability rows along the last dim.
    replacement=False is top-k of Gumbel-perturbed log-probs (exact);
    replacement=True is independent categorical draws on the logits."""
    p = _mx(probs)
    lp = mx.log(p)
    if replacement:
        return mx.random.categorical(lp, num_samples=num_samples).astype(mx.int64)
    return mx.argsort(-(lp + mx.random.gumbel(p.shape)), axis=-1)[..., :num_samples].astype(mx.int64)

def _bitwise_or_logical(a, b, bitop, logop):
    a, b = _mx(a), _mx(b)
    _float_dts = (mx.float32, mx.float16, mx.bfloat16)
    if a.dtype in _float_dts or b.dtype in _float_dts:
        _b = getattr(mx, "bool_", None) or mx.uint8
        return logop(a.astype(_b), b.astype(_b))
    return bitop(a, b)


# ---------------- pooling (PT-P) ----------------
# Pure-mx lowerings: fwd via strided window-stack, bwd via zero-insert
# (fractional stride) + per-offset shift/accumulate. Static shapes only.

def _pool_hw(v, default=None):
    if v is None or (isinstance(v, (list, tuple)) and len(v) == 0):
        return default
    t = tuple(v) if isinstance(v, (list, tuple)) else (v, v)
    return (int(t[0]), int(t[1]))


def _pool_out(in_sz, k, s, p, dil, ceil_mode):
    o = (in_sz + 2 * p - dil * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
    if (o - 1) * s >= in_sz + p:
        o -= 1
    return max(int(o), 0)


def _pool_windows(xp, oh, ow, kh, kw, sh, sw, dh, dw, pad_value=0.0):
    # (N,C,Hp,Wp) -> (N,C,oh,ow,K), K = kh*kw row-major window offsets.
    # ceil_mode lets the last window start inside the pad and overrun the
    # padded extent; grow the array on the bottom/right so those positions
    # read pad_value instead of slicing ragged.
    need_h = (oh - 1) * sh + (kh - 1) * dh + 1
    need_w = (ow - 1) * sw + (kw - 1) * dw + 1
    if need_h > xp.shape[2] or need_w > xp.shape[3]:
        xp = mx.pad(xp, ((0, 0), (0, 0),
                         (0, max(need_h - xp.shape[2], 0)),
                         (0, max(need_w - xp.shape[3], 0))),
                    constant_values=pad_value)
    cols = [xp[:, :, i * dh: i * dh + (oh - 1) * sh + 1: sh,
               j * dw: j * dw + (ow - 1) * sw + 1: sw]
            for i in range(kh) for j in range(kw)]
    return mx.stack(cols, axis=-1)


def _unstride(g, oh, ow, sh, sw):
    # (N,C,oh,ow) -> (N,C,(oh-1)*sh+1,(ow-1)*sw+1): values at strided spots,
    # zeros between (insert s-1 zeros after each element along each axis)
    n, c = g.shape[0], g.shape[1]
    g = mx.pad(mx.reshape(g, (n, c, oh, ow, 1)),
               ((0, 0), (0, 0), (0, 0), (0, 0), (0, sw - 1)))
    g = mx.reshape(g, (n, c, oh, ow * sw))
    g = mx.pad(mx.reshape(g, (n, c, oh, 1, ow * sw)),
               ((0, 0), (0, 0), (0, 0), (0, sh - 1), (0, 0)))
    g = mx.reshape(g, (n, c, oh * sh, ow * sw))
    return g[:, :, : (oh - 1) * sh + 1, : (ow - 1) * sw + 1]


def _pool_scatter(contrib, hp, wp, kh, kw, sh, sw, dh, dw):
    # contrib (N,C,oh,ow,K) -> (N,C,hp,wp): each offset's field shifted into
    # its window position and summed (handles overlap stride < kernel)
    n, c, oh, ow = contrib.shape[0], contrib.shape[1], contrib.shape[2], contrib.shape[3]
    acc = mx.zeros((n, c, hp, wp), dtype=contrib.dtype)
    for i in range(kh):
        for j in range(kw):
            g = _unstride(contrib[..., i * kw + j], oh, ow, sh, sw)
            g = mx.pad(g, ((0, 0), (0, 0), (i * dh, hp), (j * dw, wp)))
            acc = acc + g[:, :, :hp, :wp]
    return acc


def _pool_shape(a, kernel_size, stride, padding, dilation, ceil_mode):
    kh, kw = _pool_hw(kernel_size)
    sh, sw = _pool_hw(stride, (kh, kw))
    ph, pw = _pool_hw(padding, (0, 0))
    dh, dw = _pool_hw(dilation, (1, 1))
    n, c, h, w = a.shape
    oh = _pool_out(h, kh, sh, ph, dh, ceil_mode)
    ow = _pool_out(w, kw, sw, pw, dw, ceil_mode)
    return (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw)


def _max_pool2d(a, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False):
    x = _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw) = \
        _pool_shape(x, kernel_size, stride, padding, dilation, ceil_mode)
    xp = mx.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-mx.inf)
    win = _pool_windows(xp, oh, ow, kh, kw, sh, sw, dh, dw,
                        pad_value=-mx.inf)
    out = win.max(axis=-1)
    if not return_indices:
        return out
    idx = win.argmax(axis=-1)
    ii, jj = idx // kw, idx % kw
    hi = mx.arange(oh).reshape(1, 1, oh, 1) * sh + ii * dh - ph
    wi = mx.arange(ow).reshape(1, 1, 1, ow) * sw + jj * dw - pw
    return out, (hi * w + wi).astype(mx.int64)


def _max_pool2d_bwd(grad_output, a, kernel_size, stride=None, padding=0,
                    dilation=1, ceil_mode=False, indices=None):
    g, x = _mx(grad_output), _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw) = \
        _pool_shape(x, kernel_size, stride, padding, dilation, ceil_mode)
    xp = mx.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-mx.inf)
    win = _pool_windows(xp, oh, ow, kh, kw, sh, sw, dh, dw,
                        pad_value=-mx.inf)
    idx = win.argmax(axis=-1)
    mask = mx.equal(idx[..., None], mx.arange(kh * kw)).astype(g.dtype)
    contrib = g[..., None] * mask
    acc = _pool_scatter(contrib, h + 2 * ph, w + 2 * pw, kh, kw, sh, sw, dh, dw)
    return acc[:, :, ph:ph + h, pw:pw + w]


def _avg_pool2d(a, kernel_size, stride=None, padding=0, ceil_mode=False,
                count_include_pad=True, divisor_override=None):
    x = _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw) = \
        _pool_shape(x, kernel_size, stride, padding, (1, 1), ceil_mode)
    xp = mx.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    s = _pool_windows(xp, oh, ow, kh, kw, sh, sw, 1, 1).sum(axis=-1)
    if divisor_override is not None:
        return s / float(divisor_override)
    # Divisor = window positions inside the padded input; ceil overrun
    # never counts. include_pad counts the p-pad region itself.
    ones = mx.ones((n, c, h, w), dtype=x.dtype)
    cnt = _pool_windows(
        mx.pad(ones, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
               constant_values=1.0 if count_include_pad else 0.0),
        oh, ow, kh, kw, sh, sw, 1, 1).sum(axis=-1)
    return s / cnt


def _avg_pool2d_backward(grad_output, a, kernel_size, stride=None, padding=0,
                         ceil_mode=False, count_include_pad=True,
                         divisor_override=None):
    g, x = _mx(grad_output), _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, _, _) = \
        _pool_shape(x, kernel_size, stride, padding, (1, 1), ceil_mode)
    if divisor_override is not None:
        div = float(divisor_override)
    else:
        ones = mx.ones((n, c, h, w), dtype=x.dtype)
        div = _pool_windows(
            mx.pad(ones, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                   constant_values=1.0 if count_include_pad else 0.0),
            oh, ow, kh, kw, sh, sw, 1, 1).sum(axis=-1)
    contrib = mx.broadcast_to((g / div)[..., None],
                              (n, c, oh, ow, kh * kw))
    acc = _pool_scatter(contrib, h + 2 * ph, w + 2 * pw, kh, kw, sh, sw, 1, 1)
    return acc[:, :, ph:ph + h, pw:pw + w]


def _adaptive_avg_pool2d(a, output_size):
    x = _mx(a)
    n, c, h, w = x.shape
    if output_size is None:
        output_size = (h, w)
    elif isinstance(output_size, (list, tuple)):
        output_size = tuple(h if s is None else s for s in output_size[:1]) + \
            tuple(w if s is None else s for s in output_size[1:2])
    oh, ow = _pool_hw(output_size)
    if oh == h and ow == w:
        return x
    rows = []
    for i in range(oh):
        hs = i * h // oh
        he = (i + 1) * h // oh + (1 if (i + 1) * h % oh else 0)  # ceil
        cols = []
        for j in range(ow):
            ws = j * w // ow
            we = (j + 1) * w // ow + (1 if (j + 1) * w % ow else 0)
            cols.append(x[:, :, hs:he, ws:we].mean(axis=(-2, -1)))
        rows.append(mx.stack(cols, axis=-1))
    return mx.stack(rows, axis=-2)


def _adaptive_avg_pool2d_backward(grad_output, a):
    g, x = _mx(grad_output), _mx(a)
    n, c, h, w = x.shape
    oh, ow = g.shape[2], g.shape[3]
    if oh == h and ow == w:
        return g
    acc = mx.zeros((n, c, h, w), dtype=g.dtype)
    for i in range(oh):
        hs = i * h // oh
        he = (i + 1) * h // oh + (1 if (i + 1) * h % oh else 0)
        for j in range(ow):
            ws = j * w // ow
            we = (j + 1) * w // ow + (1 if (j + 1) * w % ow else 0)
            cnt = float((he - hs) * (we - ws))
            mask = mx.pad(mx.ones((1, 1, he - hs, we - ws), dtype=g.dtype),
                          ((0, 0), (0, 0), (hs, h - he), (ws, w - we)))
            acc = acc + g[:, :, i, j][:, :, None, None] * (mask / cnt)
    return acc


def _bn3(x, w, b, rm, rv, training, momentum, eps):
    """batch-norm forward, channel dim 1 (N,C,... layout). Training
    normalizes with the biased batch variance (torch semantics); eval
    reads the running stats. Returns (out, save_mean, save_invstd)."""
    x = _mx(x)
    bc = [1, -1] + [1] * (x.ndim - 2)
    if training:
        axes = tuple(d for d in range(x.ndim) if d != 1)
        mean = mx.mean(x, axis=axes, keepdims=True)
        var = mx.var(x, axis=axes, keepdims=True)
    else:
        mean = mx.reshape(_mx(rm), bc)
        var = mx.reshape(_mx(rv), bc)
    invstd = mx.rsqrt(var + eps)
    y = (x - mean) * invstd
    if w is not None:
        y = y * mx.reshape(_mx(w), bc)
    if b is not None:
        y = y + mx.reshape(_mx(b), bc)
    squeeze = lambda t: mx.reshape(t, (-1,)) if t.ndim > 1 else t
    return y, squeeze(mean), squeeze(invstd)

def _bn_legit_functional(x, w, b, rm, rv, training, momentum, eps):
    """aten._native_batch_norm_legit_functional (aot train path): same core as
    _bn3 but returns updated running stats: (out, save_mean, save_invstd,
    new_running_mean, new_running_var). Torch updates running_var with the
    UNBIASED batch var while normalizing with the biased one."""
    x = _mx(x)
    out, sm, si = _bn3(x, w, b, rm, rv, training, momentum, eps)
    rmx, rvx = _mx(rm), _mx(rv)
    if training:
        axes = tuple(d for d in range(x.ndim) if d != 1)
        n = x.size / x.shape[1]
        var_b = mx.var(x, axis=axes)
        unbias = var_b * (n / (n - 1)) if n > 1 else var_b
        new_rm = (1.0 - momentum) * rmx + momentum * sm
        new_rv = (1.0 - momentum) * rvx + momentum * unbias
    else:
        new_rm, new_rv = rmx, rvx
    return out, sm, si, new_rm, new_rv


def _select_backward(g, sizes, dim, index):
    """aten.select_backward: grad scattered into zeros shaped like the input."""
    g = _mx(g)
    sizes = [int(s) for s in sizes]
    dim = int(dim) % len(sizes)
    shape = [1] * len(sizes)
    shape[dim] = sizes[dim]
    mask = (mx.arange(sizes[dim]) == int(index)).astype(g.dtype).reshape(shape)
    return mx.expand_dims(g, dim) * mask


def _bn_backward(dy, x, w, rm, rv, save_mean, save_invstd, training, eps, out_mask):
    """aten.native_batch_norm_backward: (dx, dw, db) honoring output_mask."""
    dy, x = _mx(dy), _mx(x)
    axes = tuple(d for d in range(x.ndim) if d != 1)
    bc = [1, -1] + [1] * (x.ndim - 2)
    if training:
        mean = mx.reshape(_mx(save_mean), bc)
        invstd = mx.reshape(_mx(save_invstd), bc)
    else:
        mean = mx.reshape(_mx(rm), bc)
        invstd = mx.rsqrt(mx.reshape(_mx(rv), bc) + eps)
    wb = mx.reshape(_mx(w), bc) if w is not None else 1.0
    xh = (x - mean) * invstd
    if training:
        dx = (dy - mx.mean(dy, axis=axes, keepdims=True)
              - xh * mx.mean(dy * xh, axis=axes, keepdims=True)) * invstd * wb
    else:
        dx = dy * invstd * wb
    dw = mx.sum(dy * xh, axis=axes) if (w is not None and out_mask[1]) else None
    db = mx.sum(dy, axis=axes) if out_mask[2] else None
    return (dx if out_mask[0] else None), dw, db

# ---------------- lowering tables ----------------

# torch-level function objects -> lowering. Built lazily below so missing
# attributes on any given torch build can't break the import.
def _coerce_index(v):
    # fx graphs under dynamic shapes carry torch.SymInt/SymBool indices;
    # mx slicing needs concrete python scalars. Coerce recursively through
    # slices and tuples; tensors/arrays pass through for advanced indexing.
    if isinstance(v, torch.SymInt):
        return int(v)
    if isinstance(v, torch.SymBool):
        return bool(v)
    if isinstance(v, torch.SymFloat):
        return float(v)
    if isinstance(v, slice):
        return slice(_coerce_index(v.start), _coerce_index(v.stop),
                     _coerce_index(v.step))
    if isinstance(v, (tuple, list)):
        return type(v)(_coerce_index(x) for x in v)
    return v


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
    "t": lambda a: mx.transpose(a), "permute": lambda a, *d: mx.transpose(a, axes=tuple(d[0]) if len(d) == 1 else tuple(d)),
    "reshape": _reshape, "view": _reshape, "_unsafe_view": _reshape,
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
    "select": lambda a, dim, i: _slice(a, dim, i % a.shape[dim % a.ndim], i % a.shape[dim % a.ndim] + 1, 1)[tuple(0 if d == dim % a.ndim else slice(None) for d in range(a.ndim))],
    "slice": _slice, "narrow": lambda a, dim, s, n: _slice(a, dim, s, s + n, 1),
    "mean": _mean, "sum": _sum,
    "max": lambda a, d=None, keepdim=False, *k: (mx.max(a, axis=d, keepdims=keepdim), mx.argmax(a, axis=d, keepdims=keepdim)) if d is not None else mx.max(a),
    "min": lambda a, d=None, keepdim=False, *k: (mx.min(a, axis=d, keepdims=keepdim), mx.argmin(a, axis=d, keepdims=keepdim)) if d is not None else mx.min(a),
    "where": lambda c, a, b: mx.where(_mx(c), _mx(a), _mx(b)),
    "zeros": lambda *a, **k: mx.zeros(_shape_of(a, k), dtype=_MX_DT.get(k.get("dtype"))),
    "zeros_like": lambda a, **k: mx.zeros(a.shape, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "ones": lambda *a, **k: mx.ones(_shape_of(a, k), dtype=_MX_DT.get(k.get("dtype"))),
    "ones_like": lambda a, **k: mx.ones(a.shape, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "full": lambda s, v, **k: mx.full(tuple(s), v, dtype=_MX_DT.get(k.get("dtype"))),
    "full_like": lambda a, v, **k: mx.full(a.shape, v, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "tril": lambda a, d=0: mx.tril(a, d) if hasattr(mx, "tril") else _raise("tril"),
    "getitem": lambda t, i: t[_coerce_index(i)],
    "detach": lambda a: a, "clone": lambda a: a, "contiguous": lambda a: a,
    "alias": lambda a: a, "lift_fresh": lambda a: a, "lift_fresh_copy": lambda a: a, "to": lambda a, *a_, **k_: a,
    "_to_copy": lambda a, **k_: a, "size": lambda a, d=None: a.shape[d] if d is not None else a.shape,
    "native_layer_norm": _ln_native,
    "dropout": lambda a, p=0.5, training=False, *k: a,
    # aten.batch_norm order: (x, weight, bias, running_mean, running_var, training, momentum, eps, cudnn)
    "batch_norm": lambda x, w, b, rm, rv, training, momentum, eps, *k: _bn3(x, w, b, rm, rv, training, momentum, eps)[0],
    "native_batch_norm": lambda x, w, b, rm, rv, training, momentum, eps: _bn3(x, w, b, rm, rv, training, momentum, eps),
    "_native_batch_norm_legit_functional": _bn_legit_functional,
    "native_batch_norm_backward": _bn_backward,
    "conv2d": _conv2d,
    "max_pool2d": lambda a, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False: _max_pool2d(a, kernel_size, stride, padding, dilation, ceil_mode, return_indices),
    "max_pool2d_with_indices": lambda a, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: _max_pool2d(a, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True),
    "max_pool2d_with_indices_backward": _max_pool2d_bwd,
    "avg_pool2d": _avg_pool2d,
    "avg_pool2d_backward": _avg_pool2d_backward,
    "adaptive_avg_pool2d": _adaptive_avg_pool2d,
    "_adaptive_avg_pool2d": _adaptive_avg_pool2d,
    "adaptive_avg_pool2d_backward": _adaptive_avg_pool2d_backward,
    "_adaptive_avg_pool2d_backward": _adaptive_avg_pool2d_backward,
    "conv1d": _conv1d,
    "diff": _diff,
    "convolution": _convolution,
    "convolution_backward": _convolution_backward,
    "t": _wt,
    "threshold_backward": lambda g, x, t=0.0: mx.where(mx.greater(_mx(x), t), _mx(g), mx.zeros_like(_mx(g))),
    "select_backward": _select_backward,
    "tanh_backward": lambda g, out: _mx(g) * (1.0 - _mx(out) * _mx(out)),
    "nll_loss_forward": _nll_loss_forward,
    "nll_loss_backward": _nll_loss_backward,
    "_log_softmax": lambda a, dim, half_to_float=False: a - mx.log(mx.sum(mx.exp(a), axis=dim, keepdims=True)),
    "_log_softmax_backward_data": lambda go, out, dim, dt=None: _mx(go) - mx.exp(_mx(out)) * mx.sum(_mx(go), axis=dim, keepdims=True),
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
    "not_equal": lambda a, b: mx.not_equal(_mx(a), _mx(b)),
    "gt": lambda a, b: mx.greater(_mx(a), _mx(b)),
    "lt": lambda a, b: mx.less(_mx(a), _mx(b)),
    "cos": mx.cos, "sin": mx.sin, "log": mx.log, "log1p": mx.log1p,
    "outer": mx.outer, "einsum": lambda *a: mx.einsum(*a),
    "argmax": lambda a, dim=None, keepdim=False: mx.argmax(a, axis=dim, keepdims=keepdim),
    "argmin": lambda a, dim=None, keepdim=False: mx.argmin(a, axis=dim, keepdims=keepdim),
    "topk": _topk,
    "multinomial": _multinomial,
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
    "bitwise_and": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_and, mx.logical_and),
    "bitwise_or": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_or, mx.logical_or),
    "bitwise_xor": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_xor, mx.bitwise_xor),
    "and_": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_and, mx.logical_and),
    "or_": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_or, mx.logical_or),
    "xor": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_xor, mx.bitwise_xor),
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
    "embedding": lambda w, idx, *a_, **k_: _mx(w)[_mx(idx).astype(mx.int32)],
    "all": lambda a, dim=None, keepdim=False: mx.all(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "any": lambda a, dim=None, keepdim=False: mx.any(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "count_nonzero": lambda a, dim=None: mx.sum((a != 0).astype(mx.int32), axis=dim),
    "empty": lambda *a, **k: mx.zeros(_shape_of(a, k), dtype=_MX_DT.get(k.get("dtype"))),
    "empty_like": lambda a, **k: mx.zeros(a.shape, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_zeros": lambda a, s, **k: mx.zeros(tuple(s), dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_ones": lambda a, s, **k: mx.ones(tuple(s), dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_empty": lambda a, s, **k: mx.zeros(tuple(s), dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_full": lambda a, s, v, **k: mx.full(tuple(s), v, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "scalar_tensor": lambda v, **k: mx.array(v, _MX_DT.get(k.get("dtype"), mx.int64 if type(v) is int else None)),
    "tensor": lambda d, **k: _mx(d) if isinstance(d, torch.Tensor) else mx.array(np.array(_torch(d)) if isinstance(d, (list, tuple)) else d),
    "as_tensor": lambda d, **k: _mx(d),
    "einsum_fn": mx.einsum,
}

def _raise(name):
    raise NotImplementedError(name)

FUNC_TABLE = {}
_FN_SOURCES = [torch, F, torch.special, operator]

def _resolve(dotted):
    parts = dotted.split(".")
    obj = globals()[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj

def _build_func_table():
    candidates = []
    for fn_name, lowering in _FUNC_SPECS.items():
        if fn_name in ("embedding", "batch_norm"):
            continue  # specs are aten order; python-order names map explicitly below
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
        ("torch.nn.functional.batch_norm", lambda x, rm, rv, w=None, b=None, training=False, momentum=0.1, eps=1e-5: _bn3(x, w, b, rm, rv, training, momentum, eps)[0]),
        ("torch.batch_norm", _FUNC_SPECS["batch_norm"]),
        ("torch.nn.functional.embedding", lambda idx, w, *a_, **k_: _mx(w)[_mx(idx).astype(mx.int32)]),
        ("torch.embedding", lambda idx, w, *a_, **k_: _mx(w)[_mx(idx).astype(mx.int32)]),
        ("operator.getitem", lambda t, i: t[i]),
        ("operator.matmul", mx.matmul),
        ("operator.add", _FUNC_SPECS["add"]),
        ("operator.sub", _FUNC_SPECS["sub"]),
        ("operator.mul", _FUNC_SPECS["mul"]),
        ("operator.truediv", _FUNC_SPECS["div"]),
        ("operator.floordiv", _FUNC_SPECS["div"]),
        ("operator.neg", _FUNC_SPECS["neg"]),
        ("operator.and_", _FUNC_SPECS["and_"]),
        ("operator.or_", _FUNC_SPECS["or_"]),
        ("operator.xor", _FUNC_SPECS["xor"]),
        ("operator.invert", _FUNC_SPECS["invert"]),
    ]
    for dotted, lowering in explicit:
        try:
            candidates.append((_resolve(dotted), lowering))
        except AttributeError:
            pass
    for fn, lowering in candidates:
        FUNC_TABLE[fn] = lowering

_build_func_table()
# dotted-name resolve only covers torch attrs; operator.* objects register directly
FUNC_TABLE[operator.iadd] = _FUNC_SPECS["add"]

# name-keyed fallback: dynamo may reference e.g. torch._C._VariableFunctions.matmul
# which is a different object than torch.matmul
FUNC_TABLE_BY_NAME = {}
for _fn, _low in FUNC_TABLE.items():
    FUNC_TABLE_BY_NAME.setdefault(getattr(_fn, "__name__", ""), _low)

# ops that produce arrays from scalars/shapes — lower even with no mx inputs
_PRODUCERS = {"arange", "zeros", "ones", "full", "empty", "eye", "linspace",
              "tensor", "zeros_like", "ones_like", "full_like", "empty_like",
              "rand", "randn", "randint", "randperm", "new_zeros", "new_ones",
              "new_empty", "new_full", "scalar_tensor"}

# ops that must never CPU-fallback: lower even when upstream nodes produced
# torch tensors (e.g. pool bwd behind an unlowered conv/bn backward)
_ALWAYS_LOWER = {"max_pool2d", "max_pool2d_with_indices",
                 "max_pool2d_with_indices_backward", "avg_pool2d",
                 "avg_pool2d_backward", "adaptive_avg_pool2d",
                 "_adaptive_avg_pool2d", "adaptive_avg_pool2d_backward",
                 "_adaptive_avg_pool2d_backward", "convolution_backward",
                 "threshold_backward", "native_batch_norm_backward"}

# aten.* OpOverload name table (same lowerings, keyed by opname)
ATEN_TABLE = {name: f for name, f in _FUNC_SPECS.items()}
ATEN_TABLE["_native_batch_norm_legit"] = _FUNC_SPECS["native_batch_norm"]

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
# symbolic-int bookkeeping over dynamic shapes (e.g. HF DynamicCache
# seq-length sums): concrete ints at call time - pure glue, no kernel.
for _sf in ("sym_sum",):
    _f = getattr(torch, _sf, None)
    if _f is not None:
        GLUE_FNS.add(_f)

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
        if name in ("new_empty", "new_zeros", "new_ones", "new_full"):
            dt = _MX_DT.get(kwargs.get("dtype"), obj.dtype)
            if name == "new_full" and len(args) < 2:
                args = (kwargs["size"], args[0] if args else kwargs["fill_value"])
            shape = tuple(args[0] if args else kwargs["size"])
            if name == "new_ones":
                return mx.ones(shape, dtype=dt), True
            if name == "new_full":
                return mx.full(shape, args[1], dtype=dt), True
            return mx.zeros(shape, dtype=dt), True
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
                try:
                    env[node] = args[0][_coerce_index(args[1])]
                    stats["glue"].append("getitem")
                    continue
                except Exception as e:
                    # Mixed-domain getitem (torch source, mlx index): coerce the
                    # indexable side to mx when any operand lives there.
                    if has_mx((args[0], args[1])):
                        try:
                            env[node] = _mx(args[0])[_deep_mx(_coerce_index(args[1]))]
                            stats["glue"].append("getitem")
                            continue
                        except Exception:
                            pass
                    log.info("LOWER-FAIL	getitem	%s", e)

            val = None
            on_mlx = False
            if node.op == "call_function":
                lowering = (FUNC_TABLE.get(node.target)
                            or FUNC_TABLE_BY_NAME.get(getattr(node.target, "__name__", ""))
                            or ATEN_TABLE.get(name))
                is_producer = (getattr(node.target, "__name__", "") in _PRODUCERS
                               or getattr(node.target, "__name__", "") in _ALWAYS_LOWER
                               or name in _PRODUCERS
                               or name in _ALWAYS_LOWER)
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
            if _DBG_ENV is not None:
                _DBG_ENV[node] = val
        return None

    return run, stats


def _eager_run(gm):
    run, stats = make_executor(gm)
    run._mlx_stats = stats
    return run


@torch._dynamo.register_backend(name="mlx_vulkan")
def mlx_vulkan_backend(gm: torch.fx.GraphModule, example_inputs, _dt=None):
    if os.environ.get("MLX_VK_EAGER"):
        return _eager_run(gm)
    _record_artifact(gm)
    try:
        run_c = make_compiled_executor(gm, dt=_dt)
    except Exception as e:
        log.warning("compiled executor setup failed: %s", e)
        return _eager_run(gm)
    holder = {"run": None}

    def guarded(*args):
        if holder["run"] is None:
            try:
                out = run_c(*args)
            except Exception as e:
                log.warning("compiled executor failed (%s); falling back to eager", e)
                holder["run"] = _eager_run(gm)
            else:
                holder["run"] = run_c
                return out
        return holder["run"](*args)

    guarded._mlx_stats = run_c._mlx_stats
    return guarded


# Training variant: AOTAutograd splits fwd/bwd and compiles both subgraphs
# through the same lowering tables, so *_backward aten nodes lower to mx
# (aten.convolution_backward -> transposed conv + batch->groups conv via
# mx.conv_general; pools fwd+bwd). Use for train graphs:
# backend="mlx_vulkan_train". Kept a separate name: aot functionalizes
# input mutations, which the decode state machinery in
# make_compiled_executor does not model.
def _aot_compiler(gm, example_inputs):
    try:
        from torch._functorch._aot_autograd.utils import make_boxed_func
    except ImportError:
        from functorch.compile import make_boxed_func
    # Training defaults to fp32: bf16 grads drift the loss vs CPU. Explicit
    # MV_DT still overrides.
    run = mlx_vulkan_backend(gm, example_inputs,
                             _dt=None if os.environ.get("MV_DT") else mx.float32)

    def unboxed(*args):
        # aot's bw runtime may deliver the arg list nested one level deep
        while len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        return run(*args)

    return make_boxed_func(unboxed)


try:
    from torch._dynamo.backends.common import aot_autograd
    torch._dynamo.register_backend(name="mlx_vulkan_train")(
        aot_autograd(fw_compiler=_aot_compiler,
                     bw_compiler=_aot_compiler,
                     inference_compiler=_aot_compiler))
except Exception as e:
    log.warning("mlx_vulkan_train backend unavailable: %s", e)


# ================= compiled executor (v2) =================
# Whole-graph lazy lowering wrapped in mx.compile. Cache mutations
# (index_copy_/add_ on StaticCache buffers) are modeled functionally: updated
# tensors become extra outputs fed back as inputs next call, so HF's buffers
# only seed state once. RMSNorm chains fuse to mx.fast.rms_norm; the sdpa FX
# node maps to mx.fast.scaled_dot_product_attention; GEMM/sdpa inputs run bf16
# while the residual/elementwise spine stays fp32 -> ids match CPU fp32.
#
# bf16 is the right default on Apple/Honeykrisp (measured ~2x decode on
# Qwen3-0.6B) but not on Intel ANV: there fp32 GEMM is faster (7.5 vs 6.9
# tok/s, B60) AND bf16 activation casts flip ids - a 0.024-margin top-2 tie
# at decode step 5 resolved to the wrong index (PT-D5 bisect, greedy ids vs
# CPU diverge at token 5 under bf16, 32/32 match under fp32). Default fp32 on
# Intel parts; MV_DT=bf16|fp32 forces either arm.

def _default_dt():
    v = os.environ.get("MV_DT")
    if v:
        return {"bf16": mx.bfloat16, "fp32": mx.float32}[v]
    try:
        info = mx.device_info()
        if info.get("vendor_id") == 0x8086 or "intel" in str(info.get("driver", "")).lower():
            return mx.float32
    except Exception:
        pass
    return mx.bfloat16

_DT = _default_dt()

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
    return (getattr(n.target, "__name__", None) or _opname(n)).split(".")[0]


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


def make_compiled_executor(gm, dt=None):
    nodes = list(gm.graph.nodes)
    phs = [n for n in nodes if n.op == "placeholder"]
    skip, fused = _fuse_rmsnorm(nodes)
    fp32_params = {str(w.target) for _, w, _ in fused.values()}
    # embedding gathers feed the whole residual spine; keep their source weights
    # fp32 (a gather is not a GEMM and gains nothing from the bf16 cast).
    _pass = {"to", "detach", "clone", "contiguous", "view"}
    for n in nodes:
        if n.op == "call_function" and _opname(n) == "embedding" and len(n.args) > 1:
            w = n.args[1]
            while getattr(w, "op", None) == "call_method" and _opname(w) in _pass and w.args:
                w = w.args[0]
            if getattr(w, "op", None) == "placeholder":
                fp32_params.add(str(w.target))
    state_names = {str(n.target) for n in phs if "_cache_layers_" in str(n.target)}
    # cache placeholders only count as mutated state when the graph actually
    # mutates them in place; HF concat-style cache (cat -> new outputs) must not
    # replay step-1 state on later calls.
    _mutfns = ("index_copy_", "add_", "sub_", "mul_", "div_", "copy_", "scatter_", "index_put_")
    if not any(
            n.op == "call_method" and _opname(n) in _mutfns
            and getattr(n.args[0], "op", None) == "placeholder"
            and str(n.args[0].target) in state_names
            for n in nodes):
        state_names = set()
    static_names = {str(n.target) for n in phs if str(n.target).startswith("G_")}
    mutated = [n for n in phs if str(n.target) in state_names]
    quant_names = _quantizable_phs(phs) if _Q_BITS else set()
    conv_static = {}
    state = {}
    holder = {"ar": None}

    _dte = dt or _DT

    def _cast(x, dt=_dte):
        return x.astype(dt) if isinstance(x, mx.array) and x.dtype != dt else x

    def build(*flat):
        env = _DBG_ENV if _DBG_ENV is not None else {}
        for n, v in zip(phs, flat):
            env[n] = _QW(*v) if str(n.target) in quant_names and isinstance(v, (tuple, list)) else v

        def res(a):
            if isinstance(a, torch.fx.Node):
                return env[a]
            if isinstance(a, (tuple, list)):
                return type(a)(res(x) for x in a)
            return a

        for n in nodes:
            if n.op == "placeholder" or n in skip:
                continue
            name = (getattr(n.target, "__name__", None) or _opname(n)).split(".")[0]
            if n.op == "output":
                out = res(n.args[0])
                holder["n_out"] = len(out) if isinstance(out, (tuple, list)) else 0
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
                if mask is None and kwargs.get("is_causal"):
                    mask = "causal"
                scale = kwargs.get("scale")
                if scale is None:
                    scale = 1.0 / math.sqrt(q.shape[-1])
                if isinstance(mask, mx.array) and mask.dtype != getattr(mx, "bool_", mx.uint8):
                    mask = _cast(mask)
                env[n] = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask).astype(mx.float32)
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
                env[n] = args[0][_coerce_index(args[1])]
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
        outs = (build if _DBG_ENV is not None else cfn)(*flat)
        outs = outs if isinstance(outs, (tuple, list)) else (outs,)
        mx.eval(*outs)
        t2 = _t.perf_counter()
        # outs = real outputs + len(mutated) state tail. aot graphs have no
        # mutated placeholders, so all outputs are real (and include the
        # saved-for-backward tensors aot needs back). holder["n_out"] marks
        # the boundary when the output node ran multi-output.
        n_out = holder.get("n_out", 1)
        real = outs[:n_out] if n_out else outs[:1]
        extra = outs[n_out:] if n_out else outs[1:]
        for m, arr in zip(mutated, extra):
            state[str(m.target)] = arr
        out = tuple(_torch(x) for x in real)
        t3 = _t.perf_counter()
        if _profile:
            _acc["prep"] += t1 - t0; _acc["cfn"] += t2 - t1; _acc["conv"] += t3 - t2; _acc["n"] += 1
            if _acc["n"] % 10 == 0:
                print(f"MV_PROF n={_acc['n']} prep={_acc['prep']/_acc['n']*1e3:.1f}ms cfn={_acc['cfn']/_acc['n']*1e3:.1f}ms conv={_acc['conv']/_acc['n']*1e3:.1f}ms", flush=True)
        return out

    run._mlx_stats = stats
    return run
