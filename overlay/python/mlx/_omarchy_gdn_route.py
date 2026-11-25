"""Route pip mlx-lm's gated_delta_update to the fused backend primitive.

pip-installed mlx-lm <= 0.31.3 gates its fused path on
``mx.metal.is_available()``. That is False on this backend, so the model
falls into ``gated_delta_ops``: a per-token Python loop over T with an
f32 state (~135k GPU dispatches per 512-token prefill on Qwen3.5-2B -
small f32 elementwise ops bracketed by CastBF16F32/CastF32BF16 pairs).
mlx-lm 94cdcae / PyPI 0.32.0 calls ``mx.fast.gated_delta_update``
natively and needs no shim.

``mlx.core`` imports this module at init. When
``mlx_lm.models.gated_delta`` later loads, a meta_path hook wraps its
``gated_delta_update`` so the inference call reaches the same
``mx.fast.gated_delta_update`` / ``mx.fast.gated_delta_update_raw``
bindings the vendored serve patches (``mlx-lm-gated-delta-fast-route*``,
``mlx-lm-gated-delta-raw``) select. No mlx-lm file is modified.

Kill switch: ``MLX_OMARCHY_NO_GDN_ROUTE=1`` leaves the module unwrapped.
"""

import importlib
import importlib.abc
import importlib.machinery
import inspect
import os
import sys

_TARGET = "mlx_lm.models.gated_delta"


def _disabled():
    return os.environ.get("MLX_OMARCHY_NO_GDN_ROUTE") == "1"


def _wrap(module):
    orig = module.gated_delta_update
    if getattr(orig, "_omarchy_route", None) is not None:
        return
    # The 0.32 API line calls mx.fast.gated_delta_update natively.
    if "mx.fast.gated_delta_update" in inspect.getsource(module):
        return
    import mlx.core as mx

    def routed(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
               use_kernel=True, *rest, **kw):
        routable = (
            use_kernel
            and not rest
            and not kw
            and not _disabled()
            and hasattr(mx.fast, "gated_delta_update")
            and mx.default_device() == mx.gpu)
        if not routable:
            return orig(
                q, k, v, a, b, A_log, dt_bias,
                state=state, mask=mask, use_kernel=use_kernel,
                *rest, **kw)
        # Same routing as the vendored patch trio: prefill at Hk != Hv
        # repeats q/k in Python (the fused kernel needs Hk == Hv);
        # decode uses the raw route, which expands inside the backend.
        Hk, Hv = q.shape[-2], v.shape[-2]
        if Hv != Hk and q.shape[1] > 1:
            q = mx.repeat(q, Hv // Hk, -2)
            k = mx.repeat(k, Hv // Hk, -2)
        if q.shape[1] == 1 and hasattr(mx.fast, "gated_delta_update_raw"):
            return mx.fast.gated_delta_update_raw(
                q, k, v, a, b, A_log, dt_bias, state, mask)
        beta = mx.sigmoid(b)
        g = module.compute_g(A_log, a, dt_bias)
        if state is None:
            state = mx.zeros(
                (q.shape[0], v.shape[-2], v.shape[-1], q.shape[-1]),
                dtype=mx.float32)
        return mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)

    routed._omarchy_route = orig
    module.gated_delta_update = routed


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        if hasattr(self._loader, "create_module"):
            return self._loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._loader.exec_module(module)
        _wrap(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Finder(importlib.abc.MetaPathFinder):
    # Installed ahead of PathFinder: it resolves the real spec, then
    # wraps the loader so the patch lands right after exec_module.
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        if not hasattr(spec.loader, "exec_module"):
            return None
        spec.loader = _Loader(spec.loader)
        return spec


def _install():
    if _disabled():
        return
    loaded = sys.modules.get(_TARGET)
    if loaded is not None:
        _wrap(loaded)
        return
    for finder in sys.meta_path:
        if type(finder) is _Finder:
            return
    sys.meta_path.insert(0, _Finder())


_install()
