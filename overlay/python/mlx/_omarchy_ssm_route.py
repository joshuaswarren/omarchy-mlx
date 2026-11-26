"""Route pip mlx-lm's Mamba-2 ssm_update decode step to the Vulkan ssm_kernel port.

pip-installed mlx-lm <= 0.31.3 gates ``make_ssm_kernel()`` and
``ssm_update()`` on ``mx.metal.is_available()``. That is False on the
Omarchy Vulkan backend, so every Mamba-2 / Nemotron-H decode step
(``seq_len == 1`` with ``state is not None``) falls into ``ssm_attn``,
which executes the full chunked SSD surrogate-attention chain
(segsum/cumsum/tril matmuls) instead of a single state-update dispatch.

``mlx.core`` imports this module at init. When ``mlx_lm.models.ssm`` later
loads, a meta_path hook wraps ``ssm_update`` so decode calls
(``seq_len == 1``, ``state is not None``, ``mask is None``,
``lengths is None``, GPU device) dispatch the exact shipped ``ssm_kernel``
through ``mx.fast.metal_kernel``, hitting the named Vulkan hand port in
``custom_kernel_ssm.h``. Prefill (``seq_len > 1``) and all other cases
remain on ``ssm_attn``. No mlx-lm file is modified.

Kill switch: ``MLX_OMARCHY_NO_SSM_ROUTE=1`` leaves the module unwrapped.
"""

import importlib
import importlib.abc
import importlib.machinery
import inspect
import os
import sys

_TARGET = "mlx_lm.models.ssm"

_SSM_KERNEL_SOURCE = """
        auto n = thread_position_in_grid.z;
        auto h_idx = n % H;
        auto g_idx = n / G;
        constexpr int n_per_t = Ds / 32;

        auto x = X + n * Dh;
        out += n * Dh;
        auto i_state = state_in + n * Dh * Ds;
        auto o_state = state_out + n * Dh * Ds;

        // C and B have shape [batch, group, state_dim]
        // C and B need to be offset by group size
        auto C_ = C + g_idx * Ds;
        auto B_ = B + g_idx * Ds;

        auto ds_idx = thread_position_in_threadgroup.x;
        auto d_idx = thread_position_in_grid.y;

        auto dt_ = static_cast<float>(dt[n]);
        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
        auto dA = fast::exp(A * dt_);

        float acc = 0.0;
        auto x_ = static_cast<float>(x[d_idx]);

        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            auto idx = d_idx * Ds + s_idx;
            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
            auto state = dA * i_state[idx] + dB_by_x;
            o_state[idx] = static_cast<U>(state);
            acc += state * C_[s_idx];
        }
        acc = simd_sum(acc);
        if (thread_index_in_simdgroup == 0) {
            out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
        }
    """

_cached_kernel = None


def _disabled():
    return os.environ.get("MLX_OMARCHY_NO_SSM_ROUTE") == "1"


def _get_kernel(mx):
    global _cached_kernel
    if _cached_kernel is None:
        _cached_kernel = mx.fast.metal_kernel(
            name="ssm_kernel",
            input_names=["X", "A_log", "B", "C", "D", "dt", "state_in"],
            output_names=["out", "state_out"],
            source=_SSM_KERNEL_SOURCE,
        )
    return _cached_kernel


def _wrap(module):
    orig = getattr(module, "ssm_update", None)
    if orig is None or getattr(orig, "_omarchy_route", None) is not None:
        return
    if not hasattr(module, "compute_dt"):
        return
    # Leave sourceless modules (.pyc-only / namespace packages) untouched,
    # matching _omarchy_gdn_route's fail-open import contract.
    try:
        src = inspect.getsource(module)
    except (OSError, TypeError):
        return
    if "mx.metal.is_available()" not in src:
        return
    import mlx.core as mx

    def routed(
        hidden_states,
        A_log,
        B,
        C,
        D,
        dt,
        dt_bias,
        state=None,
        time_step_limit=(0.001, 100.0),
        mask=None,
        lengths=None,
        *rest,
        **kw,
    ):
        routable = (
            not rest
            and not kw
            and not _disabled()
            and state is not None
            and mask is None
            and lengths is None
            and len(hidden_states.shape) == 4
            and hidden_states.shape[1] == 1
            and len(B.shape) == 4
            and B.shape[:2] == (hidden_states.shape[0], 1)
            and C.shape == B.shape
            and B.shape[-1] > 0
            and B.shape[-1] % 32 == 0
            and B.shape[-2] > 0
            and hidden_states.shape[2] % B.shape[-2] == 0
            and state.shape == (
                hidden_states.shape[0],
                hidden_states.shape[2],
                hidden_states.shape[3],
                B.shape[-1],
            )
            and hasattr(mx.fast, "metal_kernel")
            and mx.default_device() == mx.gpu
        )
        if not routable:
            return orig(
                hidden_states,
                A_log,
                B,
                C,
                D,
                dt,
                dt_bias,
                state=state,
                time_step_limit=time_step_limit,
                mask=mask,
                lengths=lengths,
                *rest,
                **kw,
            )
        n, _, h, d = hidden_states.shape
        input_type = hidden_states.dtype
        state_type = state.dtype
        hb, ds = B.shape[-2:]
        dt_comp = module.compute_dt(dt, dt_bias, time_step_limit)
        out, state_out = _get_kernel(mx)(
            inputs=[hidden_states, A_log, B, C, D, dt_comp, state],
            template=[
                ("T", input_type),
                ("U", state_type),
                ("Dh", d),
                ("Ds", ds),
                ("H", h),
                ("G", h // hb),
            ],
            grid=(32, d, h * n),
            threadgroup=(32, 8, 1),
            output_shapes=[(n, 1, h, d), state.shape],
            output_dtypes=[input_type, state_type],
        )
        return out, state_out

    routed._omarchy_route = orig
    module.ssm_update = routed


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
