#!/usr/bin/env python3
"""Route qwen3_5 / qwen3_next GDN norms to mx.fast.rms_norm_gated /
mx.fast.rms_norm_scaled (fused FastNormGatedBF16 kernel).

Idempotent patch for mlx-lm 0.31.3 venvs. Two sites:
- Qwen3NextRMSNormGated.__call__ (qwen3_next.py): the gated norm +
  _precise_swiglu chain becomes one rms_norm_gated call.
- GatedDeltaNet.__call__ (qwen3_5.py): the q/k rms_norm + scalar
  multiply pairs become rms_norm_scaled calls.

Both sites self-guard on hasattr, so the patch is a no-op on stacks
without the primitives. bf16-only routing keeps non-bf16 models on the
composed path. Routing is further gated to decode-sized rows
(size <= 32768 = 256 rows of 128): the fused kernel is proven
bit-exact on every decode shape, but at >= 512 rows (prefill chunks) a
rare ~1e-5-of-outputs 1-ULP deviation vs the composed path appears
(side undetermined: fused large-grid vs composed large-tensor dispatch
difference), so prefill keeps the composed ops and stays bit-identical
to the installed baseline. Usage: python3
patch-mlx-lm-qwen35-gdn-norm.py /path/to/venv
"""
import glob
import sys

GATED_OLD = """    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)"""
GATED_NEW = """    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        if (
            gate is not None
            and hidden_states.dtype == mx.bfloat16
            and hidden_states.size <= 32768
            and mx.default_device() == mx.gpu
            and hasattr(mx.fast, "rms_norm_gated")
        ):
            return mx.fast.rms_norm_gated(
                hidden_states, gate, self.weight, self.eps
            )
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)"""

QK_OLD = """        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)"""
QK_NEW = """        if (
            q.dtype == mx.bfloat16
            and q.size <= 32768
            and hasattr(mx.fast, "rms_norm_scaled")
        ):
            q = mx.fast.rms_norm_scaled(q, None, inv_scale * inv_scale, 1e-6)
            k = mx.fast.rms_norm_scaled(k, None, inv_scale, 1e-6)
        else:
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)"""


def patch_file(path, old, new, marker):
    text = open(path).read()
    if marker in text:
        print("already patched:", path)
        return
    if old not in text:
        sys.exit("unrecognized content in " + path + "; refusing to patch")
    open(path, "w").write(text.replace(old, new))
    print("patched:", path)


venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
site = site[0]

patch_file(
    site + "/qwen3_next.py",
    GATED_OLD,
    GATED_NEW,
    "rms_norm_gated",
)
for model in ("qwen3_5.py", "qwen3_next.py"):
    patch_file(
        site + "/" + model,
        QK_OLD,
        QK_NEW,
        "rms_norm_scaled",
    )
