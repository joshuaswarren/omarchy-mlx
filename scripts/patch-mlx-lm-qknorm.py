#!/usr/bin/env python3
"""Fold the GDN q/k rms_norm_scaled pair into mx.fast.gdn_conv_update
(F4, the gdn_conv_decode.comp epilogue). Idempotent patch for mlx-lm
0.31.3 venvs, mirroring patch-mlx-lm-rope-norm.py. Site: GatedDeltaNet.
__call__ in qwen3_5.py (the only model file with the gdn_conv_update
fast branch).

The fused branch is gated on MLX_OMARCHY_GDN_QKNORM_FUSE=1 (default 0:
the exact composed chain runs; kill switch =0 at any time) and on the
geometry the fused kernel requires (head_k_dim == 128, key_dim % 256 ==
0, B*C % 256 == 0); the backend refuses any remaining violation loudly.
Bit-exact: the epilogue reproduces the FastNormGatedBF16 mode-1
reduction tree and rounding sequence per 128-channel head-row of the
conv output. Usage: python3 patch-mlx-lm-qknorm.py /path/to/venv
"""
import glob
import sys

CONV_OLD = """        and hasattr(mx.fast, "gdn_conv_update")
        ):
            conv_out, new_state = mx.fast.gdn_conv_update(
                conv_state, qkv, self.conv1d.weight, activate=True
            )
            cache[0] = new_state
        else:
"""
CONV_NEW = """        and hasattr(mx.fast, "gdn_conv_update")
        ):
            # mlx-omarchy qk-norm patch (F4): fold the GDN q/k
            # rms_norm_scaled pair into the conv dispatch epilogue.
            # Kill switch: MLX_OMARCHY_GDN_QKNORM_FUSE=0 (or any gate
            # mismatch) runs the exact composed chain below.
            qknorm_fused = False
            if (
                os.environ.get("MLX_OMARCHY_GDN_QKNORM_FUSE", "0") == "1"
                and self.head_k_dim == 128
                and self.key_dim % 256 == 0
                and (qkv.shape[0] * qkv.shape[2]) % 256 == 0
            ):
                inv_scale_qk = self.head_k_dim**-0.5
                conv_out, new_state = mx.fast.gdn_conv_update(
                    conv_state,
                    qkv,
                    self.conv1d.weight,
                    activate=True,
                    qk_key_dim=self.key_dim,
                    qk_scale_q=inv_scale_qk * inv_scale_qk,
                    qk_scale_k=inv_scale_qk,
                    qk_eps=1e-6,
                )
                qknorm_fused = True
            else:
                conv_out, new_state = mx.fast.gdn_conv_update(
                    conv_state, qkv, self.conv1d.weight, activate=True
                )
            cache[0] = new_state
        else:
            qknorm_fused = False
"""

NORM_OLD = """        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        if (
            q.dtype == mx.bfloat16
            and hasattr(mx.fast, "rms_norm_scaled")
        ):
"""
NORM_NEW = """        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        if qknorm_fused:
            pass  # q/k norms folded into the gdn_conv_update epilogue
        elif (
            q.dtype == mx.bfloat16
            and hasattr(mx.fast, "rms_norm_scaled")
        ):
"""

MARKER = "MLX_OMARCHY_GDN_QKNORM_FUSE"

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
q = site[0] + "/qwen3_5.py"
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if "\nimport os\n" not in text:
    anchor = "from dataclasses import dataclass, field\n"
    if anchor not in text:
        sys.exit("import anchor missing in " + q)
    text = text.replace(anchor, anchor + "\nimport os\n", 1)
    print("added os import:", q)
for old, new in ((CONV_OLD, CONV_NEW), (NORM_OLD, NORM_NEW)):
    if old not in text:
        sys.exit("unrecognized content in " + q + "; refusing to patch")
    text = text.replace(old, new, 1)
open(q, "w").write(text)
print("patched:", q)
