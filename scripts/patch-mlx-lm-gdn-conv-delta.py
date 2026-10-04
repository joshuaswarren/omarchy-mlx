#!/usr/bin/env python3
"""Fuse the GDN decode conv and gated-delta update into one dispatch (F7).

Replaces the composed chain
  mx.fast.gdn_conv_update(...)          (conv + silu + ring + F4 q/k norms)
  split/reshape -> gated_delta_update_raw
with mx.fast.gdn_conv_delta_update(...) in qwen3_5.py's GatedDeltaNet
__call__ decode branch. The backend kernel (gdn_conv_delta_decode.comp)
copies both deployed kernels' arithmetic verbatim, so the fused route is
bit-exact to the chain it replaces; the captured-operand doctest and the
production digest pins carry the proof.

Gate: MLX_OMARCHY_GDN_CONV_DELTA (default OFF until proven; kill switch is
leaving it unset). The route self-guards on hasattr, decode geometry
(head dims 128, Hv % Hk == 0), qknorm_fused (the F4 conv branch), a None
mask, and a live GDN state; anything else runs the exact existing chain.

Site: GatedDeltaNet.__call__ in qwen3_5.py, applied AFTER
patch-mlx-lm-qknorm.py (requires the qknorm_fused branch it introduces).
mlx-lm 0.31.3 series only; the 0.32 line refuses loudly until its tail is
ported (recorded in the lane entry).

Usage: python3 patch-mlx-lm-gdn-conv-delta.py /path/to/venv
"""
import glob
import sys

INSERT_OLD = """            cache[0] = new_state
        else:
"""
INSERT_NEW = """            cache[0] = new_state
            gdu_fused = False
            if (
                os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA", "0") == "1"
                and qknorm_fused
                and mask is None
                and self.head_k_dim == 128
                and self.head_v_dim == 128
                and self.num_v_heads % self.num_k_heads == 0
                and hasattr(mx.fast, "gdn_conv_delta_update")
            ):
                state = cache[1]
                if state is not None:
                    (
                        out,
                        new_conv_state,
                        state,
                    ) = mx.fast.gdn_conv_delta_update(
                        conv_state,
                        state,
                        qkv,
                        self.conv1d.weight,
                        a,
                        b,
                        self.A_log,
                        self.dt_bias,
                        activate=True,
                        qk_key_dim=self.key_dim,
                        qk_scale_q=inv_scale_qk * inv_scale_qk,
                        qk_scale_k=inv_scale_qk,
                        qk_eps=1e-6,
                    )
                    cache[0] = new_conv_state
                    cache[1] = state
                    cache.advance(S)
                    gdu_fused = True
        else:
"""

GUARD_OLD = """        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        if qknorm_fused:
            pass  # q/k norms folded into the gdn_conv_update epilogue
        elif (
            q.dtype == mx.bfloat16
            and hasattr(mx.fast, "rms_norm_scaled")
        ):
            q = mx.fast.rms_norm_scaled(q, None, inv_scale * inv_scale, 1e-6)
            k = mx.fast.rms_norm_scaled(k, None, inv_scale, 1e-6)
        else:
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)
"""
GUARD_NEW = """        if gdu_fused:
            # conv + q/k norms + gated delta update ran in one dispatch;
            # out, the GDN state, and both cache slots are already current.
            pass
        else:
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                    [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                    [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                )
            ]

            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            if qknorm_fused:
                pass  # q/k norms folded into the gdn_conv_update epilogue
            elif (
                q.dtype == mx.bfloat16
                and hasattr(mx.fast, "rms_norm_scaled")
            ):
                q = mx.fast.rms_norm_scaled(q, None, inv_scale * inv_scale, 1e-6)
                k = mx.fast.rms_norm_scaled(k, None, inv_scale, 1e-6)
            else:
                q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
                k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            out, state = gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                mask,
                use_kernel=not self.training,
            )

            if cache is not None:
                cache[1] = state
                cache.advance(S)
"""

INIT_OLD = """        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        if (
            qkv.dtype == mx.bfloat16
"""
INIT_NEW = """        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        gdu_fused = False
        if (
            qkv.dtype == mx.bfloat16
"""

MARKER = "MLX_OMARCHY_GDN_CONV_DELTA"

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
q = site[0] + "/qwen3_5.py"
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if "qknorm_fused" not in text:
    sys.exit(
        "qwen3_5.py has no qknorm_fused branch; run patch-mlx-lm-qknorm.py "
        "first (this patcher sits on top of the F4 fold)"
    )
for old, new in ((INSERT_OLD, INSERT_NEW), (GUARD_OLD, GUARD_NEW), (INIT_OLD, INIT_NEW)):
    if old not in text:
        sys.exit("unrecognized content in " + q + "; refusing to patch")
    text = text.replace(old, new, 1)
open(q, "w").write(text)
import py_compile  # noqa: E402

py_compile.compile(q, doraise=True)
print("patched:", q)
