#!/usr/bin/env python3
"""Route the Qwen3.5 / Qwen3.8 GDN decode step to mx.fast.gdn_conv_delta_update
(conv, SiLU, q/k norm and the gated-delta step as one dispatch).

Idempotent patch for mlx-lm 0.31.3 venvs, mirroring
patch-mlx-lm-qwen35-gdn-conv.py. The new branch runs only when
MLX_OMARCHY_GDN_CONV_DELTA=1 is set in the process environment, the
attribute exists on mx.fast, and the step is a plain bf16 decode (T == 1, no
mask, no cache lengths, an existing recurrent state, no tensor sharding, GPU
default device). Every other case takes the composed path unchanged, and so
does the op itself wherever it cannot fuse. Apply it after the conv, norm and
raw-gates patches: its anchor exists with or without them.
Usage: python3 patch-mlx-lm-qwen35-gdn-conv-delta.py /path/to/venv
"""
import glob
import sys

IMPORT_OLD = "from dataclasses import dataclass, field\n"
IMPORT_NEW = "import os\nfrom dataclasses import dataclass, field\n"

BRANCH_ANCHOR = """        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
"""
BRANCH_NEW = """        if (
            os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA") == "1"
            and mask is None
            and S == 1
            and cache is not None
            and cache.lengths is None
            and cache[1] is not None
            and qkv.dtype == mx.bfloat16
            and self.sharding_group is None
            and mx.default_device() == mx.gpu
            and hasattr(mx.fast, "gdn_conv_delta_update")
        ):
            inv_scale = self.head_k_dim**-0.5
            out, state, new_conv_state = mx.fast.gdn_conv_delta_update(
                conv_state,
                qkv,
                self.conv1d.weight,
                a,
                b,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.key_dim,
                inv_scale**2,
                inv_scale,
                1e-6,
            )
            cache[0] = new_conv_state
            cache[1] = state
            cache.advance(S)
            out = self.norm(out, z)
            return self.out_proj(out.reshape(B, S, -1))
""" + BRANCH_ANCHOR


def patch_file(path, pairs, marker):
    text = open(path).read()
    if marker in text:
        print("already patched:", path)
        return
    for old, _ in pairs:
        if text.count(old) != 1:
            sys.exit("unrecognized content in " + path + "; refusing to patch")
    for old, new in pairs:
        text = text.replace(old, new)
    open(path, "w").write(text)
    print("patched:", path)


def main(argv):
    venv = argv[1] if len(argv) > 1 else "."
    site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
    if not site:
        sys.exit("mlx_lm/models not found under " + venv)
    # qwen3_5.py only: the Qwen3.5 / Qwen3.8 GatedDeltaNet lives here.
    patch_file(
        site[0] + "/qwen3_5.py",
        [(IMPORT_OLD, IMPORT_NEW), (BRANCH_ANCHOR, BRANCH_NEW)],
        "gdn_conv_delta_update",
    )


if __name__ == "__main__":
    main(sys.argv)
