#!/usr/bin/env python3
"""Route the Qwen3.5 GDN decode conv to mx.fast.gdn_conv_update (fused
GdnConvDecodeBF16 kernel: state++x concatenation folded into the conv
read, carry-out state written in-kernel).

Idempotent patch for mlx-lm 0.31.3 venvs, mirroring
patch-mlx-lm-qwen35-gdn-norm.py. The site self-guards on hasattr, so
the patch is a no-op on stacks without the primitive. Decode-only
(qkv.shape[1] == 1) and requires the no-lengths ArraysCache branch the
composed path takes (take_along_axis state update stays composed).
bf16-only routing keeps other dtypes on the composed path. Usage:
python3 patch-mlx-lm-qwen35-gdn-conv.py /path/to/venv
"""
import glob
import sys

CONV_OLD = """        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))"""
CONV_NEW = """        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        if (
            qkv.dtype == mx.bfloat16
            and qkv.shape[1] == 1
            and cache is not None
            and cache.lengths is None
            and mx.default_device() == mx.gpu
            and hasattr(mx.fast, "gdn_conv_update")
        ):
            conv_out, new_state = mx.fast.gdn_conv_update(
                conv_state, qkv, self.conv1d.weight
            )
            cache[0] = new_state
            conv_out = nn.silu(conv_out)
        else:
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                n_keep = self.conv_kernel_size - 1
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
            conv_out = nn.silu(self.conv1d(conv_input))"""


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

# qwen3_5.py only: the Qwen3.8 (qwen3_5) GatedDeltaNet lives here;
# qwen3_next.py's GatedDeltaNet variant has a different conv block.
patch_file(
    site + "/qwen3_5.py",
    CONV_OLD,
    CONV_NEW,
    "gdn_conv_update",
)
