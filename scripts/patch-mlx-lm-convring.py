#!/usr/bin/env python3
"""Preallocated rolling conv-state buffer for mlx-lm GDN decode.

Idempotent patch for mlx-lm 0.31.3 qwen3_5.py venvs. Replaces the
per-token concatenate([conv_state, qkv]) + contiguous() conv-state
rolling window (the dominant CopyGeneralBF16 source, ~43 dispatches/
token) with a preallocated ring buffer: one slice-update row write per
step, conv1d on the contiguous 4-row window slice. Decode (S == 1,
no lengths) only; prefill and lengthed paths keep the original code.

Usage: python scripts/patch-mlx-lm-convring.py /path/to/venv
"""
import glob
import sys

RING_CLASS = '''
class _ConvRing:
    """Preallocated rolling conv-state window (decode, S == 1)."""

    def __init__(self, seed, n):
        B, _, C = seed.shape
        self.n = n
        self.cap = 260
        self.buf = mx.zeros((B, self.cap, C), seed.dtype)
        self.buf[:, :n, :] = seed
        self.off = 0

    def step(self, qkv):
        if self.off + self.n + 1 > self.cap:
            self.buf[:, : self.n, :] = self.buf[
                :, self.off + 1 : self.off + 1 + self.n, :
            ]
            self.off = 0
        self.buf[:, self.off + self.n, :] = qkv[:, 0, :]
        win = self.buf[:, self.off : self.off + self.n + 1, :]
        self.off += 1
        return win
'''

NEW_BLOCK = '''        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        n_keep = self.conv_kernel_size - 1
        ring = getattr(cache, "_conv_ring", None) if cache is not None else None
        if ring is not None and (S != 1 or cache.lengths is not None):
            ring = None
            cache._conv_ring = None
        if (
            ring is None
            and cache is not None
            and S == 1
            and cache.lengths is None
        ):
            seed = (
                cache[0]
                if cache[0] is not None
                else mx.zeros(
                    (B, n_keep, self.conv_dim), dtype=inputs.dtype
                )
            )
            ring = _ConvRing(seed, n_keep)
            cache._conv_ring = ring
        if ring is not None:
            conv_input = ring.step(qkv)
            cache[0] = conv_input[:, -n_keep:, :]
        else:
            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, n_keep, self.conv_dim), dtype=inputs.dtype
                )
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            if cache is not None:
                if cache.lengths is not None:
                    ends = mx.clip(cache.lengths, 0, S)
                    positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                    cache[0] = mx.take_along_axis(
                        conv_input, positions, axis=1
                    )
                else:
                    cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))'''

OLD_BLOCK = '''        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
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
        conv_out = nn.silu(self.conv1d(conv_input))'''


def main(venv):
    paths = glob.glob(
        venv.rstrip("/") + "/lib/python3.*/site-packages/mlx_lm/models/qwen3_5.py"
    )
    if not paths:
        sys.exit("qwen3_5.py not found under " + venv)
    path = paths[0]
    src = open(path).read()
    if "_ConvRing" in src:
        print("already patched:", path)
        return
    if OLD_BLOCK not in src:
        sys.exit("conv block not found (mlx-lm version mismatch?) in " + path)
    src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    anchor = "\nclass GatedDeltaNet("
    if anchor not in src:
        sys.exit("GatedDeltaNet anchor not found")
    src = src.replace(anchor, "\n" + RING_CLASS + anchor, 1)
    open(path, "w").write(src)
    print("patched:", path)


if __name__ == "__main__":
    main(sys.argv[1])
