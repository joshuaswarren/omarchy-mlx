#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the oMLX Qwen3.5 decode conv-fuse patch (0006).

The oMLX MTP ``GatedDeltaNet.__call__`` routes every chunk through
``_process_chunk``'s composed chain (conv1d + silu + normalize_qk as separate
dispatches) even for single-token decode, where the mlx-lm series folds all
three into one ``mx.fast.gdn_conv_update`` dispatch. Patch 0006 adds that
route with the same gates and kill switch as the series.

Asserts on a real GPU wheel:
1. a decode-shaped chunk engages ``mx.fast.gdn_conv_update`` (fails on an
   unpatched tree — that failure is the point);
2. fused output equals the composed chain output bit-for-bit;
3. the kill switch (MLX_OMARCHY_OMLX_CONV_FUSE=0) restores the composed
   path and skips the fused dispatch;
4. non-decode chunks (S > 1, lengths present) never take the fused route.

Needs /tmp/omlx-applied (see test_compat_guards) and an omarchy mlx wheel
with a working GPU default device.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

OMLX_SRC = Path("/tmp/omlx-applied")
if not OMLX_SRC.is_dir():
    raise unittest.SkipTest(
        "omlx src not at /tmp/omlx-applied; from the worktree run: "
        "git clone --no-local --reference $HOME/src/omlx/.git --shared "
        "$HOME/src/omlx /tmp/omlx-applied && cd /tmp/omlx-applied && "
        "git checkout v0.7.0 && git apply $WORKTREE/packaging/omlx-linux/patches/*.patch"
    )
sys.path.insert(0, str(OMLX_SRC))

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError as exc:  # pragma: no cover - dev boxes without the wheel
    raise unittest.SkipTest(f"mlx wheel unavailable: {exc}")

if mx.default_device() != mx.gpu:
    raise unittest.SkipTest("fuse route is GPU-gated; no GPU default device")


class _FakeCache:
    """List-shaped cache surface the MTP GatedDeltaNet body touches."""

    def __init__(self, conv_state, ssm_state):
        self._slots = [conv_state, ssm_state]
        self.lengths = None
        self.advanced = []

    def __getitem__(self, idx):
        return self._slots[idx]

    def __setitem__(self, idx, value):
        self._slots[idx] = value

    def advance(self, n):
        self.advanced.append(n)


def _build_net():
    import mlx_lm.models.qwen3_5 as q35
    import omlx.patches.mlx_lm_mtp.qwen35_model as mtp_q35

    mtp_q35._patch_gated_delta_net(q35)

    config = SimpleNamespace(
        hidden_size=256,
        linear_num_value_heads=8,
        linear_num_key_heads=8,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    net = q35.GatedDeltaNet(config)
    mx.random.seed(3)
    for name, p in net.parameters().items():
        pass  # parameters exist; cast below
    net.set_dtype(mx.bfloat16)
    return net


def _decode_inputs(net):
    x = mx.random.normal((1, 1, net.hidden_size), key=mx.random.key(5)).astype(
        mx.bfloat16
    )
    conv_state = mx.zeros(
        (1, net.conv_kernel_size - 1, net.conv_dim), dtype=mx.bfloat16
    )
    ssm_state = mx.zeros((1, net.num_v_heads, net.head_v_dim, net.head_k_dim), dtype=mx.float32)
    a = mx.random.normal((1, 1, net.num_v_heads), key=mx.random.key(6)).astype(mx.bfloat16)
    b = mx.random.normal((1, 1, net.num_v_heads), key=mx.random.key(7)).astype(mx.bfloat16)
    return x, a, b, _FakeCache(conv_state, ssm_state)


def _bits(a) -> str:
    import hashlib

    return hashlib.sha256(
        str(mx.astype(a, mx.float32).tolist()).encode("utf-8")
    ).hexdigest()


class OmlxQ35ConvFuse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_env = os.environ.pop("MLX_OMARCHY_OMLX_CONV_FUSE", None)
        cls.net = _build_net()

    @classmethod
    def tearDownClass(cls):
        if cls._old_env is not None:
            os.environ["MLX_OMARCHY_OMLX_CONV_FUSE"] = cls._old_env
        else:
            os.environ.pop("MLX_OMARCHY_OMLX_CONV_FUSE", None)

    def _run(self, x, a, b, cache):
        mx.eval(cache._slots[0], cache._slots[1], x, a, b)
        with mx.stream(mx.gpu):
            out = self.net(x, cache=cache)
            mx.eval(out, cache._slots[0], cache._slots[1])
        return out

    def test_decode_engages_fused_conv(self):
        x, a, b, cache = _decode_inputs(self.net)
        calls = []
        real = mx.fast.gdn_conv_update

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        mx.fast.gdn_conv_update = spy
        try:
            self._run(x, a, b, cache)
        finally:
            mx.fast.gdn_conv_update = real
        self.assertTrue(calls, "decode chunk did not engage gdn_conv_update")

    def test_fused_matches_composed_bits(self):
        x, a, b, cache = _decode_inputs(self.net)
        fused = self._run(x, a, b, cache)
        os.environ["MLX_OMARCHY_OMLX_CONV_FUSE"] = "0"
        try:
            x2, a2, b2, cache2 = _decode_inputs(self.net)
            composed = self._run(x2, a2, b2, cache2)
        finally:
            os.environ.pop("MLX_OMARCHY_OMLX_CONV_FUSE", None)
        self.assertEqual(_bits(fused), _bits(composed))

    def test_kill_switch_skips_fused(self):
        x, a, b, cache = _decode_inputs(self.net)
        calls = []
        real = mx.fast.gdn_conv_update

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        mx.fast.gdn_conv_update = spy
        os.environ["MLX_OMARCHY_OMLX_CONV_FUSE"] = "0"
        try:
            self._run(x, a, b, cache)
        finally:
            mx.fast.gdn_conv_update = real
            os.environ.pop("MLX_OMARCHY_OMLX_CONV_FUSE", None)
        self.assertFalse(calls, "kill switch did not skip the fused route")

    def test_multi_token_chunk_skips_fused(self):
        calls = []
        real = mx.fast.gdn_conv_update

        def spy(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        x = mx.random.normal((1, 4, self.net.hidden_size), key=mx.random.key(8)).astype(
            mx.bfloat16
        )
        conv_state = mx.zeros(
            (1, self.net.conv_kernel_size - 1, self.net.conv_dim), dtype=mx.bfloat16
        )
        ssm_state = mx.zeros(
            (1, self.net.num_v_heads, self.net.head_v_dim, self.net.head_k_dim),
            dtype=mx.float32,
        )
        cache = _FakeCache(conv_state, ssm_state)
        mx.fast.gdn_conv_update = spy
        try:
            with mx.stream(mx.gpu):
                out = self.net(x, cache=cache)
                mx.eval(out)
        finally:
            mx.fast.gdn_conv_update = real
        self.assertFalse(calls, "multi-token chunk took the fused route")


if __name__ == "__main__":
    unittest.main()
