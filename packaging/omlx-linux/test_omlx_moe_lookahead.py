#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the oMLX MoE expert-offload lookahead patch (0011).

Patch 0011 adds, to ``omlx/patches/moe_expert_offload.py``:

1. a router-driven lookahead: once a layer's routes are read back, the next
   wrapped layer's previous-step routes are speculatively fetched (insert-only
   — never evicting a resident expert), so its SSD reads overlap the current
   layer's queued GPU work; ``OMLX_MOE_OFFLOAD_LOOKAHEAD=0`` restores v1;
2. a slot admission bill: a layer whose capacity cannot hold the routing
   floor is refused by name before any slot array is allocated, and an
   explicit ``budget_bytes`` refuses the whole offload with ``ValueError``.

Asserts on the patched tree (/tmp/omlx-applied) with any mlx device:

1. lookahead on/off produce bit-identical decode outputs;
2. the admission bill refuses a below-floor layer by name and leaves it stock;
3. ``budget_bytes`` refusal raises before wrapping anything;
4. consecutive wrapped layers are chained (``cache.next_cache``);
5. speculative installs never evict.

Dev boxes without the patched tree skip.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

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
    from mlx_lm.models.switch_layers import SwitchGLU
except ImportError as exc:  # pragma: no cover - dev boxes without the wheel
    raise unittest.SkipTest(f"mlx wheel unavailable: {exc}")

from omlx.patches import moe_expert_offload as offload
from omlx.patches.moe_expert_offload import (
    OffloadSwitchGLU,
    apply_moe_expert_offload,
    moe_offload_stats,
)

E, D, INTER, K, GROUP = 32, 64, 32, 2, 32


def _make_glu(seed=0, e=E):
    mx.random.seed(seed)
    glu = SwitchGLU(D, INTER, e)
    nn.quantize(glu, group_size=GROUP, bits=4)
    return glu


def _glu_tensors(glu, prefix):
    out = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin = getattr(glu, proj)
        for field in ("weight", "scales", "biases"):
            if lin.get(field) is not None:
                out[f"{prefix}.{proj}.{field}"] = lin[field]
    return out


class _Layer(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.experts = nn.Module()
        self.experts.switch_glu = glu


class _MiniMoE(nn.Module):
    def __init__(self, glus):
        super().__init__()
        self.layers = [_Layer(g) for g in glus]

    def __call__(self, x, indices):
        for layer in self.layers:
            x = x + layer.experts.switch_glu(x, indices).sum(axis=-2)
        return x


def _build(tmp, n_layers=3):
    """A fresh checkpoint in ``tmp`` plus its model (both sides identical)."""
    glus = [_make_glu(seed=i) for i in range(n_layers)]
    tensors = {}
    for i, g in enumerate(glus):
        tensors.update(_glu_tensors(g, f"layers.{i}.experts.switch_glu"))
    mx.save_safetensors(str(Path(tmp) / "model.safetensors"), tensors)
    return _MiniMoE(glus)


def _caches(model):
    return [layer.experts.switch_glu.cache for layer in model.layers]


def _glus(model):
    return [layer.experts.switch_glu for layer in model.layers]


def _shutdown():
    offload._shutdown_io_pool()


class Lookahead(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        old = mx.default_device()
        mx.set_default_device(mx.cpu)  # save_safetensors quantize direction
        self._old_dev = old
        os.environ["OMLX_MOE_OFFLOAD_IO_WORKERS"] = "4"
        os.environ.pop("OMLX_MOE_OFFLOAD_LOOKAHEAD", None)
        _shutdown()

    def tearDown(self):
        os.environ.pop("OMLX_MOE_OFFLOAD_LOOKAHEAD", None)
        os.environ.pop("OMLX_MOE_OFFLOAD_IO_WORKERS", None)
        _shutdown()
        mx.set_default_device(self._old_dev)
        self._tmp.cleanup()

    def test_lookahead_bit_exact_and_kill_switch(self):
        routes = [
            mx.array([[1, 5]], dtype=mx.int32),
            mx.array([[1, 9]], dtype=mx.int32),
            mx.array([[9, 2]], dtype=mx.int32),
            mx.array([[2, 1]], dtype=mx.int32),
        ]
        xs = [mx.random.normal((1, 1, D)) for _ in routes]

        la = _build(self._tmp.name)
        self.assertEqual(apply_moe_expert_offload(la, self._tmp.name, 0.25), 3)
        outs = [la(x, i) for x, i in zip(xs, routes)]
        mx.eval(*outs)
        for c in _caches(la):
            self.assertIsNotNone(c.last_routes)
        self.assertGreaterEqual(moe_offload_stats(la)["ensure_s"], 0)

        os.environ["OMLX_MOE_OFFLOAD_LOOKAHEAD"] = "0"
        _shutdown()
        with tempfile.TemporaryDirectory() as t2:
            base = _build(t2)
            self.assertEqual(apply_moe_expert_offload(base, t2, 0.25), 3)
            base_outs = [base(x, i) for x, i in zip(xs, routes)]
            mx.eval(*base_outs)
        for got, want in zip(outs, base_outs):
            self.assertTrue(bool(mx.array_equal(got, want)))

    def test_chaining(self):
        model = _build(self._tmp.name)
        self.assertEqual(apply_moe_expert_offload(model, self._tmp.name, 0.25), 3)
        c0, c1, c2 = _caches(model)
        self.assertIs(c0.next_cache, c1)
        self.assertIs(c1.next_cache, c2)
        self.assertIsNone(c2.next_cache)

    def test_admission_refuses_below_floor_layer(self):
        model = _MiniMoE([_make_glu(seed=0, e=4)])
        _build_one(self._tmp.name, model.layers[0].experts.switch_glu,
                   "layers.0.experts.switch_glu")
        with self.assertLogs("omlx.patches.moe_expert_offload", level="WARNING") as log:
            self.assertEqual(apply_moe_expert_offload(model, self._tmp.name, 0.25), 0)
        self.assertTrue(any("admission refused" in m for m in log.output))
        self.assertNotIsInstance(
            model.layers[0].experts.switch_glu, OffloadSwitchGLU
        )

    def test_budget_refusal_raises_before_allocating(self):
        model = _build(self._tmp.name, n_layers=2)
        with self.assertRaisesRegex(ValueError, "admission refused"):
            apply_moe_expert_offload(model, self._tmp.name, 0.25, budget_bytes=1)
        self.assertNotIsInstance(
            model.layers[0].experts.switch_glu, OffloadSwitchGLU
        )

    def test_speculative_installs_never_evict(self):
        model = _build(self._tmp.name, n_layers=2)
        self.assertEqual(apply_moe_expert_offload(model, self._tmp.name, 0.25), 2)
        cache = _caches(model)[1]
        residents = set(range(cache.capacity))
        cache._spec_queue(residents)
        self.assertGreater(cache.pred_sent, 0)
        deadline = time.time() + 30
        while any(
            not all(f.done() for _, _, _, f in g)
            for g in cache.spec_pending.values()
        ):
            self.assertLess(time.time(), deadline, "speculative reads never finished")
            time.sleep(0.01)
        cache._spec_install()
        self.assertEqual(set(cache.slot_of), residents)
        cache._spec_queue(range(cache.capacity, cache.capacity + 4))
        cache._spec_install()
        self.assertEqual(set(cache.slot_of), residents)  # nothing evicted


def _build_one(tmp, glu, prefix):
    mx.save_safetensors(
        str(Path(tmp) / "model.safetensors"), _glu_tensors(glu, prefix)
    )


if __name__ == "__main__":
    unittest.main()
