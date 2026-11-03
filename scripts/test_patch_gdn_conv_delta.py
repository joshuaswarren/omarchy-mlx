#!/usr/bin/env python3
"""Tests for patch-mlx-lm-qwen35-gdn-conv-delta.py against a fixture that reproduces the real anchor text of
mlx_lm/models/qwen3_5.py (the GatedDeltaNet decode block) with fakes for mx, so the branch logic runs without a GPU."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = HERE / "patch-mlx-lm-qwen35-gdn-conv-delta.py"

FIXTURE = '''from dataclasses import dataclass, field
from typing import Any, Optional


class _Fast:
    calls = []


class _Arr:
    def __init__(self, payload):
        self.payload = payload

    def reshape(self, *shape):
        return ("reshape", self.payload, shape)


class mx:
    bfloat16 = "bf16"
    gpu = "gpu"
    fast = _Fast()
    _device = "gpu"

    @staticmethod
    def default_device():
        return mx._device

    @staticmethod
    def where(m, a, b):
        return ("where", a)


class GatedDeltaNet:
    sharding_group = None
    head_k_dim = 128
    key_dim = 2048
    conv1d = type("C", (), {"weight": "W"})()
    A_log = "A"
    dt_bias = "D"

    class norm_cls:
        weight = "NW"
        eps = 1e-6

        def __call__(self, out, z):
            return _Arr(("norm", out, z))

    norm = norm_cls()

    def out_proj(self, x):
        return ("proj", x)

    def __call__(self, qkv, a, b, z, conv_state, cache, mask=None, B=1, S=1):
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        return ("composed", qkv)
'''


class Cache:
    def __init__(self, state=1, lengths=None):
        self.items = {0: "conv0", 1: state}
        self.lengths = lengths
        self.advanced = 0

    def __getitem__(self, i):
        return self.items[i]

    def __setitem__(self, i, v):
        self.items[i] = v

    def advance(self, n):
        self.advanced += n


class PatchScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        models = Path(self.tmp.name) / "lib/python3.11/site-packages/mlx_lm/models"
        models.mkdir(parents=True)
        self.model = models / "qwen3_5.py"
        self.model.write_text(FIXTURE)

    def tearDown(self):
        self.tmp.cleanup()

    def run_patch(self):
        return subprocess.run([sys.executable, str(PATCH), self.tmp.name], capture_output=True, text=True)

    def load(self):
        spec = importlib.util.spec_from_file_location("patched_qwen3_5", self.model)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def call(self, mod, cache, **kw):
        net = mod.GatedDeltaNet()
        return net(qkv="Q", a="a", b="b", z="z", conv_state="CS", cache=cache, **kw)

    def install_fake_op(self, mod):
        def op(*args):
            mod.mx.fast.calls.append(args)
            return mod._Arr("OUT"), "STATE", "NEWCONV"
        mod.mx.fast.gdn_conv_delta_update = op

    def test_patch_applies_compiles_and_is_idempotent(self):
        first = self.run_patch()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("patched:", first.stdout)
        compile(self.model.read_text(), str(self.model), "exec")
        second = self.run_patch()
        self.assertIn("already patched", second.stdout)
        self.assertEqual(self.model.read_text().count("gdn_conv_delta_update"), 2)  # attribute check and call

    def test_refuses_unrecognized_content(self):
        self.model.write_text("class X:\n    pass\n")
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("refusing to patch", r.stderr + r.stdout)
        self.assertEqual(self.model.read_text(), "class X:\n    pass\n")

    def test_flag_unset_takes_the_composed_path(self):
        self.run_patch()
        os.environ.pop("MLX_OMARCHY_GDN_CONV_DELTA", None)
        mod = self.load()
        self.install_fake_op(mod)
        self.assertEqual(self.call(mod, Cache())[0], "composed")
        self.assertEqual(mod.mx.fast.calls, [])

    def test_flag_set_calls_the_op_with_the_model_scales_and_updates_the_cache(self):
        self.run_patch()
        os.environ["MLX_OMARCHY_GDN_CONV_DELTA"] = "1"
        self.addCleanup(os.environ.pop, "MLX_OMARCHY_GDN_CONV_DELTA", None)
        mod = self.load()
        self.install_fake_op(mod)
        cache = Cache()
        # the fixture qkv is a string, so give the branch a bf16-looking dtype
        qkv = type("A", (), {"dtype": "bf16"})()
        net = mod.GatedDeltaNet()
        result = net(qkv=qkv, a="a", b="b", z="z", conv_state="CS", cache=cache)
        self.assertEqual(result[0], "proj")
        self.assertEqual(result[1][2], (1, 1, -1))
        self.assertEqual(result[1][1][0], "norm")  # flag 1: the separate gated norm still runs
        self.assertEqual(len(mod.mx.fast.calls[0]), 12)
        (args,) = mod.mx.fast.calls
        self.assertEqual(args[0], "CS")
        self.assertEqual(args[2:8], ("W", "a", "b", "A", "D", 1))
        self.assertEqual(args[8], 2048)
        inv = 128 ** -0.5
        self.assertEqual(args[9:], (inv**2, inv, 1e-6))
        self.assertEqual((cache[0], cache[1], cache.advanced), ("NEWCONV", "STATE", 1))

    def test_flag_two_passes_the_norm_arguments_and_skips_the_separate_norm(self):
        self.run_patch()
        os.environ["MLX_OMARCHY_GDN_CONV_DELTA"] = "2"
        self.addCleanup(os.environ.pop, "MLX_OMARCHY_GDN_CONV_DELTA", None)
        mod = self.load()
        self.install_fake_op(mod)
        cache = Cache()
        qkv = type("A", (), {"dtype": "bf16"})()
        result = mod.GatedDeltaNet()(qkv=qkv, a="a", b="b", z="z", conv_state="CS", cache=cache)
        (args,) = mod.mx.fast.calls
        self.assertEqual(args[12:], ("z", "NW", 1e-6))
        self.assertEqual(result, ("proj", ("reshape", "OUT", (1, 1, -1))))
        self.assertEqual((cache[0], cache[1], cache.advanced), ("NEWCONV", "STATE", 1))

    def test_other_flag_values_compose(self):
        self.run_patch()
        os.environ["MLX_OMARCHY_GDN_CONV_DELTA"] = "3"
        self.addCleanup(os.environ.pop, "MLX_OMARCHY_GDN_CONV_DELTA", None)
        mod = self.load()
        self.install_fake_op(mod)
        qkv = type("A", (), {"dtype": "bf16"})()
        out = mod.GatedDeltaNet()(qkv=qkv, a="a", b="b", z="z", conv_state="CS", cache=Cache())
        self.assertEqual(out[0], "composed")

    def test_unsupported_cases_take_the_composed_path(self):
        self.run_patch()
        os.environ["MLX_OMARCHY_GDN_CONV_DELTA"] = "1"
        self.addCleanup(os.environ.pop, "MLX_OMARCHY_GDN_CONV_DELTA", None)
        mod = self.load()
        self.install_fake_op(mod)
        bf16 = type("A", (), {"dtype": "bf16"})()
        f32 = type("A", (), {"dtype": "f32"})()
        net = mod.GatedDeltaNet()
        cases = {
            "prefill (S > 1)": dict(qkv=bf16, cache=Cache(), S=4),
            "no recurrent state yet": dict(qkv=bf16, cache=Cache(state=None)),
            "cache lengths": dict(qkv=bf16, cache=Cache(lengths=[1])),
            "no cache": dict(qkv=bf16, cache=None),
            "not bf16": dict(qkv=f32, cache=Cache()),
            "mask": dict(qkv=bf16, cache=Cache(), mask=type("M", (), {"__getitem__": lambda s, i: "m"})()),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                out = net(a="a", b="b", z="z", conv_state="CS", **kw)
                self.assertEqual(out[0], "composed", name)
        self.assertEqual(mod.mx.fast.calls, [])
        mod.mx._device = "cpu"
        self.assertEqual(net(qkv=bf16, a="a", b="b", z="z", conv_state="CS", cache=Cache())[0], "composed")
        mod.mx._device = "gpu"
        del mod.mx.fast.gdn_conv_delta_update
        self.assertEqual(net(qkv=bf16, a="a", b="b", z="z", conv_state="CS", cache=Cache())[0], "composed")


if __name__ == "__main__":
    unittest.main()
