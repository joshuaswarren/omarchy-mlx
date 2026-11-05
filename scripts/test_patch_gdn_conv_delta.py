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


OLD_BRANCH = '''        gdn_fuse = os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA")
        if (
            gdn_fuse in ("1", "2")
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
                *((z, self.norm.weight, self.norm.eps) if gdn_fuse == "2" else ()),
            )
            cache[0] = new_conv_state
            cache[1] = state
            cache.advance(S)
            if gdn_fuse != "2":
                out = self.norm(out, z)
            return self.out_proj(out.reshape(B, S, -1))
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

    def test_upgrade_replaces_an_earlier_patch(self):
        # A file an earlier version of this patch wrote: its import block and its branch are replaced, not duplicated.
        old = FIXTURE.replace(
            "from dataclasses import dataclass, field\n",
            "import os\nfrom dataclasses import dataclass, field\n",
        ).replace(
            """        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
""",
            OLD_BRANCH,
        )
        self.model.write_text(old)
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        text = self.model.read_text()
        self.assertEqual(text.count("_GDN_CONV_DELTA_DEFAULT"), 2)  # the resolver assignment and the branch
        self.assertEqual(text.count('os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA")'), 1)
        self.assertNotIn("os.environ.get(\"MLX_OMARCHY_GDN_CONV_DELTA\")\n        if (", text)
        compile(text, str(self.model), "exec")
        again = self.run_patch()
        self.assertIn("already patched", again.stdout)

    def test_default_mode_comes_from_the_backend_abi(self):
        # The backend answers with the mode for this device: 2 means conv + delta + gated norm, so the op runs with the flag unset.
        stub = Path(self.tmp.name) / "stub"
        pkg = stub / "mlx"
        (pkg / "lib").mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "core.py").write_text("")
        src = stub / "stub.c"
        src.write_text("int mlx_omarchy_gdn_conv_delta_default(void) { return 2; }\n")
        so = pkg / "lib" / "libmlx.so"
        cc = subprocess.run(["cc", "-shared", "-fPIC", "-o", str(so), str(src)], capture_output=True, text=True)
        if cc.returncode != 0:
            self.skipTest("no C compiler for the stub library: " + cc.stderr[-200:])
        import types

        fake = types.ModuleType("mlx")
        fake.__file__ = str(pkg / "__init__.py")
        fake.__path__ = [str(pkg)]
        core = types.ModuleType("mlx.core")
        core.__file__ = str(pkg / "core.py")
        fake.core = core
        saved = {k: sys.modules.get(k) for k in ("mlx", "mlx.core")}

        def restore():
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        sys.modules["mlx"] = fake
        sys.modules["mlx.core"] = core
        self.addCleanup(restore)
        self.run_patch()
        os.environ.pop("MLX_OMARCHY_GDN_CONV_DELTA", None)
        mod = self.load()
        self.assertEqual(mod._GDN_CONV_DELTA_DEFAULT, "2")
        self.install_fake_op(mod)
        qkv = type("A", (), {"dtype": "bf16"})()
        result = mod.GatedDeltaNet()(qkv=qkv, a="a", b="b", z="z", conv_state="CS", cache=Cache())
        (args,) = mod.mx.fast.calls
        self.assertEqual(args[12:], ("z", "NW", 1e-6))  # the gated norm is folded in
        self.assertEqual(result, ("proj", ("reshape", "OUT", (1, 1, -1))))

    def test_backend_without_the_symbol_keeps_the_stock_route(self):
        # A wheel without the ABI symbol (or no mlx at all) resolves to "0": the stock route, exactly as before the default existed.
        self.run_patch()
        os.environ.pop("MLX_OMARCHY_GDN_CONV_DELTA", None)
        mod = self.load()
        self.assertEqual(mod._GDN_CONV_DELTA_DEFAULT, "0")
        self.install_fake_op(mod)
        self.assertEqual(self.call(mod, Cache())[0], "composed")
        self.assertEqual(mod.mx.fast.calls, [])


if __name__ == "__main__":
    unittest.main()
