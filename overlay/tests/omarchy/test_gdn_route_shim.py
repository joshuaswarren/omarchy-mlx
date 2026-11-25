# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Contract test for mlx._omarchy_gdn_route: the wheel's pip-mlx-lm
gated-delta route hook.

The shim is exercised with stub mlx.core / mlx_lm.modules.gated_delta
modules: prefill routes to mx.fast.gated_delta_update, decode to
gated_delta_update_raw, Hk != Hv prefill repeats q/k, use_kernel=False
and the env kill switch keep the original call, and a module that
already calls mx.fast.gated_delta_update is left alone.
"""

import importlib
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

_SHIM = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "mlx"
    / "_omarchy_gdn_route.py"
)

_MLX_CORE_STUB = """
calls = []


class _Fast:
    @staticmethod
    def gated_delta_update(q, k, v, g, beta, state, mask):
        calls.append(("gdu", q, k, v, g, beta, state, mask))
        return ("y", state)

    @staticmethod
    def gated_delta_update_raw(q, k, v, a, b, A_log, dt_bias, state, mask):
        calls.append(("raw", q, k, v, a, b, A_log, dt_bias, state, mask))
        return ("y", state)


fast = _Fast()
gpu = object()


def default_device():
    return gpu


def sigmoid(x):
    return ("sig", x)


def zeros(shape, dtype=None):
    return ("zeros", shape, dtype)


def repeat(x, n, ax):
    return _T([x.shape[0], x.shape[1], x.shape[2] * n, x.shape[3]])


class _T:
    def __init__(self, shape):
        self.shape = shape


float32 = "f32"
"""

_GATED_DELTA_STUB = """
def compute_g(A_log, a, dt_bias):
    return ("g", A_log, a, dt_bias)


def gated_delta_ops(*a, **k):
    return "OPS"


def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None,
                       mask=None, use_kernel=True):
    return gated_delta_ops(q, k, v)
"""

_GATED_DELTA_NATIVE_ROUTE = _GATED_DELTA_STUB.replace(
    "return gated_delta_ops(q, k, v)",
    "if hasattr(mx.fast, 'gated_delta_update'):\n"
    "        return mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)\n"
    "    return gated_delta_ops(q, k, v)",
).replace(
    "def gated_delta_ops",
    "import mlx.core as mx\n\n\ndef gated_delta_ops",
)


class _T:
    def __init__(self, shape):
        self.shape = shape


def _stage(stub_root, gated_delta_source):
    """Write the stub mlx / mlx_lm packages and load the real shim."""
    mlx_dir = stub_root / "mlx"
    gd_dir = stub_root / "mlx_lm" / "models"
    gd_dir.mkdir(parents=True)
    mlx_dir.mkdir(parents=True)
    (stub_root / "mlx_lm" / "__init__.py").write_text("")
    (gd_dir / "__init__.py").write_text("")
    (gd_dir / "gated_delta.py").write_text(gated_delta_source)
    (stub_root / "mlx" / "__init__.py").write_text("")
    (mlx_dir / "core.py").write_text(_MLX_CORE_STUB)
    shim_src = _SHIM.read_text()
    (mlx_dir / "_omarchy_gdn_route.py").write_text(shim_src)


class GdnRouteShimTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._root = Path(self._tmp.name)
        _stage(self._root, _GATED_DELTA_STUB)
        self._saved_path = list(sys.path)
        self._saved_modules = dict(sys.modules)
        self.addCleanup(self._restore)
        sys.path.insert(0, str(self._root))
        for name in list(sys.modules):
            if name.split(".")[0] in ("mlx", "mlx_lm"):
                del sys.modules[name]
        os.environ.pop("MLX_OMARCHY_NO_GDN_ROUTE", None)

    def _restore(self):
        sys.path[:] = self._saved_path
        for name in list(sys.modules):
            if name.split(".")[0] in ("mlx", "mlx_lm"):
                del sys.modules[name]
        sys.modules.update(
            {k: v for k, v in self._saved_modules.items()
             if k.split(".")[0] in ("mlx", "mlx_lm")})

    def _import(self):
        import mlx.core  # noqa: F401 -- stub; the shim import is below
        import mlx._omarchy_gdn_route  # noqa: F401 -- NB_MODULE hook
        import mlx_lm.models.gated_delta as gd
        return gd

    def test_prefill_routes_to_gated_delta_update(self):
        gd = self._import()
        q = _T([1, 512, 16, 128])
        v = _T([1, 512, 16, 128])
        out = gd.gated_delta_update(q, q, v, "a", "b", "Al", "dtb")
        import mlx.core as mx
        call = mx.calls[-1]
        self.assertEqual(call[0], "gdu")
        self.assertEqual(call[4], ("g", "Al", "a", "dtb"))
        self.assertEqual(out[0], "y")

    def test_decode_routes_to_raw(self):
        gd = self._import()
        q = _T([1, 1, 16, 128])
        gd.gated_delta_update(q, q, q, "a", "b", "Al", "dtb", state="S0")
        import mlx.core as mx
        self.assertEqual(mx.calls[-1][0], "raw")

    def test_gqa_prefill_repeats_qk(self):
        gd = self._import()
        q = _T([1, 8, 4, 128])
        v = _T([1, 8, 16, 128])
        gd.gated_delta_update(q, q, v, "a", "b", "Al", "dtb")
        import mlx.core as mx
        call = mx.calls[-1]
        self.assertEqual(call[0], "gdu")
        self.assertEqual(call[1].shape, [1, 8, 16, 128])
        self.assertEqual(call[2].shape, [1, 8, 16, 128])

    def test_use_kernel_false_keeps_ops(self):
        gd = self._import()
        q = _T([1, 4, 16, 128])
        out = gd.gated_delta_update(
            q, q, q, "a", "b", "Al", "dtb", use_kernel=False)
        self.assertEqual(out, "OPS")

    def test_kill_switch_keeps_ops(self):
        gd = self._import()
        os.environ["MLX_OMARCHY_NO_GDN_ROUTE"] = "1"
        q = _T([1, 4, 16, 128])
        out = gd.gated_delta_update(q, q, q, "a", "b", "Al", "dtb")
        self.assertEqual(out, "OPS")

    def test_native_route_module_left_alone(self):
        gd_dir = self._root / "mlx_lm" / "models"
        (gd_dir / "gated_delta.py").write_text(_GATED_DELTA_NATIVE_ROUTE)
        gd = self._import()
        self.assertIsNone(
            getattr(gd.gated_delta_update, "_omarchy_route", None))

    def test_module_already_imported_is_wrapped(self):
        import mlx.core  # noqa: F401
        import mlx_lm.models.gated_delta as gd  # before the shim
        import mlx._omarchy_gdn_route  # noqa: F401 -- sys.modules path
        self.assertIsNotNone(
            getattr(gd.gated_delta_update, "_omarchy_route", None))

    def test_raw_route_gets_f32_state_when_state_none(self):
        gd = self._import()
        q = _T([1, 1, 16, 128])
        v = _T([1, 1, 16, 128])
        gd.gated_delta_update(q, q, v, "a", "b", "Al", "dtb")
        import mlx.core as mx
        call = mx.calls[-1]
        self.assertEqual(call[0], "raw")
        self.assertEqual(call[8], ("zeros", (1, 16, 128, 128), "f32"))

    def test_sourceless_module_left_alone(self):
        import py_compile
        gd_dir = self._root / "mlx_lm" / "models"
        py = gd_dir / "gated_delta.py"
        py_compile.compile(str(py), str(gd_dir / "gated_delta.pyc"))
        py.unlink()
        gd = self._import()
        self.assertTrue(gd.__file__.endswith(".pyc"))
        self.assertIsNone(
            getattr(gd.gated_delta_update, "_omarchy_route", None))

    def test_module_without_exports_left_alone(self):
        gd_dir = self._root / "mlx_lm" / "models"
        (gd_dir / "gated_delta.py").write_text("X = 1\n")
        gd = self._import()
        self.assertFalse(hasattr(gd, "gated_delta_update"))


if __name__ == "__main__":
    unittest.main()
