# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Contract test for mlx._omarchy_ssm_route: the wheel's pip-mlx-lm
Mamba-2 / Nemotron-H ssm_update decode route hook.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_SHIM = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "mlx"
    / "_omarchy_ssm_route.py"
)

_MLX_CORE_STUB = """
calls = []


class _Kernel:
    def __init__(self, name, input_names, output_names, source):
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.source = source

    def __call__(self, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        calls.append(("ssm_kernel", self.name, inputs, template, grid, threadgroup,
                      output_shapes, output_dtypes))
        return ["y_fast", "state_fast"]


class _Fast:
    @staticmethod
    def metal_kernel(name, input_names, output_names, source):
        return _Kernel(name, input_names, output_names, source)


fast = _Fast()
gpu = object()


def default_device():
    return gpu
"""

_SSM_STUB = """
import mlx.core as mx


def compute_dt(dt, dt_bias, time_step_limit):
    return ("dt_comp", dt, dt_bias, time_step_limit)


def ssm_attn(*a, **k):
    return ("ATTN_Y", "ATTN_S")


def ssm_update(hidden_states, A_log, B, C, D, dt, dt_bias, state=None,
               time_step_limit=(0.001, 100.0), mask=None, lengths=None):
    if not mx.metal.is_available():
        return ssm_attn(hidden_states, A_log, B, C, D, dt, dt_bias, state,
                        time_step_limit, mask=mask, lengths=lengths)
    return ("METAL_Y", "METAL_S")
"""


class _T:
    def __init__(self, shape, dtype="bf16"):
        self.shape = shape
        self.dtype = dtype


def _stage(stub_root, ssm_source):
    mlx_dir = stub_root / "mlx"
    m_dir = stub_root / "mlx_lm" / "models"
    m_dir.mkdir(parents=True)
    mlx_dir.mkdir(parents=True)
    (stub_root / "mlx_lm" / "__init__.py").write_text("")
    (m_dir / "__init__.py").write_text("")
    (m_dir / "ssm.py").write_text(ssm_source)
    (stub_root / "mlx" / "__init__.py").write_text("")
    (mlx_dir / "core.py").write_text(_MLX_CORE_STUB)
    (mlx_dir / "_omarchy_ssm_route.py").write_text(_SHIM.read_text())


class SsmRouteShimTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._root = Path(self._tmp.name)
        _stage(self._root, _SSM_STUB)
        self._saved_path = list(sys.path)
        self._saved_meta_path = list(sys.meta_path)
        self._saved_modules = dict(sys.modules)
        self.addCleanup(self._restore)
        sys.path.insert(0, str(self._root))
        for name in list(sys.modules):
            if name.split(".")[0] in ("mlx", "mlx_lm"):
                del sys.modules[name]
        os.environ.pop("MLX_OMARCHY_NO_SSM_ROUTE", None)

    def _restore(self):
        sys.path[:] = self._saved_path
        sys.meta_path[:] = self._saved_meta_path
        for name in list(sys.modules):
            if name.split(".")[0] in ("mlx", "mlx_lm"):
                del sys.modules[name]
        sys.modules.update(
            {k: v for k, v in self._saved_modules.items()
             if k.split(".")[0] in ("mlx", "mlx_lm")})

    def _import(self):
        import mlx.core  # noqa: F401
        import mlx._omarchy_ssm_route  # noqa: F401
        import mlx_lm.models.ssm as ssm
        return ssm

    def test_decode_routes_to_ssm_kernel(self):
        ssm = self._import()
        x = _T((1, 1, 64, 64), "bf16")
        B = _T((1, 1, 8, 128), "bf16")
        C = _T((1, 1, 8, 128), "bf16")
        st = _T((1, 64, 64, 128), "f32")
        out = ssm.ssm_update(x, "A", B, C, "D", "dt", "dtb", state=st)
        import mlx.core as mx
        self.assertEqual(out, ("y_fast", "state_fast"))
        tag, name, inputs, template, grid, tg, oshapes, odtypes = mx.calls[-1]
        self.assertEqual(tag, "ssm_kernel")
        self.assertEqual(name, "ssm_kernel")
        self.assertEqual(inputs[5], ("dt_comp", "dt", "dtb", (0.001, 100.0)))
        self.assertEqual(template, [
            ("T", "bf16"), ("U", "f32"), ("Dh", 64), ("Ds", 128), ("H", 64), ("G", 8),
        ])
        self.assertEqual(grid, (32, 64, 64))
        self.assertEqual(tg, (32, 8, 1))
        self.assertEqual(oshapes, [(1, 1, 64, 64), (1, 64, 64, 128)])
        self.assertEqual(odtypes, ["bf16", "f32"])

    def test_prefill_stays_on_ssm_attn(self):
        ssm = self._import()
        x = _T((1, 128, 64, 64), "bf16")
        B = _T((1, 128, 8, 128), "bf16")
        st = _T((1, 64, 64, 128), "f32")
        import mlx.core as mx
        mx.metal = type("M", (), {"is_available": staticmethod(lambda: False)})()
        self.assertEqual(ssm.ssm_update(x, "A", B, B, "D", "dt", "dtb", state=st), ("ATTN_Y", "ATTN_S"))

    def test_none_state_stays_on_ssm_attn(self):
        ssm = self._import()
        x = _T((1, 1, 64, 64), "bf16")
        B = _T((1, 1, 8, 128), "bf16")
        import mlx.core as mx
        mx.metal = type("M", (), {"is_available": staticmethod(lambda: False)})()
        self.assertEqual(ssm.ssm_update(x, "A", B, B, "D", "dt", "dtb", state=None), ("ATTN_Y", "ATTN_S"))

    def test_kill_switch_disables_route(self):
        ssm = self._import()
        os.environ["MLX_OMARCHY_NO_SSM_ROUTE"] = "1"
        x = _T((1, 1, 64, 64), "bf16")
        B = _T((1, 1, 8, 128), "bf16")
        st = _T((1, 64, 64, 128), "f32")
        import mlx.core as mx
        mx.metal = type("M", (), {"is_available": staticmethod(lambda: False)})()
        self.assertEqual(ssm.ssm_update(x, "A", B, B, "D", "dt", "dtb", state=st), ("ATTN_Y", "ATTN_S"))

    def test_already_imported_module_is_wrapped(self):
        import mlx.core  # noqa: F401
        import mlx_lm.models.ssm as ssm
        import mlx._omarchy_ssm_route  # noqa: F401
        self.assertIsNotNone(getattr(ssm.ssm_update, "_omarchy_route", None))

    def test_non_multiple_of_32_ds_stays_on_ssm_attn(self):
        ssm = self._import()
        x = _T((1, 1, 64, 64), "bf16")
        B = _T((1, 1, 8, 16), "bf16")
        st = _T((1, 64, 64, 16), "f32")
        import mlx.core as mx
        mx.metal = type("M", (), {"is_available": staticmethod(lambda: False)})()
        self.assertEqual(ssm.ssm_update(x, "A", B, B, "D", "dt", "dtb", state=st), ("ATTN_Y", "ATTN_S"))


if __name__ == "__main__":
    unittest.main()
