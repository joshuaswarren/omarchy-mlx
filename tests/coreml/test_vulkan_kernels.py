# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Source checks for the Metal kernel factories in vulkan_kernels.

This host has no Metal device, so the kernel bodies are never executed.
These tests check what is real without one: each factory hands
mx.fast.metal_kernel a documented name, input and output lists, and a
source that uses exactly those buffers; the pointwise factory picks the
operator, right-hand side addressing, and stride its docstring states;
and the _pw router picks the right layout and grid for each operand
shape.
"""

import re
import sys
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "coreml"))

import vulkan_kernels as vk

ZERO_ARG_FACTORIES = [
    "_leftover_chain_kernel",
    "_leftover_chain_bias_kernel",
    "_leftover_chain_bias_silu_kernel",
    "_linear_f16_coopmat_kernel",
    "_silu_kernel",
    "_glu_kernel",
    "_ln_cast_kernel",
    "_ln_sq_kernel",
    "_ln_tail_kernel",
    "_sm_exp_kernel",
    "_sm_div_kernel",
    "_sigmoid_kernel",
]
PW_OPS = ("add", "sub", "mul")
PW_LAYOUTS = ("flat", "scalar", "row")
MAX_THREADS_X = 65535 * 256


class _RecordedKernel:
    """Stands in for the object mx.fast.metal_kernel would return."""

    def __init__(self, spec):
        self.spec = spec

    def __call__(self, **kwargs):
        raise AssertionError("kernel dispatch is not exercised on this host")


@pytest.fixture(autouse=True)
def _fresh_caches():
    for name in ZERO_ARG_FACTORIES:
        getattr(vk, name).cache_clear()
    vk._pw_kernel.cache_clear()
    yield
    for name in ZERO_ARG_FACTORIES:
        getattr(vk, name).cache_clear()
    vk._pw_kernel.cache_clear()


@pytest.fixture
def recorded(monkeypatch):
    """Patch mx.fast.metal_kernel with a recorder of its call arguments."""
    specs = []

    def fake_metal_kernel(**kwargs):
        spec = dict(kwargs)
        specs.append(spec)
        return _RecordedKernel(spec)

    monkeypatch.setattr(vk.mx.fast, "metal_kernel", fake_metal_kernel)
    return specs


def _pw_specs(recorded):
    """Call _pw_kernel once per op and layout, return the recorded specs."""
    specs = []
    for op in PW_OPS:
        for layout in PW_LAYOUTS:
            vk._pw_kernel(op, layout, 16 + len(specs), 32, 9)
            specs.append(recorded[-1])
    return specs


def test_kernel_names_inputs_outputs_and_math_mode(recorded):
    specs = [getattr(vk, name)().spec for name in ZERO_ARG_FACTORIES]
    specs += _pw_specs(recorded)
    names = [spec["name"] for spec in specs]
    assert len(names) == len(set(names))
    for spec in specs:
        source = spec["source"]
        buffers = spec["input_names"] + spec["output_names"]
        for buffer_name in buffers:
            assert buffer_name not in ("x", "y")
        for input_name in spec["input_names"]:
            assert re.search(rf"\b{re.escape(input_name)}\b", source)
        for output_name in spec["output_names"]:
            assert re.search(
                rf"\b{re.escape(output_name)}\s*\[[^\]]*\]\s*=(?!=)", source
            )
        compile_options = spec.get("compile_options")
        if compile_options:
            assert compile_options.get("math_mode") == "safe"


def test_factories_are_cached(recorded):
    for name in ZERO_ARG_FACTORIES:
        getattr(vk, name)()
        getattr(vk, name)()
    for op in PW_OPS:
        for layout in PW_LAYOUTS:
            assert vk._pw_kernel(op, layout, 16) is vk._pw_kernel(op, layout, 16)
            assert vk._pw_kernel(op, layout, 16) is not vk._pw_kernel(
                op, layout, 32
            )
    assert len(recorded) == len(ZERO_ARG_FACTORIES) + 2 * len(PW_OPS) * len(
        PW_LAYOUTS
    )


def test_pw_kernel_source_matches_op_and_layout(recorded):
    for op, symbol in (("add", "+"), ("sub", "-"), ("mul", "*")):
        vk._pw_kernel(op, "flat", 16)
        spec = recorded[-1]
        assert spec["name"] == f"encoder_pw_{op}_flat"
        assert f"half(float(lhs[index]) {symbol} float(rhs[index]))" in spec["source"]


def test_pw_kernel_row_layout_bakes_width_and_time(recorded):
    vk._pw_kernel("mul", "row", 64, width=32, time=9)
    spec = recorded[-1]
    assert spec["name"] == "encoder_pw_mul_row"
    assert "float(rhs[(index / 32u) % 9u])" in spec["source"]


def test_pw_kernel_scalar_layout_reads_rhs_bare(recorded):
    vk._pw_kernel("add", "scalar", 16)
    assert "half(float(lhs[index]) + float(rhs))" in recorded[-1]["source"]


def test_pw_kernel_stride_caps_thread_x_span(recorded):
    vk._pw_kernel("add", "flat", 100)
    assert "+ thread_position_in_grid.y * 100u" in recorded[-1]["source"]
    vk._pw_kernel("add", "flat", MAX_THREADS_X + 5)
    assert f"+ thread_position_in_grid.y * {MAX_THREADS_X}u" in recorded[-1]["source"]


@pytest.mark.parametrize(
    "op,layout", [("div", "flat"), ("add", "diag"), ("pow", "row")]
)
def test_pw_kernel_rejects_unknown_op_or_layout(op, layout):
    with pytest.raises(KeyError):
        vk._pw_kernel(op, layout, 16, 4, 2)


class _LaunchRecorder:
    """Fake _pw_kernel: records its arguments, returns a launch closure."""

    def __init__(self):
        self.builds = []
        self.launches = []

    def __call__(self, op, layout, n, width=0, time=0):
        self.builds.append((op, layout, n, width, time))
        recorder = self

        def launch(**kwargs):
            recorder.launches.append(kwargs)
            return [mx.zeros((n,), dtype=mx.float16)]

        return launch


def _run_pw(monkeypatch, x, y, op="add"):
    recorder = _LaunchRecorder()
    monkeypatch.setattr(vk, "_pw_kernel", recorder)
    out = vk._pw(op, x, y)
    return recorder, out


def test_pw_returns_none_for_non_f16_or_unmatched_shapes(monkeypatch):
    f32 = mx.zeros((2, 3), dtype=mx.float32)
    f16 = mx.zeros((2, 3), dtype=mx.float16)
    assert vk._pw("add", f32, f32) is None
    assert vk._pw("add", f16, f32) is None
    assert vk._pw("add", f16, mx.zeros((2, 4), dtype=mx.float16)) is None
    stem_x = mx.zeros((1, 2, 5, 8), dtype=mx.float16)
    stem_y = mx.zeros((1, 1, 6, 1), dtype=mx.float16)
    assert vk._pw("mul", stem_x, stem_y) is None
    mask_x = mx.zeros((1, 4, 9, 32), dtype=mx.float16)
    for bad_y_shape in [(2, 1, 9, 1), (1, 2, 9, 1), (1, 1, 9, 2)]:
        bad_y = mx.zeros(bad_y_shape, dtype=mx.float16)
        assert vk._pw("mul", mask_x, bad_y) is None


def test_pw_routes_scalar_layout_for_size_one_y(monkeypatch):
    x = mx.zeros((2, 3), dtype=mx.float16)
    y = mx.zeros((1, 1), dtype=mx.float16)
    recorder, out = _run_pw(monkeypatch, x, y)
    assert recorder.builds == [("add", "scalar", 6, 0, 0)]
    assert out.shape == x.shape


def test_pw_routes_flat_layout_for_equal_shapes(monkeypatch):
    x = mx.zeros((2, 5), dtype=mx.float16)
    recorder, out = _run_pw(monkeypatch, x, x)
    assert recorder.builds == [("add", "flat", 10, 0, 0)]
    assert out.shape == x.shape


def test_pw_routes_row_layout_for_conv_stem_mask(monkeypatch):
    x = mx.zeros((1, 4, 9, 32), dtype=mx.float16)
    y = mx.zeros((1, 1, 9, 1), dtype=mx.float16)
    recorder, out = _run_pw(monkeypatch, x, y, op="mul")
    assert recorder.builds == [("mul", "row", 1152, 32, 9)]
    assert out.shape == x.shape


def test_pw_grid_spreads_rows_past_the_x_limit(monkeypatch):
    n = MAX_THREADS_X + 5
    x = mx.zeros((n,), dtype=mx.float16)
    recorder, out = _run_pw(monkeypatch, x, x)
    assert recorder.builds == [("add", "flat", n, 0, 0)]
    launch = recorder.launches[0]
    threads_x = min(n, MAX_THREADS_X)
    threads_y = (n + threads_x - 1) // threads_x
    assert launch["grid"] == (threads_x, threads_y, 1)
    assert launch["grid"] == (MAX_THREADS_X, 2, 1)
    assert launch["threadgroup"] == (256, 1, 1)
    assert launch["output_shapes"] == [(n,)]
    assert launch["output_dtypes"] == [mx.float16]
    assert launch["stream"] is mx.gpu
    assert out.shape == x.shape


def test_pw_grid_is_one_row_for_small_tensors(monkeypatch):
    x = mx.zeros((6,), dtype=mx.float16)
    recorder, _ = _run_pw(monkeypatch, x, x)
    assert recorder.launches[0]["grid"] == (6, 1, 1)
