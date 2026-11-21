"""Compile-only acceptance for the nemotron BM32 steel hand port.

The shipped mlx_vlm kernel wraps Steel's GEMMKernel, which the generic
translator refuses (`simdgroup_matrix`). The named port in
custom_kernel_bm32.h matches the exact body, ABI and launch and emits a
GLSL 32x64 BF16 tile; anything off-contract must refuse by name rather
than drift into the generic path.
"""
import os
import subprocess
from pathlib import Path

import pytest

from tools.kernel_recheck.defs import get_spec
from tools.kernel_recheck.dump_msl import generated_msl

NAME = "nemotron_bm32_steel_linear_nt"
SPEC = get_spec(NAME)


@pytest.fixture(scope="module")
def translator():
    binary = os.environ.get("CBQ_TRANSLATE_DUMP")
    if not binary:
        pytest.skip("set CBQ_TRANSLATE_DUMP to omarchy_custom_kernel_translate_dump")
    return binary


def translate(translator, tmp_path, source, *, grid=None, threadgroup=None, tag="case"):
    path = tmp_path / f"{tag}.msl"
    path.write_text(source)
    launch = (*(grid or SPEC["grid"]), *(threadgroup or SPEC["threadgroup"]), 1)
    return subprocess.run(
        [translator, str(path), *map(str, launch)],
        capture_output=True, text=True, timeout=20, check=False,
    )


def test_bm32_translates_and_compiles(translator, tmp_path):
    result = translate(translator, tmp_path, generated_msl(NAME), tag="ok")
    assert result.returncode == 0, result.stdout + result.stderr
    glsl = tmp_path / "bm32.comp"
    glsl.write_text(result.stdout)
    spirv = tmp_path / "bm32.spv"
    subprocess.run([
        os.environ.get("GLSLC", "glslc"), "--target-env=vulkan1.3", "-O",
        str(glsl), "-o", str(spirv),
    ], check=True, timeout=30)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(spirv)],
                   check=True, timeout=30)


def test_bm32_body_change_falls_back_to_generic_refusal(translator, tmp_path):
    source = generated_msl(NAME).replace(
        "mma_op.store_result(D, O);", "mma_op.store_result(D, 0);")
    result = translate(translator, tmp_path, source, tag="mut")
    assert result.returncode != 0
    # Not the named port: a generic-path refusal over the steel header.
    assert "BM32 kernel ABI" not in result.stdout


@pytest.mark.parametrize("swap", [
    ("const device bfloat16_t* weight", "const device float* weight"),
    ("device bfloat16_t* out", "device float* out"),
    ("uint3 threadgroup_position_in_grid", "uint3 thread_position_in_grid"),
])
def test_bm32_wrong_abi_refuses_by_name(translator, tmp_path, swap):
    source = generated_msl(NAME).replace(*swap)
    result = translate(translator, tmp_path, source, tag="abi")
    assert result.returncode != 0
    assert "BM32 kernel ABI or launch" in result.stdout


@pytest.mark.parametrize("grid,tg", [
    ((512, 1, 1), (128, 1, 1)),   # half the output tiles
    ((1024, 2, 1), (128, 1, 1)),  # two row tiles (rows=64) is legal
    ((1024, 1, 1), (64, 1, 1)),   # wrong threadgroup
])
def test_bm32_launch_contract(translator, tmp_path, grid, tg):
    result = translate(translator, tmp_path, generated_msl(NAME),
                       grid=grid, threadgroup=tg, tag="launch")
    if tg != (128, 1, 1) or grid == (512, 1, 1):
        assert result.returncode != 0
        assert "BM32 kernel ABI or launch" in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
