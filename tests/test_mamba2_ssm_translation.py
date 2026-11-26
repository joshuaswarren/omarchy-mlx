"""Compile-only acceptance for the Mamba-2 / Nemotron-H ssm_kernel hand port.

The shipped mlx_lm/models/ssm.py kernel uses integer `auto` locals, which the
generic translator refuses by name. The named port in custom_kernel_ssm.h
matches the exact body, ABI, and launch shape and emits Vulkan 1.3 GLSL; any
body mutation must fall through to the generic refusal, and any off-contract
ABI or launch must refuse by name.
"""
import os
import subprocess

import pytest

from tools.kernel_recheck.defs import get_spec
from tools.kernel_recheck.dump_msl import generated_msl

NAME = "ssm_kernel"
SPEC = get_spec(NAME)


@pytest.fixture(scope="module")
def translator():
    binary = os.environ.get("CBQ_TRANSLATE_DUMP")
    if not binary:
        pytest.skip("set CBQ_TRANSLATE_DUMP to omarchy_custom_kernel_translate_dump")
    return binary


def translate(translator, tmp_path, source, *, grid=None, threadgroup=None,
              outputs=2, tag="case"):
    path = tmp_path / f"{tag}.msl"
    path.write_text(source)
    launch = (*(grid or SPEC["grid"]), *(threadgroup or SPEC["threadgroup"]), outputs)
    return subprocess.run(
        [translator, str(path), *map(str, launch)],
        capture_output=True, text=True, timeout=20, check=False,
    )


def test_ssm_fixture_signature_matches_runtime_abi():
    msl = generated_msl(NAME)
    assert "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]" in msl
    assert "  uint3 thread_position_in_grid [[thread_position_in_grid]]" in msl
    assert "  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]" in msl


def test_ssm_translates_and_compiles(translator, tmp_path):
    result = translate(translator, tmp_path, generated_msl(NAME), tag="ok")
    assert result.returncode == 0, result.stdout + result.stderr
    glsl = tmp_path / "ssm.comp"
    glsl.write_text(result.stdout)
    spirv = tmp_path / "ssm.spv"
    subprocess.run([
        os.environ.get("GLSLC", "glslc"), "--target-env=vulkan1.3", "-O",
        str(glsl), "-o", str(spirv),
    ], check=True, timeout=30)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(spirv)],
                   check=True, timeout=30)


@pytest.mark.parametrize("t_type,u_type", [
    ("float", "float"),
    ("float16_t", "float"),
    ("half", "float"),
    ("bfloat16_t", "bfloat16_t"),
    ("float16_t", "float16_t"),
])
def test_ssm_dtype_variants_compile_and_validate(translator, tmp_path, t_type, u_type):
    msl = generated_msl(NAME)
    msl = msl.replace("bfloat16_t, float, 64, 128, 64, 8", f"{t_type}, {u_type}, 64, 128, 64, 8")
    msl = msl.replace("const device bfloat16_t* X", f"const device {t_type}* X")
    msl = msl.replace("const device bfloat16_t* B", f"const device {t_type}* B")
    msl = msl.replace("const device bfloat16_t* C", f"const device {t_type}* C")
    msl = msl.replace("device bfloat16_t* out", f"device {t_type}* out")
    msl = msl.replace("const device float* state_in", f"const device {u_type}* state_in")
    msl = msl.replace("device float* state_out", f"device {u_type}* state_out")
    result = translate(translator, tmp_path, msl, tag=f"dt_{t_type}_{u_type}")
    assert result.returncode == 0, result.stdout + result.stderr
    glsl = tmp_path / f"ssm_{t_type}_{u_type}.comp"
    glsl.write_text(result.stdout)
    spirv = tmp_path / f"ssm_{t_type}_{u_type}.spv"
    subprocess.run([
        os.environ.get("GLSLC", "glslc"), "--target-env=vulkan1.3", "-O",
        str(glsl), "-o", str(spirv),
    ], check=True, timeout=30)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(spirv)],
                   check=True, timeout=30)


def test_ssm_constant_space_small_heads_and_trailing_indent(translator, tmp_path):
    # Small H < 8 makes MLX bind A_log/D/dt as `const constant float*`, and
    # mlx_lm's verbatim triple-quoted source has trailing spaces before `"""`.
    msl = generated_msl(NAME)
    msl = msl.replace("64, 128, 64, 8", "64, 128, 4, 2")
    msl = msl.replace("const device float* A_log", "const constant float* A_log")
    msl = msl.replace("const device float* D", "const constant float* D")
    msl = msl.replace("const device float* dt", "const constant float* dt")
    msl = msl.replace("        }\n}", "        }\n    \n}")
    result = translate(translator, tmp_path, msl, grid=(32, 64, 4),
                       threadgroup=(32, 8, 1), tag="small_const")
    assert result.returncode == 0, result.stdout + result.stderr
    glsl = tmp_path / "ssm_small_const.comp"
    glsl.write_text(result.stdout)
    spirv = tmp_path / "ssm_small_const.spv"
    subprocess.run([
        os.environ.get("GLSLC", "glslc"), "--target-env=vulkan1.3", "-O",
        str(glsl), "-o", str(spirv),
    ], check=True, timeout=30)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(spirv)],
                   check=True, timeout=30)


def test_ssm_body_change_falls_back_to_generic_refusal(translator, tmp_path):
    source = generated_msl(NAME).replace("acc = simd_sum(acc);", "acc = simd_sum(acc) + 0.0;")
    result = translate(translator, tmp_path, source, tag="mut")
    assert result.returncode != 0
    assert "Mamba-2 SSM kernel ABI" not in result.stdout


@pytest.mark.parametrize("swap", [
    ("const device bfloat16_t* X", "const device float* X"),
    ("device float* state_out", "device bfloat16_t* state_out"),
    ("uint3 thread_position_in_grid", "uint thread_position_in_grid"),
])
def test_ssm_wrong_abi_refuses_by_name(translator, tmp_path, swap):
    source = generated_msl(NAME).replace(*swap)
    result = translate(translator, tmp_path, source, tag="abi")
    assert result.returncode != 0
    assert "Mamba-2 SSM kernel ABI or launch" in result.stdout


@pytest.mark.parametrize("grid,tg,ok", [
    ((32, 64, 64), (32, 8, 1), True),    # batch=1 Muse-Glimmer
    ((32, 64, 128), (32, 8, 1), True),   # batch=2 Muse-Glimmer
    ((16, 64, 64), (32, 8, 1), False),   # wrong grid_x
    ((32, 64, 63), (32, 8, 1), False),   # grid_z not multiple of H
    ((32, 64, 64), (16, 8, 1), False),   # wrong threadgroup_x
])
def test_ssm_launch_contract(translator, tmp_path, grid, tg, ok):
    result = translate(translator, tmp_path, generated_msl(NAME),
                       grid=grid, threadgroup=tg, tag="launch")
    if ok:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
        assert "Mamba-2 SSM kernel ABI or launch" in result.stdout
