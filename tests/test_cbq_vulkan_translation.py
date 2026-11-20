"""Compile-only acceptance for the nine unchanged CBQ inventory bodies."""
import os
import subprocess
from pathlib import Path

import pytest

from tools.kernel_recheck.defs import SPECS
from tools.kernel_recheck.dump_msl import generated_msl

CBQ = {spec["name"]: spec for spec in SPECS if spec["name"].startswith("cbq_")}


@pytest.fixture(scope="module")
def translator():
    binary = os.environ.get("CBQ_TRANSLATE_DUMP")
    if not binary:
        pytest.skip("set CBQ_TRANSLATE_DUMP to omarchy_custom_kernel_translate_dump")
    return binary


def translate(translator, tmp_path, name, source, *, threadgroup=None):
    spec = CBQ[name]
    path = tmp_path / f"{name}.msl"
    path.write_text(source)
    launch = (*spec["grid"], *(threadgroup or spec["threadgroup"]), len(spec["outputs"]))
    return subprocess.run(
        [translator, str(path), *map(str, launch)],
        capture_output=True, text=True, timeout=20, check=False,
    )


@pytest.mark.parametrize("name", CBQ)
def test_released_cbq_translates_and_compiles(translator, tmp_path, name):
    result = translate(translator, tmp_path, name, generated_msl(name))
    assert result.returncode == 0, result.stdout + result.stderr
    glsl = tmp_path / f"{name}.comp"
    glsl.write_text(result.stdout)
    spirv = tmp_path / f"{name}.spv"
    subprocess.run([
        os.environ.get("GLSLC", "glslc"), "--target-env=vulkan1.3", "-O",
        str(glsl), "-o", str(spirv),
    ], check=True, timeout=30)
    subprocess.run(["spirv-val", "--target-env", "vulkan1.3", str(spirv)],
                   check=True, timeout=30)
    module = subprocess.run(["spirv-dis", str(spirv)], check=True,
                            capture_output=True, text=True, timeout=30).stdout
    capabilities = {line.split()[-1] for line in module.splitlines()
                    if line.strip().startswith("OpCapability ")}
    assert not capabilities.intersection(
        {"Int8", "Int64", "Float64", "StorageBuffer8BitAccess"})
    assert "OpAtomic" not in module


def test_cbq_near_match_keeps_named_refusal(translator, tmp_path):
    name = "cbq_gather_mm_v3"
    source = generated_msl(name).replace("-0.125f : 0.125f", "-0.25f : 0.25f")
    result = translate(translator, tmp_path, name, source)
    assert result.returncode != 0
    assert "as_type<char4>" in result.stdout


@pytest.mark.parametrize("mutation", ["dtype", "atomic", "layout", "launch", "header"])
def test_cbq_exact_body_rejects_unsupported_contract(translator, tmp_path, mutation):
    name = "cbq_grad_d"
    source = generated_msl(name)
    tg = None
    if mutation == "dtype":
        source = source.replace("uint8_t* cb_qs", "int8_t* cb_qs")
    elif mutation == "atomic":
        source = source.replace("atomic<float>* grad_d", "float* grad_d")
    elif mutation == "layout":
        source = source.replace("uint8_t* cb_qs", "uint8_t& cb_qs")
    elif mutation == "launch":
        tg = (128, 1, 1)
    else:
        source = "#define cot another_buffer\n" + source
    result = translate(translator, tmp_path, name, source, threadgroup=tg)
    assert result.returncode != 0
    assert "CBQ kernel ABI or launch" in result.stdout
