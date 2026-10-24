"""Run the real sparse-expert shader on a CPU Vulkan device, without MLX or weights.

MLX_OMARCHY_SHADER_TESTS=1 python3 -m unittest discover -s tests -p test_gather_qmm_tile.py -v
Requires g++, glslc, spirv-val, Vulkan headers/loader and the llvmpipe ICD.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "overlay/mlx/backend/omarchy"
SHADER = BACKEND / "shaders/gather_qmm_tile.comp"
HARNESS = ROOT / "overlay/tests/omarchy/test_gather_qmm_tile_standalone.cpp"


class GatherQmmTileTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MLX_OMARCHY_SHADER_TESTS") == "1",
                         "set MLX_OMARCHY_SHADER_TESTS=1 for the CPU Vulkan check")
    def test_direct_expert_parameters_and_offset_views(self):
        with tempfile.TemporaryDirectory(prefix="gather-qmm-test-") as directory:
            binary = Path(directory) / "gather-qmm-test"
            subprocess.run(
                ["g++", "-std=c++20", "-O1", "-I", str(ROOT / "overlay"),
                 str(HARNESS), "-lvulkan", "-o", str(binary)],
                check=True, timeout=60,
            )
            print("[provenance] standalone shader test; libmlx=not-loaded", flush=True)
            for path in (HARNESS, SHADER, BACKEND / "compute.h"):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                print(f"[provenance] {path.relative_to(ROOT)} sha256={digest}", flush=True)
            environment = dict(os.environ, LP_NUM_THREADS="2")
            environment["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
            for dtype in ("bf16", "fp16"):
                with self.subTest(dtype=dtype):
                    spirv = Path(directory) / f"tile-{dtype}.spv"
                    subprocess.run(
                        ["glslc", "--target-env=vulkan1.2", f"-DUSE_{dtype.upper()}=1",
                         str(SHADER), "-o", str(spirv)],
                        check=True, timeout=30,
                    )
                    subprocess.run(
                        ["spirv-val", "--target-env", "vulkan1.2", str(spirv)],
                        check=True, timeout=15,
                    )
                    print(f"[provenance] {dtype} spirv_sha256="
                          f"{hashlib.sha256(spirv.read_bytes()).hexdigest()}", flush=True)
                    subprocess.run(
                        [str(binary), str(spirv), dtype],
                        env=environment, check=True, timeout=60,
                    )


if __name__ == "__main__":
    unittest.main()
