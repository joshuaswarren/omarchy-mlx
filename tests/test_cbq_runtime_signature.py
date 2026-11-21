"""Exercise the native MLX wrapper on an explicitly selected Vulkan GPU."""
import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("CBQ_GPU_TEST") != "1",
                    reason="set CBQ_GPU_TEST=1 with the candidate MLX and a Vulkan GPU")
def test_cbq_runtime_signature_matches_cpu_reference():
    result = subprocess.run(
        [sys.executable, "-m", "tools.kernel_recheck.run", "--single", "cbq_gather_mm"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT pass maxdiff " in result.stdout
