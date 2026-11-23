# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""Shared fixture: the torch-mlx-vulkan backend module loaded exactly once
(it registers itself as a dynamo backend, so a second import collides)."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mlx_vulkan():
    torch = pytest.importorskip("torch")
    mx = pytest.importorskip("mlx.core")
    path = (Path(__file__).resolve().parents[1] / "tools"
            / "torch-mlx-vulkan" / "mlx_vulkan.py")
    spec = importlib.util.spec_from_file_location("mlx_vulkan_under_test",
                                                  path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
