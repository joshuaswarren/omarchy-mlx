"""Compatibility shim for the PYTHONPATH=tools/torch-mlx-vulkan flow.

The backend lives in the wheel as mlx.omarchy_torch (registered on import).
This shim keeps `import mlx_vulkan` working for the tools scripts on hosts
with the omarchy wheel installed.
"""
from __future__ import annotations

from mlx.omarchy_torch import *  # noqa: F401,F403
from mlx.omarchy_torch import log  # noqa: F401
