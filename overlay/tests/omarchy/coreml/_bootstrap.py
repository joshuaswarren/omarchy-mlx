# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Shared test bootstrap: import the coreml tools package by path."""

import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[3] / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
