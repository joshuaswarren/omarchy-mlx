# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Refusal gate for evidence-producing tools under capability simulation.

MLX_OMARCHY_CAPS_SIM selects a simulated capability profile
(docs/new-chip-bringup.md). A simulated run is a dispatch-decision and
correctness instrument only: benchmark rates and generated-id digests
recorded from it would be mistaken for hardware results, so every tool
that produces such evidence refuses through refuse_if_simulated().
Provenance output (mlx.device_info, mlx-omarchy-info) stamps the
simulation instead of refusing.
"""

import os
import sys


def refuse_if_simulated(tool):
    profile = os.environ.get("MLX_OMARCHY_CAPS_SIM", "")
    if profile:
        sys.stderr.write(
            f"[{tool}] REFUSING: MLX_OMARCHY_CAPS_SIM='{profile}' is"
            " active. Capability simulation never produces benchmark or"
            " generated-id-digest evidence; unset the variable for a"
            " hardware run.\n"
        )
        raise SystemExit(3)
