# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Capability-simulation evidence gate stays closed.

Under MLX_OMARCHY_CAPS_SIM the evidence-producing tools must refuse
with a named error and a nonzero exit; without the variable they must
not refuse. docs/new-chip-bringup.md records why: a simulated run is a
dispatch instrument, never a source of benchmark or generated-id
digest evidence.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
GUARD_TEXT = "Capability simulation never produces"

class GuardRefusalTest(unittest.TestCase):
    def test_bench_decode_refuses_under_simulation(self):
        env = dict(os.environ)
        env["MLX_OMARCHY_CAPS_SIM"] = "m1-honeykrisp-fork"
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "bench_decode.py"), "--tokens", "4"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 3)
        self.assertIn("MLX_OMARCHY_CAPS_SIM='m1-honeykrisp-fork'",
                      proc.stderr)
        self.assertIn(GUARD_TEXT, proc.stderr)

    def test_bench_decode_refusal_names_the_profile_and_is_fatal(self):
        env = dict(os.environ)
        env["MLX_OMARCHY_CAPS_SIM"] = "no-such-profile"
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "bench_decode.py"), "--tokens", "4"],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        # The guard refuses on ANY non-empty value, including unknown
        # names: the backend would refuse the unknown profile too, so
        # producing evidence is impossible either way.
        self.assertEqual(proc.returncode, 3)
        self.assertIn("no-such-profile", proc.stderr)


if __name__ == "__main__":
    unittest.main()
