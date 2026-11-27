"""Tests for scripts/sdpa16_ab_bench.py env wiring and census guards.

Regression for the review finding: arm A is documented as the
switch-unset arm but run_leg inherited the parent shell's
MLX_OMARCHY_SDPA_CAUSAL_FLASH, so an exported kill switch turned the
capture into fallback-vs-fallback while the ID and argmax gates still
passed. These checks observe the env a leg's child process actually
sees, and pin the dispatch-census guards that make a wrong-arm capture
fail.
"""

import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "sdpa16_ab_bench.py"

spec = importlib.util.spec_from_file_location("sdpa16_ab_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

FAKE_DRIVER = """#!/usr/bin/env python3
import json, os
print(json.dumps({
    "sdpa_causal_flash": os.environ.get("MLX_OMARCHY_SDPA_CAUSAL_FLASH"),
    "ok": True}))
"""


def _write_driver(tmp):
    path = Path(tmp) / "driver.py"
    path.write_text(FAKE_DRIVER)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


ARGS = SimpleNamespace(model="m", prompt_file="p", prefill_tokens=4,
                       decode_tokens=2)


class ArmEnvTests(unittest.TestCase):
    def test_arm_a_child_sees_switch_unset_when_parent_exported_kill(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = _write_driver(tmp)
            with patch.dict(os.environ,
                            {"MLX_OMARCHY_SDPA_CAUSAL_FLASH": "0"}):
                row = bench.run_leg(driver, {}, ARGS, None, False)
            self.assertIsNone(row["sdpa_causal_flash"])

    def test_arm_b_child_sees_kill_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = _write_driver(tmp)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MLX_OMARCHY_SDPA_CAUSAL_FLASH", None)
                row = bench.run_leg(
                    driver, {"MLX_OMARCHY_SDPA_CAUSAL_FLASH": "0"}, ARGS,
                    None, False)
            self.assertEqual(row["sdpa_causal_flash"], "0")

    def test_arm_a_unset_when_parent_exported_force_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            driver = _write_driver(tmp)
            with patch.dict(os.environ,
                            {"MLX_OMARCHY_SDPA_CAUSAL_FLASH": "1"}):
                row = bench.run_leg(driver, {}, ARGS, None, False)
            self.assertIsNone(row["sdpa_causal_flash"])


class CensusGuardTests(unittest.TestCase):
    def test_target_kernel_id_resolves(self):
        kid = bench.target_kernel_id()
        self.assertIsInstance(kid, int)
        self.assertGreaterEqual(kid, 0)

    def test_target_dispatch_counts_separates_arms(self):
        # A rows keyed by int (in-memory) and str (json-loaded) both count.
        results = [
            {"arm": "A", "dispatch_events": {770: 3}},
            {"arm": "B", "dispatch_events": {"770": 0, "12": 5}},
            {"arm": "B", "dispatch_events": {"770": 1}},
            {"arm": "A", "dispatch_events": {}},
        ]
        a, b = bench.target_dispatch_counts(results, 770)
        self.assertEqual(a, 3)
        self.assertEqual(b, 1)


if __name__ == "__main__":
    unittest.main()
