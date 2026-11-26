# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for section-39 eligibility and the sections 36-37 API."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.eligibility import (
    BOUNDARY,
    SUPPORTED_PENDING_RELEASE,
    LOWERABLE_VIA_FRONTEND,
    NEEDS_COMPILER_OP,
    SUPPORTED,
    eligibility_report,
)
from coreml.model import (
    ComputeTargetUnsupported,
    CoreMLError,
    CoreMLModel,
    PackageNotEligible,
)

_HF_REVISION = "b650695c2322ee5281dff48d7345b2f3a58ff018"
_ENCODER = (
    "mweinbach1/parakeet-tdt-0.6b-v3-coreml" f"/{_HF_REVISION}"
    "/encoder.mlpackage"
)
_CLI = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "mlx-omarchy-coreml"
    / "mlx_omarchy_coreml.py"
)


def _cache() -> Path | None:
    base = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    roots = []
    if base:
        roots.append(Path(base) / "parakeet-reference")
    roots.append(Path.home() / ".cache" / "mlx-omarchy" / "parakeet-reference")
    for root in roots:
        candidate = root / _ENCODER
        if candidate.is_dir():
            return candidate
    return None


class EligibilityTest(unittest.TestCase):
    def test_pinned_encoder_reproduces_the_known_classification(self):
        cache = _cache()
        if cache is None:
            self.skipTest("parakeet reference cache not present")
        report = eligibility_report(cache)
        self.assertEqual(report["total_ops"], 3351)
        self.assertEqual(report["counts"][SUPPORTED], 2866)
        self.assertEqual(report["counts"][LOWERABLE_VIA_FRONTEND], 245)
        self.assertEqual(report["counts"][NEEDS_COMPILER_OP], 195)
        self.assertEqual(
            report["counts"][SUPPORTED_PENDING_RELEASE], 34
        )
        self.assertEqual(report["counts"][BOUNDARY], 11)
        self.assertEqual(sum(report["counts"].values()), 3351)
        blocking = {
            entry["op"]: entry["count"] for entry in report["blocking_ops"]
        }
        self.assertEqual(
            blocking,
            {
                "transpose": 146,
                "slice_by_index": 48,
                "tile": 1,
            },
        )
        # less/floor/floor_div and the -inf select family are
        # implemented on the compiler branch (registry entries + bool
        # surfaces) but not in the pinned release: pending, not blocking.
        pending = {
            entry["op"]: entry["count"]
            for entry in report["ops"]
            if entry["disposition"] == SUPPORTED_PENDING_RELEASE
        }
        self.assertEqual(
            pending,
            {"select": 24, "less": 4, "floor": 3, "floor_div": 3},
        )
        reasons = " ".join(
            entry["reason"].replace("\n", " ")
            for entry in report["ops"]
            if entry["disposition"] == SUPPORTED_PENDING_RELEASE
        )
        self.assertIn("b4f4da9", reasons)
        self.assertIn("83a4434", reasons)
        # Post-depalettize delta: palettized weights fold into consts.
        post = report["post_frontend_histogram"]
        self.assertNotIn("constexpr_lut_to_dense", post)
        self.assertEqual(post["const"], 1977)
        # Depalettize folds into consts (sum-neutral); the 24 pads
        # fold into identity convs (sum-neutral); the 24 finite-fill
        # selects, logical_and/not and reduce_min drop out fully; the
        # 24 -inf selects persist until the compiler op exists.
        self.assertEqual(sum(post.values()), 3351 - 27)
        self.assertEqual(post.get("select"), 24)
        self.assertNotIn("logical_and", post)
        self.assertNotIn("pad", post)
        self.assertEqual(post["conv"], 101)
        self.assertNotIn("reduce_min", post)


class ModelApiTest(unittest.TestCase):
    def test_pinned_encoder_target_semantics(self):
        cache = _cache()
        if cache is None:
            self.skipTest("parakeet reference cache not present")
        model = CoreMLModel.load(cache)
        eligibility = model.eligibility()
        self.assertEqual(eligibility.total_ops, 3351)
        self.assertFalse(eligibility.eligible)

        with self.assertRaises(PackageNotEligible) as caught:
            model.check_compute_target("ane")
        message = str(caught.exception)
        self.assertIn("blocked by 3 op classes", message)
        self.assertIn("transpose x146", message)

        with self.assertRaises(ComputeTargetUnsupported) as gpu:
            model.check_compute_target("gpu")
        self.assertIn("not wired", str(gpu.exception))

        with self.assertRaises(ComputeTargetUnsupported) as cpu:
            model.check_compute_target("cpu")
        self.assertIn("section 3.3", str(cpu.exception))

        with self.assertRaises(ComputeTargetUnsupported):
            model.check_compute_target("tpu")

    def test_missing_package_is_named(self):
        with self.assertRaises(CoreMLError) as caught:
            CoreMLModel.load("/nonexistent/model.mlpackage")
        self.assertIn("not a directory", str(caught.exception))


class CliTest(unittest.TestCase):
    def test_inspect_and_check_on_pinned_encoder(self):
        cache = _cache()
        if cache is None:
            self.skipTest("parakeet reference cache not present")
        inspect = subprocess.run(
            [sys.executable, str(_CLI), "inspect", str(cache), "--json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(inspect.returncode, 0, inspect.stderr)
        report = json.loads(inspect.stdout)
        self.assertEqual(report["counts"][SUPPORTED], 2866)
        self.assertEqual(report["counts"][NEEDS_COMPILER_OP], 195)
        self.assertEqual(
            report["counts"][SUPPORTED_PENDING_RELEASE], 34
        )

        check = subprocess.run(
            [
                sys.executable,
                str(_CLI),
                "check",
                str(cache),
                "--compute-target",
                "ane",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(check.returncode, 4, check.stderr)
        self.assertIn("REFUSED", check.stdout)
        self.assertIn("transpose x146", check.stdout)

        check_cpu = subprocess.run(
            [
                sys.executable,
                str(_CLI),
                "check",
                str(cache),
                "--compute-target",
                "cpu",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(check_cpu.returncode, 4)
        self.assertIn("section 3.3", check_cpu.stdout)


if __name__ == "__main__":
    unittest.main()
