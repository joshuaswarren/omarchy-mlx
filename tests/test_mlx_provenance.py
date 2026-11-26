"""Tests for the installed-binary provenance gate (mlx_provenance.py).

Stdlib unittest only, no network, no mlx import: the hash-comparison
core runs against a fake install tree built in a temp directory.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mlx_provenance as prov  # noqa: E402


class ProvenanceSelfTest(unittest.TestCase):
    def test_self_test_covers_the_comparison_core(self):
        # _self_test builds a fake site-packages and proves: matching
        # tree verifies clean, swapped binary refuses, compiled-vs-dist
        # version disagreement refuses, positive control detects a
        # foreign library, and --expect-wheel compares members.
        self.assertIsNone(prov._self_test())


class ProvenanceLineTests(unittest.TestCase):
    def test_line_carries_version_and_hashes(self):
        line = prov.provenance_line({
            "dist": "mlx-omarchy", "dist_version": "1.0",
            "mx_version": "1.0", "verified": "match",
            "files": [{"path": "mlx/lib/libmlx.so",
                       "sha256": "a" * 64}],
        })
        self.assertIn("mlx-omarchy 1.0", line)
        self.assertIn("verified=match", line)
        self.assertIn("libmlx.so=sha256:" + "a" * 16, line)

    def test_control_string_is_the_baked_in_literal(self):
        self.assertEqual(prov.CONTROL_STRING, b"MLX_DISABLE_COMPILE")


if __name__ == "__main__":
    unittest.main()
