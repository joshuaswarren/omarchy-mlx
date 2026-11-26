import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class UpstreamSuiteTests(unittest.TestCase):
    def test_mutually_exclusive_phases_cannot_report_success(self):
        runner = Path(__file__).resolve().parents[1] / "tools/run-upstream-suite.sh"
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", str(runner), "--cpp-only", "--py-only"],
                env={**os.environ, "OUT_DIR": directory},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_loaded_wheel_requires_matching_verified_source(self):
        runner = Path(__file__).resolve().parents[1] / "tools/run-upstream-suite.sh"
        guard = runner.read_text().split("<<'PYPIN' || exit 1\n", 1)[1].split("\nPYPIN", 1)[0]
        commit = "123456789abcdef"
        cases = [
            ({"verified": "match", "dist_version": "0.32.2.dev1+1234567"}, True),
            ({"verified": "match", "dist_version": "0.32.2.dev1+diag.1234567"}, True),
            ({"verified": "match", "dist_version": "0.32.2.dev1+abcdef0"}, False),
            ({"verified": "match", "dist_version": "0.32.2"}, False),
            ({"verified": "unverified", "dist_version": "0.32.2+1234567"}, False),
            ({"verified": "mismatch", "dist_version": "0.32.2+1234567"}, False),
            ({}, False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "provenance.json"
            for provenance, expected in cases:
                with self.subTest(provenance=provenance):
                    receipt.write_text(json.dumps(provenance))
                    result = subprocess.run(
                        [sys.executable, "-c", guard, str(receipt), commit],
                        capture_output=True, text=True, timeout=10, check=False,
                    )
                    self.assertEqual(result.returncode == 0, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
