"""The serve copies must match the installed mlx-omarchy wheel.

The venv carries serve/ as unpackaged copies (install.sh SERVE_PACKAGES),
so a wheel-only reinstall leaves an old shim running against a new wheel
without a word. The shim must refuse to start (loud, override-able) when
its build stamp disagrees with the wheel's version, and install.sh must
write that stamp.
"""

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import _mlxlm_server as shim  # noqa: E402


class WheelCommitTests(unittest.TestCase):
    def test_local_version_segment_wins(self):
        self.assertEqual(
            shim._wheel_commit("0.32.4.dev202611170306+ef7923c374"),
            "ef7923c374")

    def test_release_tag(self):
        self.assertEqual(shim._wheel_commit("0.7.38"), "0.7.38")

    def test_missing_version(self):
        self.assertIsNone(shim._wheel_commit(None))
        self.assertIsNone(shim._wheel_commit(""))


class ResolveStampMismatchTests(unittest.TestCase):
    def test_commit_stamp_matches_wheel_local_version(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "ef7923c374", "0.32.4.dev202611170306+ef7923c374")
        self.assertIsNone(refusal)
        self.assertIsNone(warning)

    def test_release_tag_matches_release_wheel(self):
        refusal, warning = shim.resolve_stamp_mismatch("v0.7.38", "0.7.38")
        self.assertIsNone(refusal)
        self.assertIsNone(warning)

    def test_missing_stamp_refuses(self):
        refusal, warning = shim.resolve_stamp_mismatch(None, "0.32.4.dev+ef7923c374")
        self.assertIsNotNone(refusal)
        self.assertIn("build stamp", refusal)
        self.assertIsNone(warning)

    def test_missing_wheel_with_stamp_refuses(self):
        refusal, warning = shim.resolve_stamp_mismatch("ef7923c374", None)
        self.assertIsNotNone(refusal)
        self.assertIn("wheel version", refusal)

    def test_no_stamp_and_no_wheel_warns_but_runs(self):
        # Dev from a checkout: nothing on either side to compare.
        refusal, warning = shim.resolve_stamp_mismatch(None, None)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn("cannot run", warning)

    def test_override_covers_a_missing_wheel(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "ef7923c374", None, drift_allowed=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)

    def test_override_covers_a_missing_stamp(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            None, "0.32.4.dev+ef7923c374", drift_allowed=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)

    def test_checkout_import_only_warns_on_mismatch(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev202611170306+ef7923c374", installed=False)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn("installed copies", warning)

    def test_checkout_import_only_warns_on_missing_wheel(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "ef7923c374", None, installed=False)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)

    def test_mismatch_refuses_and_names_both_sides(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev202611170306+ef7923c374")
        self.assertIsNotNone(refusal)
        self.assertIn("7ec5347", refusal)
        self.assertIn("ef7923c374", refusal)
        self.assertIsNone(warning)

    def test_drift_override_warns_instead_of_refusing(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev202611170306+ef7923c374", drift_allowed=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)


class ReadStampTests(unittest.TestCase):
    def test_reads_the_stamp_file_next_to_the_module(self):
        with tempfile.TemporaryDirectory() as d:
            stamp = Path(d) / shim.STAMP_FILE
            stamp.write_text("ef7923c374\n")
            self.assertEqual(
                shim.read_build_stamp(directory=Path(d)), "ef7923c374")

    def test_missing_file_reads_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(shim.read_build_stamp(directory=Path(d)))


if __name__ == "__main__":
    unittest.main()
