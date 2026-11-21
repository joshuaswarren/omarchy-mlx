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


class RecordVerifiedMissingStampTests(unittest.TestCase):
    def test_record_verified_missing_stamp_warns_but_runs(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            None, "0.32.4.dev+ef7923c374", record_verified=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn("RECORD", warning)

    def test_record_verified_does_not_rescue_a_stale_stamp(self):
        # A present-but-different stamp still refuses: the wheel may have
        # moved on from what the copies were stamped with.
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev+ef7923c374", record_verified=True)
        self.assertIsNotNone(refusal)
        self.assertIsNone(warning)

    def test_override_beats_record_verification_in_wording(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            None, "0.32.4.dev+ef7923c374", drift_allowed=True,
            record_verified=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)


class _FakeHash:
    def __init__(self, digest):
        self._digest = digest

    @property
    def name(self):
        return "sha256"

    def hexdigest(self):
        return self._digest.hex()

    def digest(self):
        return self._digest


class _FakeEntry:
    def __init__(self, path, digest):
        self._path = path
        self.hash = _FakeHash(digest)

    def __str__(self):
        return self._path


class VerifyAgainstRecordTests(unittest.TestCase):
    """verify_serve_copies_against_record against a faked installed wheel."""

    def setUp(self):
        import hashlib
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "mlx_omarchy_serve"
        self.root.mkdir()
        self.content = {
            "__init__.py": "x = 1\n",
            "_mlxlm_server.py": "print('serve')\n",
        }
        for name, text in self.content.items():
            (self.root / name).write_text(text)
        self.entries = [
            _FakeEntry(f"mlx_omarchy_serve/{name}",
                       hashlib.sha256(text.encode()).digest())
            for name, text in self.content.items()
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def _with_files(self, entries):
        import unittest.mock
        return unittest.mock.patch(
            "importlib.metadata.files", return_value=entries)

    def test_all_copies_match_the_record(self):
        with self._with_files(self.entries):
            self.assertTrue(
                shim.verify_serve_copies_against_record(self.root))

    def test_one_divergent_copy_fails_verification(self):
        (self.root / "_mlxlm_server.py").write_text("print('tampered')\n")
        with self._with_files(self.entries):
            self.assertFalse(
                shim.verify_serve_copies_against_record(self.root))

    def test_copy_absent_from_the_record_fails_verification(self):
        (self.root / "extra.py").write_text("print('extra')\n")
        with self._with_files(self.entries):
            self.assertFalse(
                shim.verify_serve_copies_against_record(self.root))

    def test_no_installed_wheel_fails_verification(self):
        with self._with_files(None):
            self.assertFalse(
                shim.verify_serve_copies_against_record(self.root))


if __name__ == "__main__":
    unittest.main()
