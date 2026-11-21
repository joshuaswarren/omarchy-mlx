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


class VerifyAgainstRecordTests(unittest.TestCase):
    """verify_serve_copies_against_record against a real distribution.

    A temp venv-style wheel install exercises importlib.metadata for
    real: a fake hash object once hid an AttributeError the reviewer
    found, because importlib.metadata exposes hash.mode and hash.value,
    not the hashlib methods a fake can carry.
    """

    def setUp(self):
        import base64
        import hashlib
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.root = root / "site-packages" / "mlx_omarchy_serve"
        self.root.mkdir(parents=True)
        self.content = {
            "__init__.py": "x = 1\n",
            "_mlxlm_server.py": "print('serve')\n",
        }
        for name, text in self.content.items():
            (self.root / name).write_text(text)
        distinfo = root / "site-packages" / (
            "mlx_omarchy-0.32.4.dev202611210000+ef7923c374.dist-info")
        distinfo.mkdir()
        # PEP 376 RECORD: csv rows of path, urlsafe-b64-nopad sha256, size.
        rows = [["RECORD"]]
        for name, text in self.content.items():
            path = f"mlx_omarchy_serve/{name}"
            digest = hashlib.sha256(text.encode()).digest()
            encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            rows.append([path, f"sha256={encoded}", str(len(text))])
        (distinfo / "RECORD").write_text(
            "".join(",".join(row) + "\n" for row in rows), encoding="utf-8")
        self.distinfo = distinfo

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_copies_match_a_real_record(self):
        import importlib.metadata
        import unittest.mock
        # Point metadata's site dir scan at the temp tree by patching
        # distribution discovery with a real PathDistribution.
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "files",
                return_value=dist.files):
            self.assertTrue(
                shim.verify_serve_copies_against_record(self.root))

    def test_one_divergent_copy_fails_verification(self):
        (self.root / "_mlxlm_server.py").write_text("print('tampered')\n")
        self._run_expect(False)

    def test_copy_absent_from_the_record_fails_verification(self):
        (self.root / "extra.py").write_text("print('extra')\n")
        self._run_expect(False)

    def test_non_sha256_record_value_fails_closed(self):
        import unittest.mock
        import importlib.metadata
        # Rewrite RECORD with an md5 value; the verifier must refuse
        # rather than compare, and refuse without crashing.
        record = self.distinfo / "RECORD"
        text = record.read_text(encoding="utf-8")
        record.write_text(
            text.replace("sha256=", "md5="), encoding="utf-8")
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "files",
                return_value=dist.files):
            self.assertFalse(
                shim.verify_serve_copies_against_record(self.root))

    def test_malformed_record_value_fails_closed(self):
        import importlib.metadata
        import unittest.mock
        record = self.distinfo / "RECORD"
        record.write_text(
            "mlx_omarchy_serve/__init__.py,sha256=!!,3\n"
            "mlx_omarchy_serve/_mlxlm_server.py,sha256=!!,15\n",
            encoding="utf-8")
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "files",
                return_value=dist.files):
            self.assertFalse(
                shim.verify_serve_copies_against_record(self.root))

    def _run_expect(self, expected):
        import importlib.metadata
        import unittest.mock
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "files",
                return_value=dist.files):
            self.assertEqual(
                shim.verify_serve_copies_against_record(self.root),
                expected)


if __name__ == "__main__":
    unittest.main()
