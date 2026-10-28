import hashlib
import runpy
import tempfile
import unittest
import zipfile
from pathlib import Path

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/verify-release-assets.py")
)
VERIFY = MODULE["verify"]


class Sha256SumsTests(unittest.TestCase):
    def test_parse_handles_flat_names_and_binary_marker(self):
        a, b = "a" * 64, "b" * 64
        sums = MODULE["parse_sha256sums"](
            f"{a}  mlx_omarchy-1-cp314-cp314-linux_aarch64.whl\n"
            f"{b} *omarchy-mlx-vendor-wheels-v0.7.7-cp314-aarch64.tar\n"
            "not-a-hash  ignored\n"
        )
        self.assertEqual(
            sums,
            {
                "mlx_omarchy-1-cp314-cp314-linux_aarch64.whl": a,
                "omarchy-mlx-vendor-wheels-v0.7.7-cp314-aarch64.tar": b,
            },
        )

    def test_coverage_flags_stale_and_uncovered(self):
        cov = MODULE["sums_coverage"]
        assets = {"a.whl", "b.tar", "SHA256SUMS"}
        self.assertEqual(cov(assets, {"a.whl": "x", "gone.whl": "y"}),
                         (["gone.whl"], ["b.tar"]))
        self.assertEqual(cov(assets, {n: "x" for n in assets - {"SHA256SUMS"}}),
                         ([], []))
        self.assertEqual(cov(assets, {}), ([], []))

    def test_receipt_glob_matches_dated_vNNN_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "receipts").mkdir()
            (root / "receipts" / "2026-10-01-v077-release.md").write_text("x")
            (root / "receipts" / "unrelated.md").write_text("x")
            found = MODULE["receipt_paths"](root, "v0.7.7")
            self.assertEqual([p.name for p in found],
                             ["2026-10-01-v077-release.md"])


class AliasWheelTests(unittest.TestCase):
    STAMPED = "mlx_omarchy-0.32.4.dev1+abc1234-cp314-cp314-linux_aarch64.whl"
    X86 = "mlx_omarchy-0.32.4.dev1+abc1234-cp311-cp311-linux_x86_64.whl"

    def _wheels(self, directory, alias_bytes):
        root = Path(directory)
        stamped = root / self.STAMPED
        stamped.write_bytes(b"aarch64 wheel bytes")
        x86 = root / "other" / self.X86
        x86.parent.mkdir()
        x86.write_bytes(b"x86 wheel bytes")
        alias = None
        if alias_bytes is not None:
            alias = root / MODULE["ALIAS_WHEEL"]
            alias.write_bytes(alias_bytes)
        return alias, [stamped, x86]

    def test_identical_copy_passes_and_other_cases_fail(self):
        check = MODULE["alias_failure"]
        with tempfile.TemporaryDirectory() as directory:
            alias, wheels = self._wheels(directory, b"aarch64 wheel bytes")
            self.assertIsNone(check(False, alias, wheels))
        with tempfile.TemporaryDirectory() as directory:
            alias, wheels = self._wheels(directory, b"a different wheel")
            self.assertIn("byte-identical", check(False, alias, wheels))
        with tempfile.TemporaryDirectory() as directory:
            alias, wheels = self._wheels(directory, None)
            self.assertIn("is missing", check(False, alias, wheels))

    def test_diagnostics_release_needs_no_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            alias, wheels = self._wheels(directory, None)
            self.assertIsNone(MODULE["alias_failure"](True, alias, wheels))

    def test_copy_without_an_aarch64_wheel_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            alias, wheels = self._wheels(directory, b"aarch64 wheel bytes")
            only_x86 = [w for w in wheels if "x86_64" in w.name]
            self.assertIn("no linux_aarch64 wheel",
                          MODULE["alias_failure"](False, alias, only_x86))


class ReleaseCommitTests(unittest.TestCase):
    def test_long_abbreviation_distinguishes_shared_short_prefix(self):
        version = "0.32.2+12345678"
        dist_info = f"mlx_omarchy-{version}.dist-info"
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / (
                f"mlx_omarchy-{version}-cp311-cp311-linux_x86_64.whl"
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(f"{dist_info}/METADATA", f"Version: {version}\n")
                archive.writestr(
                    f"{dist_info}/WHEEL", "Tag: cp311-cp311-linux_x86_64\n"
                )
                archive.writestr("mlx/lib/libmlx.so", b"MLX_DISABLE_COMPILE")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            matching = "12345678" + "0" * 32
            failures, _, _ = VERIFY(
                wheel, digest, version, "test", matching, matching[:7], False
            )
            self.assertEqual(failures, [])
            different = "12345679" + "0" * 32
            failures, _, _ = VERIFY(
                wheel, digest, version, "test", different, different[:7], False
            )
            self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
