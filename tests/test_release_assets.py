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
