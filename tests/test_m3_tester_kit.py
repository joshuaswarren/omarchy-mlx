"""Builder tests for the m3-tester-kit (scripts/build-m3-tester-kit.sh).

Every run is --self-test or a stub-wheel build in a temp dir: no hardware,
no network, no real wheel. The self-test is the kit's own clean-extract
verification (SHA256SUMS, file inventory, bash -n, py_compile, privacy
audit, DEFER_TOKENS pin, tamper check) and must stay green offline.
"""

import os
import pathlib
import subprocess
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build-m3-tester-kit.sh"


def stub_wheel(d: pathlib.Path) -> pathlib.Path:
    whl = d / "mlx_omarchy-0.0.0.dev0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("mlx_omarchy-0.0.0.dev0.dist-info/METADATA",
                   "Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: 0.0.0.dev0\n")
        z.writestr("mlx_omarchy-0.0.0.dev0.dist-info/WHEEL",
                   "Wheel-Version: 1.0\nTag: py3-none-any\n")
    return whl


def build(*args: str, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("MKIT_ANE_ROOT", None)
    return subprocess.run(["bash", str(BUILDER), *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


class M3TesterKitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)

    def test_self_test_passes(self):
        r = build("--self-test", "--out", str(self.base), cwd=self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("SELF-TEST PASS", r.stdout)
        tars = list(self.base.glob("m3-tester-kit-r1-*.tar"))
        self.assertEqual(len(tars), 1, r.stdout)
        self.assertTrue(tars[0].with_suffix(".tar.sha256").exists()
                        or (self.base / (tars[0].name + ".sha256")).exists())

    def test_stub_build_and_clean_extract_verify(self):
        whl = stub_wheel(self.base)
        r = build("--wheel", str(whl), "--out", str(self.base),
                  "--tag", "testtag", cwd=self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        kit = self.base / "m3-tester-kit-r1-testtag"
        self.assertTrue((kit / "SHA256SUMS").is_file())
        # Clean extract of the tar verifies against the manifest.
        x = self.base / "extract"
        x.mkdir()
        subprocess.run(["tar", "-xf", str(self.base / "m3-tester-kit-r1-testtag.tar")],
                       cwd=x, check=True)
        xkit = x / "m3-tester-kit-r1-testtag"
        v = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=xkit,
                           capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertTrue((xkit / "wheel" / whl.name).is_file())
        self.assertTrue((xkit / "scripts" / "apply-mlx-lm-patches.sh").is_file())

    def test_privacy_audit_refuses_private_ip(self):
        # Plant an RFC-1918 address via the M3TK_PLANT hook; the builder's
        # staged-file audit must refuse before sealing (fail-closed).
        leak = self.base / "leak.txt"
        # built in parts: the literal must never sit in this file (the
        # repo's own privacy_check.py scans commit blobs for it)
        leak.write_text("probe " + ".".join(map(str, (192, 168, 4, 7))) + "\n")
        whl = stub_wheel(self.base)
        env = dict(os.environ, M3TK_PLANT=str(leak))
        r = subprocess.run(
            ["bash", str(BUILDER), "--wheel", str(whl), "--out",
             str(self.base / "refused"), "--tag", "audit"],
            cwd=self.base, env=env, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("privacy audit REFUSED", r.stderr)

    def test_clean_kit_has_no_private_ips(self):
        whl = stub_wheel(self.base)
        r = build("--wheel", str(whl), "--out", str(self.base),
                  "--tag", "audit", cwd=self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        kit = self.base / "m3-tester-kit-r1-audit"
        self.assertFalse(
            subprocess.run(
                ["grep", "-rEl", r"192\.168\.[0-9]{1,3}\.[0-9]{1,3}", str(kit)],
                capture_output=True).stdout.strip(),
            "private IP leaked into a clean kit build")

    def test_no_absolute_home_paths_in_kit(self):
        whl = stub_wheel(self.base)
        r = build("--wheel", str(whl), "--out", str(self.base),
                  "--tag", "paths", cwd=self.base)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        kit = self.base / "m3-tester-kit-r1-paths"
        bad = subprocess.run(
            ["grep", "-rEl", r"/home/[a-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/",
             str(kit)], capture_output=True, text=True)
        self.assertEqual(bad.stdout.strip(), "", bad.stdout)


if __name__ == "__main__":
    unittest.main()
