"""Builder tests for the m3-tester-kit (scripts/build-m3-tester-kit.sh).

Every run is --self-test or a stub-wheel build in a temp dir: no hardware,
no network, no real wheel. The self-test is the kit's own clean-extract
verification (SHA256SUMS, file inventory, bash -n, py_compile, privacy
audit, DEFER_TOKENS pin, tamper check) and must stay green offline.
"""

import os
import re
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


PREFLIGHT = ROOT / "m3-tester-kit" / "ane-preflight.sh"


def fake_root(d: pathlib.Path, soc: bytes, optin_key: str | None) -> pathlib.Path:
    root = d / "aneroot"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "proc/device-tree/compatible").write_bytes(soc + b"\0")
    if optin_key is not None:
        (root / "etc/omarchy-mac-boot").mkdir(parents=True)
        (root / "etc/omarchy-mac-boot/dtb-overlays.opt-in").write_text(
            optin_key + "\n")
    return root


class AnePreflightTests(unittest.TestCase):
    """ane-preflight.sh runs (not just bash -n): the ANE_ROOT seam re-roots
    /proc, /etc and /sys onto a stub device-tree root so the report and the
    refusal summary are exercised."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)

    def run_preflight(self, root: pathlib.Path) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(PREFLIGHT)], capture_output=True,
                              text=True,
                              env={**os.environ, "ANE_ROOT": str(root)})

    def test_ready_stub_root_passes_with_the_ready_summary(self):
        root = fake_root(self.base, b"apple,t6031", "ane-h15-experimental")
        r = self.run_preflight(root)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("chip-map: PASS", r.stdout)
        self.assertIn("opt-in-key: PASS", r.stdout)
        self.assertIn("no refusals — this base M3 is bring-up ready",
                      r.stdout)
        self.assertNotIn("unbound variable", r.stderr)

    def test_missing_opt_in_refuses_and_names_the_fix(self):
        root = fake_root(self.base, b"apple,t6031", None)
        r = self.run_preflight(root)
        self.assertEqual(r.returncode, 1, r.stderr + r.stdout)
        self.assertIn("opt-in-key: REFUSED", r.stdout)
        self.assertIn("ane-h15-experimental", r.stdout)
        self.assertIn("1 refusal(s)", r.stdout)
        self.assertNotIn("unbound variable", r.stderr)

    def test_unknown_chip_skips_the_opt_in_check(self):
        root = fake_root(self.base, b"apple,t6000", None)
        r = self.run_preflight(root)
        self.assertEqual(r.returncode, 1, r.stderr + r.stdout)
        self.assertIn("chip-map: REFUSED", r.stdout)
        self.assertIn("no opt-in key applies", r.stdout)
        self.assertNotIn("line ''", r.stdout)
        self.assertNotIn("unbound variable", r.stderr)


RUN_CARD = ROOT / "m3-tester-kit" / "run-m3-card.sh"
GUARD = ROOT / "scripts" / "defer_tokens_guard.py"


class RunM3CardTests(unittest.TestCase):
    """run-m3-card.sh calls into scripts/defer_tokens_guard.py by name; a
    typo'd entry point kills the runner under set -e before any card cell
    executes."""

    def test_guard_calls_resolve_to_module_attributes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("defer_tokens_guard",
                                                      GUARD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        calls = set(re.findall(r"defer_tokens_guard\.([a-z_]+)\(",
                               RUN_CARD.read_text()))
        self.assertTrue(calls, "no defer_tokens_guard calls found")
        for name in calls:
            self.assertTrue(callable(getattr(mod, name, None)),
                            f"defer_tokens_guard.{name} is not callable")

    def test_guard_step_runs_the_real_call(self):
        # The [3/4] step calls the guard inside a venv python against the
        # staged scripts/; replay the same import + call with the defer pin.
        code = ("import sys; sys.path.insert(0, sys.argv[1] + '/scripts'); "
                "import defer_tokens_guard; "
                "defer_tokens_guard.require_defer_tokens('m3-tester-kit')")
        r = subprocess.run([os.sys.executable, "-c", code, str(ROOT)],
                           capture_output=True, text=True,
                           env={**os.environ, "MLX_OMARCHY_DEFER_TOKENS": "0"})
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
