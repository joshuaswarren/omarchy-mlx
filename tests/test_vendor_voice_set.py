"""The vendored system venv carries the voice set that install.sh --voice installs.

The packaged voice assistant (offline, /usr/lib/omarchy-mlx venv) imports
mlx-audio, onnxruntime, openwakeword and the Kokoro G2P stack. These tests
pin the vendor inputs to install.sh's --voice pins, keep the --no-deps
packages out of the resolved set (mlx-audio would pull upstream mlx over the
mlx-omarchy wheel), and check that build-venv.sh installs the locked closure
without resolving dependencies again.
"""

import base64
import hashlib
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"
LOCK_IN = PACKAGING / "requirements-lock.in"
NODEPS_IN = PACKAGING / "requirements-nodeps.in"


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_lines(path):
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def requirement_name(line):
    return norm(re.split(r"[\s\[<>=!~@;]", line, 1)[0])


def install_sh_voice_pins():
    """Every requirement string install.sh's --voice block hands to pip."""
    text = (REPO / "install.sh").read_text()
    blocks = [b.split("\nfi\n", 1)[0] for b in text.split("if (( VOICE )); then")[1:]]
    block = next(b for b in blocks if "MLX_AUDIO_VERSION" in b)
    audio = re.search(r'^MLX_AUDIO_VERSION=(\S+)', text, re.M).group(1)
    pins = re.findall(r'"([A-Za-z0-9_.\-\[\]]+(?:[<>=!~]=?[^"]+|\s*@\s*https://[^"]+))"', block)
    return [p.replace("$MLX_AUDIO_VERSION", audio) for p in pins]


def make_wheel(vendor, name, version, requires=()):
    path = vendor / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    distinfo = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {r}\n" for r in requires)
    files = {f"{name.replace('-', '_')}/__init__.py": "OK = True\n",
             f"{distinfo}/METADATA": metadata,
             f"{distinfo}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    records = []
    with zipfile.ZipFile(path, "w") as zf:
        for fname, body in files.items():
            data = body.encode()
            zf.writestr(fname, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append(f"{fname},sha256={digest},{len(data)}")
        records.append(f"{distinfo}/RECORD,,")
        zf.writestr(f"{distinfo}/RECORD", "\n".join(records) + "\n")


class VoiceSetTest(unittest.TestCase):
    def test_every_install_sh_voice_pin_is_vendored(self):
        vendored = {line.replace(" ", "") for line in requirement_lines(LOCK_IN) + requirement_lines(NODEPS_IN)}
        pins = install_sh_voice_pins()
        self.assertGreater(len(pins), 15, pins)
        missing = [p for p in pins if p.replace(" ", "") not in vendored]
        self.assertEqual(missing, [], "install.sh --voice pins absent from the vendor inputs")

    def test_no_deps_packages_stay_out_of_the_resolved_set(self):
        nodeps = {requirement_name(line): line for line in requirement_lines(NODEPS_IN)}
        self.assertEqual(sorted(nodeps), ["mlx-audio", "num2words", "openwakeword"])
        resolved = {requirement_name(line) for line in requirement_lines(LOCK_IN)}
        self.assertEqual(resolved & set(nodeps), set())

    def test_vendor_download_is_binary_only_and_nodeps_has_no_deps(self):
        script = (PACKAGING / "vendor-wheels.sh").read_text()
        downloads = [l for l in script.splitlines() if "pip download" in l and not l.lstrip().startswith("#")]
        self.assertEqual(len(downloads), 2, downloads)
        for line in downloads:
            self.assertIn("--only-binary=:all:", line)
        nodeps = [l for l in downloads if "requirements-nodeps.in" in l]
        self.assertEqual(len(nodeps), 1)
        self.assertIn("--no-deps", nodeps[0])


CHECK = PACKAGING / "check-nodeps-closure.py"


class NodepsClosureCheckTest(unittest.TestCase):
    """check-nodeps-closure.py: every Requires-Dist of a --no-deps wheel is in the lock."""

    def run_check(self, requires, lock_lines, nodeps_extra=""):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wheels = tmp / "wheels"
            wheels.mkdir()
            make_wheel(wheels, "voicepkg", "1.0", requires=requires)
            nodeps = tmp / "nodeps.in"
            nodeps.write_text("voicepkg==1.0\n" + nodeps_extra)
            lock = tmp / "lock.txt"
            lock.write_text("\n".join(lock_lines) + "\n")
            return subprocess.run([sys.executable, str(CHECK), "--nodeps", str(nodeps),
                                   "--wheels", str(wheels), "--lock", str(lock)],
                                  capture_output=True, text=True)

    def test_covered_dependencies_pass(self):
        result = self.run_check(["scipy>=1.10", "tqdm"], ["scipy==1.18.1 --hash=sha256:" + "0" * 64,
                                                          "tqdm==4.67.1 --hash=sha256:" + "1" * 64])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_dependency_missing_from_the_lock_fails_by_name(self):
        result = self.run_check(["scipy>=1.10", "einops>=0.8"], ["scipy==1.18.1 --hash=sha256:" + "0" * 64])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("einops", result.stdout + result.stderr)

    def test_a_pinned_version_outside_the_specifier_fails(self):
        result = self.run_check(["scipy>=1.20"], ["scipy==1.18.1 --hash=sha256:" + "0" * 64])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scipy", result.stdout + result.stderr)

    def test_extras_and_other_platforms_are_ignored(self):
        result = self.run_check(['torch>=2; extra == "full"', 'pywin32; sys_platform == "win32"'], ["tqdm==1"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_declared_not_vendored_dependency_passes(self):
        result = self.run_check(["mlx>=0.31.1"], ["tqdm==1"],
                                nodeps_extra="# not-vendored: mlx  mlx-omarchy provides it\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_specifier_lines_of_the_input_file_count_as_covered(self):
        result = self.run_check(["scipy>=1.10"], ["scipy>=1.10.0"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_real_nodeps_file_declares_its_exemptions(self):
        text = NODEPS_IN.read_text()
        for name in ("mlx", "tflite-runtime", "docopt"):
            self.assertRegex(text, rf"(?m)^# not-vendored: {re.escape(name)}\s+\S")

    def test_vendor_wheels_runs_the_check_on_the_real_lock(self):
        script = (PACKAGING / "vendor-wheels.sh").read_text()
        self.assertIn("check-nodeps-closure.py", script)


class VoiceVenvProofTest(unittest.TestCase):
    def test_release_vendoring_runs_the_offline_voice_import_proof(self):
        script = (PACKAGING / "vendor-wheels.sh").read_text()
        self.assertIn("verify-voice-venv.sh", script)
        proof = (PACKAGING / "verify-voice-venv.sh").read_text()
        self.assertIn("unshare", proof)
        for module in ("mlx_omarchy_assistant.voice_loop", "mlx_omarchy_assistant.synthesis",
                       "mlx_audio", "onnxruntime", "openwakeword", "misaki.en"):
            self.assertIn(module, proof)


class BuildVenvClosureTest(unittest.TestCase):
    def test_locked_closure_installs_without_resolving_dependencies(self):
        # mlx-audio declares mlx; the vendor set satisfies it with mlx-omarchy,
        # which pip does not count. The lock is the closure, so no resolve step.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            vendor = tmp / "vendor"
            vendor.mkdir()
            make_wheel(vendor, "needsmlx", "1.0", requires=["upstream-mlx-not-vendored>=0.31"])
            lock = tmp / "lock.txt"
            lock.write_text(subprocess.run(["bash", str(PACKAGING / "gen-vendor-lock.sh"), str(vendor)],
                                           check=True, capture_output=True, text=True).stdout)
            venv = tmp / "venv"
            result = subprocess.run(["bash", str(PACKAGING / "build-venv.sh"), "--vendor", str(vendor),
                                     "--lock", str(lock), "--venv", str(venv), "--python", sys.executable],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            check = subprocess.run([str(venv / "bin" / "python"), "-c", "import needsmlx; print(needsmlx.OK)"],
                                   capture_output=True, text=True)
            self.assertEqual(check.stdout.strip(), "True", check.stderr)


if __name__ == "__main__":
    unittest.main()
