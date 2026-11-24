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
import io
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


def zip_bytes(members, compression=zipfile.ZIP_STORED):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression) as zf:
        for member, data in members.items():
            zf.writestr(member, data)
    return bytearray(out.getvalue())


def encrypted_metadata_wheel():
    """A wheel whose METADATA is flagged encrypted, so reading it needs a password."""
    raw = zip_bytes({"voicepkg-1.0.dist-info/METADATA": b"Requires-Dist: tqdm\n"})
    raw[6] |= 1  # local file header flag
    raw[raw.index(b"PK\x01\x02") + 8] |= 1  # central directory flag
    return bytes(raw)


def corrupt_deflate_wheel():
    member = "voicepkg-1.0.dist-info/METADATA"
    raw = zip_bytes({member: "".join(f"Requires-Dist: pkg{i}\n" for i in range(400))}, zipfile.ZIP_DEFLATED)
    data = 30 + len(member)  # compressed data follows the local file header
    for i in range(data + 10, data + 30):
        raw[i] ^= 0xFF
    return bytes(raw)


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

    def run_raw(self, nodeps, lock, wheels):
        """nodeps, lock: file bytes (None leaves the file out); wheels: filename -> bytes or {member: bytes}."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "wheels").mkdir()
            for filename, body in wheels.items():
                if isinstance(body, bytes):
                    (tmp / "wheels" / filename).write_bytes(body)
                    continue
                with zipfile.ZipFile(tmp / "wheels" / filename, "w") as zf:
                    for member, data in body.items():
                        zf.writestr(member, data)
            for name, data in (("nodeps.in", nodeps), ("lock.txt", lock)):
                if data is not None:
                    (tmp / name).write_bytes(data)
            return subprocess.run([sys.executable, str(CHECK), "--nodeps", str(tmp / "nodeps.in"),
                                   "--wheels", str(tmp / "wheels"), "--lock", str(tmp / "lock.txt")],
                                  capture_output=True, text=True)

    def assert_clean_error(self, result, text):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertTrue(result.stderr.startswith("error: "), result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn(text, result.stderr)

    def test_an_entry_that_is_not_one_exact_pin_fails_cleanly(self):
        # An unpinned or URL entry used to raise StopIteration; a range or
        # compound one checked whichever wheel its first specifier named.
        lock = f"tqdm==1 --hash=sha256:{'0' * 64}\n".encode()
        wheels = {f"voicepkg-{v}-py3-none-any.whl": {f"voicepkg-{v}.dist-info/METADATA": meta}
                  for v, meta in (("1.0", b"Name: voicepkg\nRequires-Dist: tqdm\n"),
                                  ("2.0", b"Name: voicepkg\nRequires-Dist: einops\n"))}
        for entry in ("voicepkg", "voicepkg @ https://example.com/voicepkg-1.0-py3-none-any.whl",
                      "voicepkg>=1.0", "voicepkg!=2.0", "voicepkg~=1.0", "voicepkg==1.0,!=1.0.1",
                      "voicepkg==1.*", "voicepkg===1.0", "voicepkg[extra]==1.0",
                      'voicepkg==1.0; sys_platform == "win32"'):
            with self.subTest(entry=entry):
                result = self.run_raw(f"{entry}\n".encode(), lock, wheels)
                self.assert_clean_error(result, f"nodeps.in:1: {entry!r} must pin one version as name==version")

    def test_the_pin_finds_its_wheel_by_normalized_name_and_version(self):
        lock = f"tqdm==1 --hash=sha256:{'0' * 64}\n".encode()
        meta = b"Name: voice.pkg\nRequires-Dist: tqdm\n"
        for entry, filename in (("voice-pkg==1.0.0", "voice_pkg-1.0-py3-none-any.whl"),
                                ("Voice_Pkg==1.0", "voice.pkg-1.0-py3-none-any.whl")):
            with self.subTest(entry=entry, wheel=filename):
                result = self.run_raw(f"{entry}\n".encode(), lock,
                                      {filename: {f"{filename.split('-')[0]}-1.0.dist-info/METADATA": meta}})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_only_the_exact_version_with_wheel_tags_is_checked(self):
        # ==1.0 also contains 1.0+cpu, so a contains() match checked
        # whichever wheel the directory listed first.
        lock = f"tqdm==1 --hash=sha256:{'0' * 64}\n".encode()
        local = {"voicepkg-1.0+cpu-py3-none-any.whl": {"voicepkg-1.0+cpu.dist-info/METADATA":
                                                       b"Requires-Dist: einops\n"}}
        untagged = {"voicepkg-1.0-anything.whl": {"voicepkg-1.0.dist-info/METADATA": b"Requires-Dist: einops\n"}}
        for label, wheels in (("local version only", local), ("no wheel tags", untagged)):
            with self.subTest(label):
                self.assert_clean_error(self.run_raw(b"voicepkg==1.0\n", lock, wheels),
                                        "error: no wheel for voicepkg==1.0")
        # foo-1.0---.whl sorts first; older packaging parses its empty tags.
        empty_tags = dict(untagged, **{"voicepkg-1.0---.whl": {"voicepkg-1.0.dist-info/METADATA": b"Name: voicepkg\n"},
                                       "voicepkg-1.0-py3-none-any.whl": untagged["voicepkg-1.0-anything.whl"]})
        self.assert_clean_error(self.run_raw(b"voicepkg==1.0\n", lock, empty_tags), "voicepkg needs einops")
        both = dict(local, **{"voicepkg-1.0-py3-none-any.whl": {"voicepkg-1.0.dist-info/METADATA":
                                                                 b"Requires-Dist: tqdm\n"}})
        result = self.run_raw(b"voicepkg==1.0\n", lock, both)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_vendored_dist_info_inside_the_wheel_is_not_its_metadata(self):
        result = self.run_raw(b"voicepkg==1.0\n", f"tqdm==1 --hash=sha256:{'0' * 64}\n".encode(),
                              {"voicepkg-1.0-py3-none-any.whl": {
                                  "voicepkg/_vendor/ghost-9.dist-info/METADATA":
                                      b"Name: ghost\nRequires-Dist: einops\n",
                                  "voicepkg-1.0.dist-info/METADATA": b"Name: voicepkg\nRequires-Dist: tqdm\n"}})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_malformed_inputs_fail_with_an_error_line(self):
        pin, lock = b"voicepkg==1.0\n", f"tqdm==1 --hash=sha256:{'0' * 64}\n".encode()
        good = {"voicepkg-1.0-py3-none-any.whl": {"voicepkg-1.0.dist-info/METADATA": b"Requires-Dist: tqdm\n"}}
        cases = {
            "invalid nodeps line": (b"voicepkg==\n", lock, good, "nodeps.in:1: invalid requirement 'voicepkg=='"),
            "invalid compound specifier": (b"voicepkg===1.0,2.0\n", lock, good, "nodeps.in:1: invalid requirement"),
            "non-UTF-8 nodeps file": (b"voicepkg==1.0  # caf\xe9\n", lock, good, "cannot read"),
            "missing lock file": (pin, None, good, "cannot read"),
            "non-UTF-8 lock file": (pin, lock.rstrip() + b" # caf\xe9\n", good, "cannot read"),
            "invalid lock version": (pin, f"tqdm==x!y --hash=sha256:{'0' * 64}\n".encode(), good,
                                     "lock.txt:1: Invalid version: 'x!y'"),
            "invalid lock line": (pin, b"???\n", good, "lock.txt:1: invalid requirement '???'"),
            "corrupt wheel": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": b"not a zip"}, "cannot read"),
            "corrupt compressed METADATA": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": corrupt_deflate_wheel()},
                                            "cannot read"),
            "encrypted METADATA": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": encrypted_metadata_wheel()},
                                   "cannot read"),
            "wheel without METADATA": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {"voicepkg/__init__.py": b""}},
                                       "has no .dist-info/METADATA"),
            "non-UTF-8 METADATA": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b"Summary: caf\xe9\n"}}, "cannot read"),
            "invalid Requires-Dist": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b"Requires-Dist: tqdm>>1\n"}},
                "Requires-Dist: invalid requirement 'tqdm>>1'"),
            "METADATA header line without a colon": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b"Name: voicepkg\nRequires-Dist missing\nRequires-Dist: einops\n"}},
                "METADATA has a malformed header line"),
            "METADATA starting with a continuation line": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b" Requires-Dist: einops\nName: voicepkg\n"}},
                "METADATA has a malformed header line"),
            "Requires-Dist with an invalid compound specifier": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b"Requires-Dist: bar===1.0,2.0\n"}},
                "Requires-Dist: invalid requirement"),
            "Requires-Dist with an unterminated quote": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b"Requires-Dist: bar; sys_platform == 'win32\\'\n"}},
                "Requires-Dist: invalid requirement"),
            "Requires-Dist marker that cannot be evaluated": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b'Requires-Dist: bar; os_name ~= "1.0"\n'}},
                "Requires-Dist: cannot evaluate"),
            "Requires-Dist marker on an unknown variable": (pin, lock, {"voicepkg-1.0-py3-none-any.whl": {
                "voicepkg-1.0.dist-info/METADATA": b'Requires-Dist: bar; "dev" in dependency_groups\n'}},
                "Requires-Dist: cannot evaluate"),
        }
        for label, (nodeps, lock_bytes, wheels, text) in cases.items():
            with self.subTest(label):
                self.assert_clean_error(self.run_raw(nodeps, lock_bytes, wheels), text)

    def test_the_real_nodeps_file_declares_its_exemptions(self):
        text = NODEPS_IN.read_text()
        for name in ("mlx", "tflite-runtime"):
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


BUILDWHEEL_IN = PACKAGING / "requirements-buildwheel.in"
SPLIT = PACKAGING / "split-vendor-lock.py"


class BuildWheelInputTest(unittest.TestCase):
    """docopt ships no wheel upstream; vendor-wheels.sh builds one from the sdist."""

    def test_docopt_is_in_the_buildwheel_input(self):
        # Without docopt the vendored set fails pip --require-hashes
        # resolution: num2words declares docopt>=0.6.2.
        pins = [p.replace(" ", "") for p in requirement_lines(BUILDWHEEL_IN)] if BUILDWHEEL_IN.exists() else []
        self.assertIn("docopt==0.6.2", pins)
        self.assertNotIn("not-vendored: docopt", NODEPS_IN.read_text())

    def test_vendor_script_builds_the_buildwheel_set(self):
        script = (PACKAGING / "vendor-wheels.sh").read_text()
        lines = [l for l in script.splitlines() if "pip wheel" in l and not l.lstrip().startswith("#")]
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("--no-deps", lines[0])
        self.assertIn("requirements-buildwheel.in", lines[0])


class SplitVendorLockTest(unittest.TestCase):
    """split-vendor-lock.py: only wheels with an unvendored dep stay --no-deps."""

    HASH = "0" * 64

    def run_split(self, wheels_spec, nodeps_text, lock_lines):
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            wheels = tmp / "wheels"
            wheels.mkdir()
            for name, version, reqs in wheels_spec:
                make_wheel(wheels, name, version, requires=reqs)
            nodeps = tmp / "nodeps.in"
            nodeps.write_text(nodeps_text)
            lock = tmp / "lock.txt"
            lock.write_text("\n".join(lock_lines) + "\n")
            resolvable = tmp / "resolvable.txt"
            nodeps_only = tmp / "nodeps.txt"
            result = subprocess.run([sys.executable, str(SPLIT), "--nodeps", str(nodeps),
                                     "--wheels", str(wheels), "--lock", str(lock),
                                     "--resolvable", str(resolvable), "--nodeps-only", str(nodeps_only)],
                                    capture_output=True, text=True)
            return result, (resolvable.read_text() if resolvable.exists() else "",
                            nodeps_only.read_text() if nodeps_only.exists() else "")

    def test_only_packages_with_exempt_dependencies_stay_nodeps(self):
        # voicepkg declares a vendored dep and must resolve; legacy declares a
        # not-vendored dep and cannot.
        result, (resolvable, nodeps) = self.run_split(
            [("voicepkg", "1.0", ["docopt>=0.6"]),
             ("legacy", "1.0", ["ghostpkg"]),
             ("docopt", "0.6.2", [])],
            "voicepkg==1.0\nlegacy==1.0\n# not-vendored: ghostpkg  no wheel for the target\n",
            [f"voicepkg==1.0 --hash=sha256:{self.HASH}",
             f"legacy==1.0 --hash=sha256:{self.HASH}",
             f"docopt==0.6.2 --hash=sha256:{self.HASH}"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(resolvable.splitlines()),
                         sorted([f"voicepkg==1.0 --hash=sha256:{self.HASH}",
                                 f"docopt==0.6.2 --hash=sha256:{self.HASH}"]))
        self.assertEqual(nodeps.strip(), f"legacy==1.0 --hash=sha256:{self.HASH}")

    def test_pip_resolves_the_resolvable_side_and_fails_on_a_missing_dep(self):
        # The mechanism the release gate exercises: pip --require-hashes
        # resolution over the resolvable lines aborts by name when a declared
        # dependency is absent from the vendored set.
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            wheels = tmp / "wheels"
            wheels.mkdir()
            make_wheel(wheels, "voicepkg", "1.0", requires=["docopt>=0.6"])
            make_wheel(wheels, "docopt", "0.6.2")
            target = tmp / "target"
            lock = tmp / "lock.txt"
            lock.write_text("\n".join(
                f"{w.name.split('-', 1)[0]}=={w.name.split('-', 2)[1]} "
                f"--hash=sha256:{hashlib.sha256(w.read_bytes()).hexdigest()}"
                for w in sorted(wheels.iterdir())) + "\n")
            base = [sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", str(wheels), "--require-hashes", "--target", str(target)]
            ok = subprocess.run(base + ["-r", str(lock)], capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            lock.write_text("\n".join(l for l in lock.read_text().splitlines()
                                      if "docopt" not in l) + "\n")
            bad = subprocess.run(base + ["-r", str(lock)], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("docopt", bad.stderr)
