"""Installer refusal, uninstall, and ANE-smoke gate in a disposable HOME."""

import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
LAUNCHERS = ("mlx-omarchy", "mlx-omarchy-demo", "mlx-omarchy-chat", "mlx-omarchy-info")

# Deterministic offline contract tests: never resolve the latest release
# over the network (api.github.com rate limits make that flake).
PINNED_VERSION_ENV = {"MLX_OMARCHY_VERSION": "v0.7.1"}


def installer_text():
    return INSTALLER.read_text()


def extract_ane_smoke():
    match = re.search(r"<<'ANE_SMOKE'[^\n]*\n(.*)\nANE_SMOKE", installer_text(), re.S)
    if match is None:
        raise AssertionError("ANE_SMOKE heredoc missing from install.sh")
    return match.group(1)


class InstallerContractTests(unittest.TestCase):
    @unittest.skipIf(Path("/dev/accel/accel0").exists(), "requires a host without ANE")
    def test_ane_install_refuses_before_writing_without_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(INSTALLER), "--ane"],
                env={**os.environ, **PINNED_VERSION_ENV, "HOME": tmp},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ANE installation requires /dev/accel/accel0", result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_uninstall_removes_owned_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            prefix = home / "custom-prefix"
            binary = home / ".local/bin"
            apps = home / ".local/share/applications"
            for directory in (prefix, binary, apps):
                directory.mkdir(parents=True)
            artifacts = [prefix / "venv", apps / "mlx-omarchy-chat.desktop",
                         apps / "mlx-omarchy-demo.desktop"]
            artifacts.extend(binary / name for name in LAUNCHERS)
            for artifact in artifacts:
                artifact.touch()
            sentinel = binary / "unrelated"
            sentinel.write_text("preserve")
            result = subprocess.run(
                ["bash", str(INSTALLER), "--uninstall"],
                env={**os.environ, "HOME": str(home), "MLX_OMARCHY_HOME": str(prefix)},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(prefix.exists())
            for artifact in artifacts:
                self.assertFalse(artifact.exists(), artifact)
            self.assertEqual(sentinel.read_text(), "preserve")

    @unittest.skipIf(platform.machine() == "aarch64", "requires an off-target host")
    def test_refused_install_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            result = subprocess.run(
                ["bash", str(INSTALLER)],
                env={**os.environ, "HOME": tmp, "MLX_OMARCHY_HOME": str(home / "prefix")},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(home.iterdir()), [])


class VoiceDependencyContractTests(unittest.TestCase):
    """install.sh --voice must carry every pin the default engine declares,
    so the shipped installer can never drift from KOKORO_PACK's runtime."""

    def test_voice_install_covers_default_engine_pins(self):
        sys.path.insert(0, str(ROOT / "serve"))
        from mlx_omarchy_assistant import synthesis

        text = installer_text()
        for name, constraint in \
                synthesis.KOKORO_PACK["runtime"]["constraints"].items():
            if name == "en-core-web-sm":
                self.assertIn(f"en_core_web_sm-{constraint.removeprefix('==')}",
                              text,
                              "install.sh must pin the en_core_web_sm wheel")
            elif name == "mlx_audio":
                version = re.search(r'^MLX_AUDIO_VERSION=(\S+)', text,
                                    re.M)
                self.assertIsNotNone(version,
                                     "install.sh must define MLX_AUDIO_VERSION")
                self.assertEqual(
                    version.group(1), constraint.removeprefix("=="),
                    "install.sh MLX_AUDIO_VERSION must match the default "
                    "engine's mlx-audio pin")
            else:
                self.assertIn(f"{name}{constraint}", text,
                              f"install.sh --voice must pin {name}{constraint}")


ROOT = INSTALLER.parents[0]


class AneSmokeGateTests(unittest.TestCase):
    def test_gpu_smoke_always_present(self):
        text = installer_text()
        self.assertIn("import mlx.core as mx", text)
        self.assertIn("say \"Smoke test\"", text)

    def test_ane_smoke_is_character_device_gated(self):
        text = installer_text()
        self.assertIn('[[ -c "${MLX_OMARCHY_ACCEL_DEV:-/dev/accel/accel0}" ]]', text)
        self.assertIn("ANE: unavailable (no /dev/accel/accel0); GPU-only install", text)
        self.assertIn('die "ANE smoke failed"', text)
        self.assertNotIn("omarchy-pkg-add kmod-ane", text)
        self.assertNotIn("pacman -S kmod-ane", text)
        self.assertNotIn("omarchy-pkg-add ane", text)

    def test_uninstall_still_covers_info_launcher(self):
        text = installer_text()
        self.assertIn('"$BIN/mlx-omarchy-info"', text)
        self.assertIn('"$BIN/mlx-omarchy-demo"', text)
        self.assertIn('"$BIN/mlx-omarchy-chat"', text)
        self.assertIn('"$APPS/mlx-omarchy-chat.desktop"', text)
        # Pre-existing installs named the desktop entry after the demo.
        self.assertIn('"$APPS/mlx-omarchy-demo.desktop"', text)

    def test_extracted_ane_smoke_refuses_char_device_without_fdt(self):
        script = extract_ane_smoke()
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, **PINNED_VERSION_ENV, "MLX_OMARCHY_ACCEL_DEV": "/dev/null",
                   "MLX_OMARCHY_SYSROOT": tmp}
            result = subprocess.run(
                ["python3", "-c", script],
                env=env, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing FDT node or ane module", result.stderr)

    def test_extracted_ane_smoke_passes_composed_fake_tree(self):
        self.assertTrue(stat.S_ISCHR(os.stat("/dev/null", follow_symlinks=False).st_mode))
        script = extract_ane_smoke()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "sys/firmware/devicetree/base/ane@26a000000"
            node.mkdir(parents=True)
            (node / "compatible").write_bytes(b"apple,t8103-ane\x00")
            module = root / "sys/module/ane"
            module.mkdir(parents=True)
            (module / "version").write_text("f2a3e5e+lifecycle6\n")
            result = subprocess.run(
                ["python3", "-c", script],
                env={**os.environ, "MLX_OMARCHY_ACCEL_DEV": "/dev/null",
                     "MLX_OMARCHY_SYSROOT": str(root)},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ANE smoke OK", result.stdout)


    def test_extracted_ane_smoke_accepts_t6021_module_name(self):
        """The M2 driver registers as ane_t6021; the smoke must accept it.

        omarchy-ane installs a per-chip module on the M2 Max; an install on
        a t6021 host with only that module loaded must pass (v0.7.8 draft
        gate: /sys/module/ane alone was checked and the install refused).
        """
        script = extract_ane_smoke()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "sys/firmware/devicetree/base/ane@284000000"
            node.mkdir(parents=True)
            (node / "compatible").write_bytes(b"apple,t6021-ane\x00")
            module = root / "sys/module/ane_t6021"
            module.mkdir(parents=True)
            (module / "version").write_text("0.4.0\n")
            result = subprocess.run(
                ["python3", "-c", script],
                env={**os.environ, "MLX_OMARCHY_ACCEL_DEV": "/dev/null",
                     "MLX_OMARCHY_SYSROOT": str(root)},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ANE smoke OK", result.stdout)
            self.assertIn("(0.4.0)", result.stdout)

    def test_extracted_ane_smoke_refuses_accel0_with_neither_module_name(self):
        script = extract_ane_smoke()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            node = root / "sys/firmware/devicetree/base/ane@284000000"
            node.mkdir(parents=True)
            (node / "compatible").write_bytes(b"apple,t6021-ane\x00")
            result = subprocess.run(
                ["python3", "-c", script],
                env={**os.environ, "MLX_OMARCHY_ACCEL_DEV": "/dev/null",
                     "MLX_OMARCHY_SYSROOT": str(root)},
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing FDT node or ane module", result.stderr)


class ServeCliContractTests(unittest.TestCase):
    """Serve CLI integration: package fetch, launchers, omarchy conventions."""

    def serve_section(self):
        text = installer_text()
        start = text.index('# 5b. Serve CLI')
        end = text.index('# 5d. MLX Chat assistant')
        return text[start:end]

    def test_installer_per_file_lists_cover_the_source_trees(self):
        """Every file in a serve package's source tree must appear in the
        installer's per-file fetch list for that package.

        Regression: v0.7.15's release lane staged serve files from a
        hardcoded list that lacked perf_placement.py, so every user-style
        install died at serve startup with ModuleNotFoundError even though
        the module shipped in the repo and the --system lane (whole-dir
        copy) carried it. This test compares each `for X_file in ...`
        list in install.sh against the actual directory contents, so a
        new module without a list update fails here, not on a user box.
        """
        installer = installer_text()
        # var name -> (source dir relative to repo root, strip prefix for
        # static lists). One entry per per-file curl list in install.sh.
        lists = {
            "serve_file": ("serve/mlx_omarchy_serve", ""),
            "laya_file": ("serve/mlx_omarchy_laya", ""),
            "bonsai2_file": ("serve/mlx_omarchy_bonsai2", ""),
            "assistant_file": ("serve/mlx_omarchy_assistant", ""),
            "assistant_static": ("serve/mlx_omarchy_assistant/static", ""),
        }
        for var, (src_rel, _) in lists.items():
            match = re.search(
                rf"for {var} in ([^;]+); do", installer)
            self.assertIsNotNone(match, f"no per-file list for {var}")
            listed = set(match.group(1).split())
            if var == "assistant_static":
                static_root = INSTALLER.parent / src_rel
                expected = {str(p.relative_to(static_root))
                            for p in static_root.rglob("*") if p.is_file()}
            else:
                expected = {p.name for p in (INSTALLER.parent / src_rel).iterdir()
                            if p.is_file()}
            # Repo documentation (CONTRACT.md, README*) deliberately stays
            # in-repo; the installer never ships it. ids_probe.py is a
            # bench-only, env-gated hook (its docstring: default OFF) that
            # the bonsai2 shim imports lazily only when a bench env var is
            # set; it is not part of a user install.
            expected -= {n for n in expected
                         if n == "CONTRACT.md" or n.startswith("README")
                         or n == "ids_probe.py"}
            self.assertEqual(
                expected, listed,
                f"{var} list out of sync with {src_rel}: "
                f"missing_from_list={sorted(expected - listed)} "
                f"stale_in_list={sorted(listed - expected)}")


    def test_serve_package_fetched_from_pinned_release_tag(self):
        section = self.serve_section()
        self.assertIn('"https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_serve/$serve_file"', section)
        for member in ("__init__.py", "catalog.py", "budget.py", "perf_placement.py", "export_fit_table.py", "__main__.py", "_mlxlm_server.py", "catalog.json"):
            self.assertIn(member, section.split("for serve_file in", 1)[1].split(";", 1)[0],
                          f"missing catalog/package file {member}")
        self.assertIn('"https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_laya/$laya_file"', section)
        for member in ("__init__.py", "model.py", "sequence.py", "api.py",
                       "server.py", "convert.py", "qualify.py"):
            self.assertIn(member, section.split("for laya_file in", 1)[1].split(";", 1)[0],
                          f"missing laya package file {member}")
        self.assertIn('"https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_bonsai2/$bonsai2_file"', section)
        for member in ("__init__.py", "packed.py", "loader.py", "server.py"):
            self.assertIn(member, section.split("for bonsai2_file in", 1)[1].split(";", 1)[0],
                          f"missing bonsai2 package file {member}")
        self.assertNotIn("CONTRACT.md", section)  # repo documentation stays in-repo

    def test_apply_script_helper_scripts_are_fetched(self):
        """The installer downloads every scripts/*.py the apply script runs.

        The apply script invokes its helpers at $ROOT/scripts/<name>.py with
        $ROOT = the install prefix; a helper it names must never be missing
        from a fresh install (v0.7.7 draft gate: patch-mlx-lm-rope-norm.py
        was missing and the install died mid-patch).
        """
        text = installer_text()
        section = text.split("Applying mlx-lm serve patches", 1)[1]
        self.assertIn("grep -oE 'scripts/[a-z0-9_-]+\\.py'", section)
        self.assertIn('"https://raw.githubusercontent.com/$REPO/$VERSION/$s"', section)
        self.assertIn('-o "$PREFIX/$s"', section)
        # the loop sits before the apply-script run it feeds
        self.assertLess(section.index("for s in $(grep -oE 'scripts/"),
                        section.index('bash "$PREFIX/apply-mlx-lm-patches.sh"'))
        # every helper the apply script names is matched by the fetch pattern
        apply = INSTALLER.parent / "scripts" / "apply-mlx-lm-patches.sh"
        for helper in set(re.findall(r'scripts/([a-z0-9_-]+\.py)', apply.read_text())):
            self.assertRegex(helper, r"^[a-z0-9_-]+\.py$",
                             f"helper {helper} does not match the installer fetch pattern")

    def test_serve_launcher_sets_pythonpath_and_module(self):
        text = self.serve_section()
        self.assertIn('"$BIN/mlx-omarchy-serve"', text)
        self.assertIn('export PYTHONPATH="$PREFIX\\${PYTHONPATH:+:\\$PYTHONPATH}"', text)
        self.assertIn('exec "$VENV/bin/python" -m mlx_omarchy_serve', text)

    def test_omarchy_launcher_carries_command_center_metadata(self):
        section = self.serve_section()
        self.assertIn("# omarchy:group=mlx", section)
        self.assertIn("# omarchy:name=serve", section)
        self.assertIn("# omarchy:summary=", section)
        self.assertIn("# omarchy:examples=", section)

    def test_existing_non_ours_omarchy_binary_is_never_overwritten(self):
        section = self.serve_section()
        self.assertIn("grep -qs 'mlx_omarchy_serve'", section)
        self.assertIn("left untouched", section)

    def test_unwritable_omarchy_bin_falls_back_to_user_bin_with_sudo_hint(self):
        section = self.serve_section()
        self.assertIn("sudo install -m 755 $BIN/omarchy-mlx-serve", section)

    def test_uninstall_covers_serve_launchers_and_omarchy_copy(self):
        text = installer_text()
        case_block = text[text.index("  --uninstall)"):text.index('  "") ;;')]
        self.assertIn('"$BIN/mlx-omarchy-serve"', case_block)
        self.assertIn('"$BIN/omarchy-mlx-serve"', case_block)
        self.assertIn("omarchy-mlx-serve", case_block)

    def test_no_background_scheduler_is_installed(self):
        text = installer_text()
        self.assertNotIn(".timer", text)
        services = {word.rsplit("/", 1)[-1] for line in text.splitlines()
                    for word in line.replace('"', " ").split() if word.endswith(".service")}
        self.assertLessEqual(services, {"mlx-omarchy-chat.service"})
        for line in text.splitlines():
            if "systemctl" in line and "command -v systemctl" not in line and not line.lstrip().startswith("#"):
                self.assertIn("--user", line)

    def test_periodic_checker_ships_only_with_the_release_that_has_it(self):
        # Migration contract: the checker is part of the serve package fetched
        # at the pinned tag. Old installs have no serve package and therefore
        # no checker; nothing retroactive, nothing polling in the background.
        section = self.serve_section()
        self.assertIn("serve/mlx_omarchy_serve/$serve_file", section)
        self.assertNotIn("cron", section)
        self.assertNotIn("systemd-run", section)


class AssistantInstallTests(unittest.TestCase):
    """Voice dependencies contract.

    The assistant package staging, static layout, launchers, and desktop
    entry are exercised for real by tests/test_serve_bootstrap.py (local
    curl shim against this checkout). Only the voice leg has no bootstrap
    coverage -- it is a pip/system-package branch -- so its installer
    contract stays checked here.
    """

    def test_voice_dependencies_are_optional_and_pinned(self):
        text = installer_text()
        self.assertIn("VOICE=0", text)
        self.assertIn("--voice) VOICE=1 ;;", text)
        self.assertIn("(supported: --ane, --voice, --system, --uninstall)", text)
        self.assertIn("MLX_AUDIO_VERSION=0.5.6", text)
        voice_section = text[text.index("if (( VOICE )); then"):]
        self.assertIn('--no-deps "mlx-audio==$MLX_AUDIO_VERSION"', voice_section)
        # Verified mlx-audio 0.5.6 runtime floors, extras excluded.
        for floor in ('"huggingface_hub>=1.0"', '"miniaudio>=1.61"', '"numpy>=1.26.4"',
                      '"scipy>=1.10.0"', '"sounddevice>=0.5.3"', '"tqdm>=4.67.1"',
                      '"transformers>=5.14.0"'):
            self.assertIn(floor, voice_section, f"missing voice dependency {floor}")
        # mlx-audio declares an mlx requirement; the custom wheel satisfies
        # it. Any pip install of upstream mlx here would clobber the vendored
        # build.
        self.assertNotIn(" mlx>=", voice_section)
        self.assertNotIn('"mlx"', voice_section)
        # Playback loads the system PortAudio library (sounddevice ships no
        # Linux binary), so the runtime package installs only under --voice.
        self.assertIn("--needed --noconfirm portaudio", voice_section)

class SocGateTests(unittest.TestCase):
    """The installer's SoC gate must accept the whole M1 family.

    Regression for a community M1 Max report (MacBook Pro 14-inch, M1 Max,
    2021): that machine reads device-tree compatible
    ["apple,j314c", "apple,t6001", "apple,arm-platform"], and installers
    older than v0.7.2 warned "this is not an Apple M1 (t8103)" on exactly
    that machine, which reads as a chip rejection.
    """

    USER_M1_MAX_COMPATIBLE = ["apple,j314c", "apple,t6001", "apple,arm-platform"]
    ACCEPTED = ["apple,t8103", "apple,t6000", "apple,t6001", "apple,t6002", "apple,t6021"]
    NOT_ACCEPTED = ["apple,t6031"]

    def soc_gate(self):
        match = re.search(r"grep -qE '(apple,t[^']+)'; then", installer_text())
        self.assertIsNotNone(match, "SoC gate grep missing from install.sh")
        return match.group(1)

    def warns(self, compatible_tokens):
        line = "\0".join(compatible_tokens).replace("\0", " ")
        return re.search(self.soc_gate(), line) is None

    def test_soc_gate_accepts_m1_family_and_the_reported_m1_max(self):
        self.assertFalse(self.warns(self.USER_M1_MAX_COMPATIBLE),
                         "the reported M1 Max (j314c + t6001) must install without a warning")
        for chip in self.ACCEPTED:
            self.assertFalse(self.warns([chip, "apple,arm-platform"]), chip)
        for chip in self.NOT_ACCEPTED:
            self.assertTrue(self.warns([chip, "apple,arm-platform"]),
                            f"{chip} is unverified and must keep the warning")

    def test_serve_catalog_arches_cover_the_m1_family(self):
        import importlib.util
        # catalog.py does `import mlx_omarchy_paths`, a top-level serve
        # module: put serve/ on the path or the exec fails under unittest.
        sys.path.insert(0, str(INSTALLER.parent / "serve"))
        spec = importlib.util.spec_from_file_location(
            "mlx_omarchy_serve_catalog",
            INSTALLER.parent / "serve/mlx_omarchy_serve/catalog.py")
        catalog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(catalog)
        for chip in ("t8103", "t6000", "t6001", "t6002", "t6021"):
            self.assertIn(chip, catalog.ARCHES)


if __name__ == "__main__":
    unittest.main()
