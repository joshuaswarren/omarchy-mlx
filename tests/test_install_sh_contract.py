"""Installer refusal, uninstall, and ANE-smoke gate in a disposable HOME."""

import os
import platform
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
LAUNCHERS = ("mlx-omarchy", "mlx-omarchy-demo", "mlx-omarchy-info")

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
            artifacts = [prefix / "venv", apps / "mlx-omarchy-demo.desktop"]
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


class ServeCliContractTests(unittest.TestCase):
    """Serve CLI integration: package fetch, launchers, omarchy conventions."""

    def serve_section(self):
        text = installer_text()
        start = text.index('# 5b. Serve CLI')
        end = text.index('if command -v omarchy-launch-floating-terminal-with-presentation')
        return text[start:end]

    def test_serve_package_fetched_from_pinned_release_tag(self):
        section = self.serve_section()
        self.assertIn('"https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_serve/$serve_file"', section)
        for member in ("__init__.py", "catalog.py", "budget.py", "__main__.py", "_mlxlm_server.py", "catalog.json"):
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
        self.assertNotIn("systemctl", text)
        self.assertNotIn(".timer", text)
        self.assertNotIn(".service", text.replace("omarchy-launch-floating-terminal-with-presentation", ""))

    def test_periodic_checker_ships_only_with_the_release_that_has_it(self):
        # Migration contract: the checker is part of the serve package fetched
        # at the pinned tag. Old installs have no serve package and therefore
        # no checker; nothing retroactive, nothing polling in the background.
        section = self.serve_section()
        self.assertIn("serve/mlx_omarchy_serve/$serve_file", section)
        self.assertNotIn("cron", section)
        self.assertNotIn("systemd-run", section)


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
        spec = importlib.util.spec_from_file_location(
            "mlx_omarchy_serve_catalog",
            INSTALLER.parent / "serve/mlx_omarchy_serve/catalog.py")
        catalog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(catalog)
        for chip in ("t8103", "t6000", "t6001", "t6002", "t6021"):
            self.assertIn(chip, catalog.ARCHES)


if __name__ == "__main__":
    unittest.main()
