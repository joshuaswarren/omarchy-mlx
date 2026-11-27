"""Installer refusal, uninstall, and ANE-smoke gate in a disposable HOME."""

import hashlib
import http.server
import os
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

INSTALLER = Path(__file__).resolve().parents[1] / "install.sh"
LAUNCHERS = ("mlx-omarchy", "mlx-omarchy-demo", "mlx-omarchy-chat", "mlx-omarchy-info",
             "mlx-omarchy-parakeet")

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


def extract_vad_staging_defs():
    text = installer_text()
    match = re.search(
        r'^(VAD_PIN_COMMIT="[^"]+"\n.*?^stage_silero_vad\(\) \{\n.*?\n\})\nstage_silero_vad\n',
        text,
        re.M | re.S,
    )
    if match is None:
        raise AssertionError("Silero VAD staging block missing from install.sh")
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

    def _uninstall_prefix_with_state(self, tmp, *flags):
        """A prefix holding install files and user state; returns (prefix, state files)."""
        home = Path(tmp)
        prefix = home / "custom-prefix"
        assistant = prefix / "assistant"
        state = {
            assistant / "pair-locks" / "quality.json": '{"pair": "quality"}',
            assistant / "history" / "c1.json": '{"id": "c1"}',
            # the converted decision model a saved pair's lock points at (pairs.py _setup_locked)
            prefix / "models" / "decision-x" / "config.json": '{"converted": true}',
        }
        removed = [
            prefix / "venv" / "bin" / "python",
            prefix / "mlx_omarchy_assistant" / "server.py",
            prefix / "cache" / "recommended-catalog.json",
            assistant / "logs" / "download-x.log",
        ]
        for path, text in state.items():
            path.parent.mkdir(parents=True)
            path.write_text(text)
        for path in removed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        result = subprocess.run(
            ["bash", str(INSTALLER), "--uninstall", *flags],
            env={**os.environ, "HOME": str(home), "MLX_OMARCHY_HOME": str(prefix)},
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return prefix, state, removed, result

    def test_uninstall_keeps_chat_history_and_saved_pairs(self):
        # Issue #112: the data home is the install prefix, so uninstall used to
        # delete the pair lock that `mlx-omarchy-chat --resume` needs.
        with tempfile.TemporaryDirectory() as tmp:
            prefix, state, removed, result = self._uninstall_prefix_with_state(tmp)
            for path, text in state.items():
                self.assertEqual(path.read_text(), text, path)
            for path in removed:
                self.assertFalse(path.exists(), path)
            self.assertFalse((prefix / "venv").exists())
            self.assertFalse((prefix / "assistant" / "logs").exists())
            self.assertIn("--purge", result.stdout)

    def test_uninstall_purge_removes_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix, state, removed, _ = self._uninstall_prefix_with_state(tmp, "--purge")
            self.assertFalse(prefix.exists())
            for path in [*state, *removed]:
                self.assertFalse(path.exists(), path)

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
        for member in ("__init__.py", "jsonfile.py", "model.py", "sequence.py", "api.py",
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
        section = text.split("fetch_patch_assets()", 1)[1]
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
        start = text.index("  --uninstall)")
        case_block = text[start:text.index("\nesac\n", start)]
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


class PatcherCoverageTests(unittest.TestCase):
    """Every mlx-lm patcher in scripts/ must be accounted for by the installer.

    Regression class: v0.7.15 burned because the serve installer list missed
    a file, and the v0.7.25 draft cut caught the sibling defect one level
    up: scripts/patch-mlx-lm-qwen3-rope-norm.py sat in the tree while
    apply-mlx-lm-patches.sh (the file install.sh greps to derive the helper
    fetch list) never invoked it, so a fresh install would silently ship
    without the default-ON qwen3 dense rope-norm fold. This test fails when
    a scripts/patch-mlx-lm-*.py is neither invoked by the apply script nor
    explicitly classified below, when a classification names a file that no
    longer exists, and when an invoked patcher would be missed by the
    installer's derive-from-the-script fetch loop.
    """

    # Patchers intentionally NOT invoked by apply-mlx-lm-patches.sh, with
    # the reason each is safe to leave out of a fresh install. Audit table
    # with per-item fresh-install status: the release receipt.
    NOT_INVOKED = {
        "patch-mlx-lm-convring.py":
            "dev twin of patches/mlx-lm-convring.patch; conv-ring ships OFF and the .patch is the shipped path",
        "patch-mlx-lm-gdn.py":
            "dev twin of patches/mlx-lm-gated-delta-fast-route.patch (applied via apply())",
        "patch-mlx-lm-gdn-raw.py":
            "dev twin of patches/mlx-lm-gated-delta-raw.patch (applied via apply())",
        "patch-mlx-lm-qwen35-gdn-conv.py":
            "dev twin of patches/mlx-lm-qwen35-gdn-conv.patch (applied via apply())",
        "patch-mlx-lm-qwen35-gdn-norm.py":
            "dev twin of patches/mlx-lm-qwen35-gated-norm.patch (applied via apply())",
        "patch-mlx-lm-qwen3next-qgate-split.py":
            "experiment patcher, not part of any install path",
        "patch-mlx-lm-swiglu-eager.py":
            "experiment patcher, not part of any install path",
    }

    def invoked_patchers(self):
        apply = (INSTALLER.parent / "scripts" / "apply-mlx-lm-patches.sh").read_text()
        return set(re.findall(r"scripts/(patch-mlx-lm-[a-z0-9-]+\.py)", apply))

    def test_every_patcher_is_invoked_or_classified(self):
        patchers = sorted(p.name for p in
                          (INSTALLER.parent / "scripts").glob("patch-mlx-lm-*.py"))
        self.assertGreaterEqual(len(patchers), 4)
        invoked = self.invoked_patchers()
        unaccounted = [p for p in patchers
                       if p not in invoked and p not in self.NOT_INVOKED]
        self.assertEqual(
            unaccounted, [],
            "patchers neither invoked by apply-mlx-lm-patches.sh nor classified "
            f"in NOT_INVOKED (wire them into the install path or classify them): "
            f"{unaccounted}")
        stale = [p for p in self.NOT_INVOKED if p not in patchers]
        self.assertEqual(stale, [],
                         f"NOT_INVOKED classifies files that no longer exist: {stale}")

    def test_every_invoked_patcher_is_fetched_by_the_installer(self):
        text = installer_text()
        section = text.split("fetch_patch_assets()", 1)[1]
        self.assertIn("grep -oE 'scripts/[a-z0-9_-]+\\.py'", section)
        self.assertIn('"https://raw.githubusercontent.com/$REPO/$VERSION/$s"', section)
        # Simulate the installer's exact fetch grep over the apply script:
        # whatever it would download must cover every invoked patcher.
        fetched = set(re.findall(r"scripts/([a-z0-9_-]+\.py)",
                                 (INSTALLER.parent / "scripts" / "apply-mlx-lm-patches.sh").read_text()))
        missing = self.invoked_patchers() - fetched
        self.assertEqual(missing, set(),
                         f"invoked patchers the installer fetch loop would miss: {missing}")


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
        self.assertIn("(supported: --ane, --voice, --system, --uninstall, --patch-venv PATH)", text)
        self.assertIn("MLX_AUDIO_VERSION=0.5.6", text)
        voice_section = text[text.index("if (( VOICE )); then"):]
        self.assertIn('--no-deps "mlx-audio==$MLX_AUDIO_VERSION"', voice_section)
        # Verified mlx-audio 0.5.6 runtime floors, extras excluded.
        for floor in ('"huggingface_hub>=1.0"', '"miniaudio>=1.61"', '"numpy>=1.26.4"',
                      '"scipy>=1.10.0"', '"sounddevice>=0.5.3"', '"tqdm>=4.67.1"',
                      '"transformers>=5.14.0"'):
            self.assertIn(floor, voice_section, f"missing voice dependency {floor}")
        # Wake word: openwakeword installs --no-deps (its tflite-runtime
        # declaration has no wheel here and the ONNX path never imports it);
        # the verified ONNX-path floors must be pinned explicitly.
        self.assertIn('--no-deps "openwakeword==0.6.0"', voice_section)
        for floor in ('"onnxruntime>=1.17.0"', '"requests>=2.31.0"',
                      '"scikit-learn>=1.3.0"'):
            self.assertIn(floor, voice_section, f"missing wake word dependency {floor}")
        self.assertNotIn("tflite", voice_section)
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


class ArgumentParsingTests(unittest.TestCase):
    """Every argument is parsed: install flags combine in any order, and an
    unknown or misplaced argument fails before any network call or write."""

    def run_installer(self, home, *args, trace=False):
        """Returns (result, set of stubbed commands that ran).

        Failing curl and systemctl stubs record their calls, and no version
        is pinned: the install path stops at release resolution on every
        host, so nothing downloads, installs or touches the user session
        even on Apple Silicon.
        """
        stubs = Path(home) / "stubs"
        stubs.mkdir()
        for name in ("curl", "systemctl"):
            stub = stubs / name
            stub.write_text(f'#!/bin/sh\ntouch "{stubs}/{name}-called"\nexit 1\n')
            stub.chmod(0o755)
        env = {**os.environ, "HOME": str(home), "PATH": f"{stubs}:{os.environ['PATH']}",
               "PS4": "+ "}
        for name in ("MLX_OMARCHY_VERSION", "MLX_OMARCHY_HOME", "BASH_XTRACEFD"):
            env.pop(name, None)
        result = subprocess.run(
            ["bash", *(["-x"] if trace else []), str(INSTALLER), *args],
            env=env, cwd=home, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=30, check=False,
        )
        return result, {name for name in ("curl", "systemctl")
                        if (stubs / f"{name}-called").exists()}

    def test_unknown_argument_after_a_flag_fails_before_network(self):
        for args in (["--voice", "--bogus-flag"], ["--ane", "--bogus-flag"],
                     ["--voice", "--uninstall"], ["--ane", "--system"]):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as tmp:
                result, called = self.run_installer(tmp, *args)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"unknown option: {args[1]}", result.stderr)
                self.assertEqual(called, set())
                self.assertEqual(sorted(os.listdir(tmp)), ["stubs"])

    def test_install_flags_combine_in_either_order(self):
        for args in (["--ane", "--voice"], ["--voice", "--ane"]):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as tmp:
                result, _ = self.run_installer(tmp, *args, trace=True)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("unknown option", result.stderr)
                self.assertIn("+ ANE=1\n", result.stderr)
                self.assertIn("+ VOICE=1\n", result.stderr)
                self.assertEqual(sorted(os.listdir(tmp)), ["stubs"])

    def test_uninstall_with_unknown_argument_removes_nothing(self):
        for args in (["--uninstall", "--bogus-flag"], ["--uninstall", "--purge", "--bogus-flag"]):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as tmp:
                launcher = Path(tmp) / ".local/bin/mlx-omarchy"
                history = Path(tmp) / ".local/share/mlx-omarchy/assistant/history/c1.json"
                for path in (launcher, history):
                    path.parent.mkdir(parents=True)
                    path.touch()
                result, called = self.run_installer(tmp, *args)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("unknown option: --bogus-flag", result.stderr)
                self.assertTrue(launcher.exists())
                self.assertTrue(history.exists())
                self.assertEqual(called, set(), "the chat service must not be disabled")

    def test_patch_venv_rejects_an_extra_or_option_argument(self):
        # A directory named like the flag must not be taken as the venv.
        for args, bad in ((["venv", "--bogus-flag"], "--bogus-flag"), (["--system"], "--system")):
            with self.subTest(args=args), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "venv").mkdir()
                (Path(tmp) / "--system").mkdir()
                result, called = self.run_installer(tmp, "--patch-venv", *args)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(f"unknown option: {bad}", result.stderr)
                self.assertEqual(called, set())
                self.assertFalse((Path(tmp) / ".local").exists())
        with tempfile.TemporaryDirectory() as tmp:
            # The valid form still reaches release resolution.
            result, called = self.run_installer(tmp, "--patch-venv", tmp)
            self.assertIn("could not resolve the latest release", result.stderr)
            self.assertEqual(called, {"curl"})


class PipedInstallerTests(unittest.TestCase):
    """`curl ... | bash` runs install.sh from stdin: BASH_SOURCE[0] is unset under set -u."""

    def run_piped(self, *args):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", "-s", "--", *args],
                input=INSTALLER.read_text(),
                env={**os.environ, **PINNED_VERSION_ENV, "HOME": tmp},
                capture_output=True, text=True, timeout=30, check=False,
            )
            return result, sorted(os.listdir(tmp))

    def test_piped_script_does_not_trip_set_u_on_bash_source(self):
        # An unknown option exits before anything is installed, after the top-level code has run.
        result, left = self.run_piped("--no-such-option")
        self.assertNotIn("unbound variable", result.stderr)
        self.assertNotIn("null directory", result.stderr)
        self.assertIn("unknown option: --no-such-option", result.stderr)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(left, [], "the disposable HOME must stay empty")

    def test_piped_system_install_refuses_because_it_needs_a_checkout(self):
        result, left = self.run_piped("--system")
        self.assertNotIn("unbound variable", result.stderr)
        self.assertIn("--system runs from a checkout", result.stderr)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(left, [])

    def test_file_run_still_resolves_the_checkout(self):
        # Same --system call from the file: ROOT is set, so it gets past the checkout gate to the next check.
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(INSTALLER), "--system"],
                env={**os.environ, **PINNED_VERSION_ENV, "HOME": tmp},
                capture_output=True, text=True, timeout=30, check=False,
            )
        self.assertNotIn("runs from a checkout", result.stderr)
        self.assertIn("--system requires --vendor DIR and --lock FILE", result.stderr)
        self.assertEqual(result.returncode, 1)


class SileroVadStagingTests(unittest.TestCase):
    """Behavioral tests for install.sh's Silero VAD staging block.

    Every test runs the verbatim bash definitions extracted from install.sh
    against finite fixtures and a local loopback HTTP server (or a bounded
    local curl stub). No real network access and no endless inputs.
    """

    DATA = b"silero-vad-pinned-bytes-" * 64
    DIGEST = hashlib.sha256(DATA).hexdigest()

    def _start_server(self, routes):
        hits = []
        lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                with lock:
                    hits.append(self.path)
                spec = routes.get(self.path)
                if spec is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                mode = spec.get("mode", "fixed")
                status = spec.get("status", 200)
                body = spec.get("body", b"")
                headers = spec.get("headers", {})
                if mode == "fixed":
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass
                elif mode == "chunked":
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    step = spec.get("chunk_size", 4096)
                    try:
                        for i in range(0, len(body), step):
                            chunk = body[i:i + step]
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.write(b"0\r\n\r\n")
                    except OSError:
                        pass
                elif mode == "close":
                    self.protocol_version = "HTTP/1.0"
                    self.send_response(status)
                    for k, v in headers.items():
                        self.send_header(k, v)
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except OSError:
                        pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        return f"http://127.0.0.1:{port}", hits

    def _run_vad_script(
        self,
        prefix,
        body,
        *,
        url=None,
        sha=None,
        max_bytes=None,
        extra_env=None,
        timeout=20,
    ):
        defs = extract_vad_staging_defs()
        lines = [
            "set -euo pipefail",
            'say() { printf "SAY: %s\\n" "$*"; }',
            f"PREFIX={shlex.quote(str(prefix))}",
            defs,
        ]
        if sha is not None:
            lines.append(f"VAD_PIN_SHA={shlex.quote(sha)}")
        if max_bytes is not None:
            lines.append(f"VAD_MAX_BYTES={int(max_bytes)}")
        if url is not None:
            lines.append(f"VAD_URL={shlex.quote(url)}")
        lines.append(body)
        script = "\n".join(lines) + "\n"
        env = {**os.environ, **(extra_env or {})}
        return subprocess.run(
            ["bash", "-c", script],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _import_wake_word(self):
        serve_dir = str(ROOT / "serve")
        if serve_dir not in sys.path:
            sys.path.insert(0, serve_dir)
            self.addCleanup(
                lambda: sys.path.remove(serve_dir) if serve_dir in sys.path else None
            )
        from mlx_omarchy_assistant import wake_word
        return wake_word

    def test_install_sh_syntax_and_cap_match_runtime(self):
        syntax = subprocess.run(
            ["bash", "-n", str(INSTALLER)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        wake_word = self._import_wake_word()

        defs = extract_vad_staging_defs()
        match = re.search(r"^VAD_MAX_BYTES=(\d+)$", defs, re.M)
        self.assertIsNotNone(match, "VAD_MAX_BYTES missing from install.sh")
        self.assertEqual(int(match.group(1)), wake_word.MAX_MODEL_BYTES)

    def test_valid_download_stages_and_rerun_skips_fetch(self):
        base, hits = self._start_server({"/model": {"body": self.DATA}})
        wake_word = self._import_wake_word()

        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "prefix"
            res = self._run_vad_script(
                prefix,
                "umask 0022\nstage_silero_vad",
                url=f"{base}/model",
                sha=self.DIGEST,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("Silero VAD model staged", res.stdout)
            model = prefix / "vad" / "silero_vad.onnx"
            self.assertFalse(model.is_symlink())
            self.assertTrue(model.is_file())
            self.assertEqual(model.read_bytes(), self.DATA)
            self.assertEqual(stat.S_IMODE(model.stat().st_mode), 0o644)
            self.assertEqual(os.listdir(prefix / "vad"), ["silero_vad.onnx"])
            self.assertEqual(len(hits), 1)

            self.assertTrue(
                wake_word.pinned_file_ok(model, self.DIGEST, wake_word.WakeModelError)
            )

            res2 = self._run_vad_script(
                prefix,
                "stage_silero_vad",
                url=f"{base}/model",
                sha=self.DIGEST,
            )
            self.assertEqual(res2.returncode, 0, res2.stderr)
            self.assertEqual(res2.stdout, "")
            self.assertEqual(len(hits), 1, "verified regular file must skip re-download")

            # Respect restrictive umask (0077 -> 0600, matching wake_word._stage).
            prefix_priv = Path(tmp) / "prefix-priv"
            res_priv = self._run_vad_script(
                prefix_priv,
                "umask 0077\nstage_silero_vad",
                url=f"{base}/model",
                sha=self.DIGEST,
            )
            self.assertEqual(res_priv.returncode, 0, res_priv.stderr)
            priv_model = prefix_priv / "vad" / "silero_vad.onnx"
            self.assertEqual(stat.S_IMODE(priv_model.stat().st_mode), 0o600)

    def test_symlinked_model_even_to_valid_bytes_is_rejected_and_replaced(self):
        base, hits = self._start_server({"/model": {"body": self.DATA}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "prefix"
            vad_dir = prefix / "vad"
            vad_dir.mkdir(parents=True)
            target = root / "target.onnx"
            target.write_bytes(self.DATA)
            model = vad_dir / "silero_vad.onnx"
            os.symlink(target, model)

            check = self._run_vad_script(
                prefix,
                'vad_model_ok "$VAD_MODEL"',
                url=f"{base}/model",
                sha=self.DIGEST,
            )
            self.assertNotEqual(check.returncode, 0, "vad_model_ok must refuse symlinks")

            res = self._run_vad_script(
                prefix,
                "stage_silero_vad",
                url=f"{base}/model",
                sha=self.DIGEST,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("Silero VAD model staged", res.stdout)
            self.assertEqual(len(hits), 1)
            self.assertFalse(model.is_symlink())
            self.assertTrue(model.is_file())
            self.assertEqual(model.read_bytes(), self.DATA)
            self.assertEqual(target.read_bytes(), self.DATA)
            # Mutating the original symlink target must not affect staged model.
            target.write_bytes(b"mutated")
            self.assertEqual(model.read_bytes(), self.DATA)
            self.assertEqual(os.listdir(vad_dir), ["silero_vad.onnx"])

    def test_symlink_to_directory_and_dangling_symlink_are_atomically_replaced(self):
        base, _ = self._start_server({"/model": {"body": self.DATA}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Symlink pointing at a directory must be replaced, not entered.
            prefix1 = root / "p1"
            (prefix1 / "vad").mkdir(parents=True)
            victim_dir = root / "victim_dir"
            victim_dir.mkdir()
            model1 = prefix1 / "vad" / "silero_vad.onnx"
            os.symlink(victim_dir, model1)
            res1 = self._run_vad_script(
                prefix1, "stage_silero_vad", url=f"{base}/model", sha=self.DIGEST
            )
            self.assertEqual(res1.returncode, 0, res1.stderr)
            self.assertIn("Silero VAD model staged", res1.stdout)
            self.assertFalse(model1.is_symlink())
            self.assertEqual(model1.read_bytes(), self.DATA)
            self.assertEqual(os.listdir(victim_dir), [])

            # Dangling symlink must also be atomically replaced.
            prefix2 = root / "p2"
            (prefix2 / "vad").mkdir(parents=True)
            model2 = prefix2 / "vad" / "silero_vad.onnx"
            os.symlink(root / "does-not-exist", model2)
            res2 = self._run_vad_script(
                prefix2, "stage_silero_vad", url=f"{base}/model", sha=self.DIGEST
            )
            self.assertEqual(res2.returncode, 0, res2.stderr)
            self.assertFalse(model2.is_symlink())
            self.assertEqual(model2.read_bytes(), self.DATA)

    def test_predictable_part_symlink_is_never_followed_or_modified(self):
        base, _ = self._start_server({
            "/ok": {"body": self.DATA},
            "/bad": {"body": b"wrong-bytes"},
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / "victim"
            victim.write_bytes(b"keep me untouched")

            for endpoint, expect_staged in (("/ok", True), ("/bad", False)):
                with self.subTest(endpoint=endpoint):
                    prefix = root / f"p-{endpoint.strip('/')}"
                    vad_dir = prefix / "vad"
                    vad_dir.mkdir(parents=True)
                    part_link = vad_dir / "silero_vad.onnx.part"
                    os.symlink(victim, part_link)

                    res = self._run_vad_script(
                        prefix,
                        "stage_silero_vad",
                        url=f"{base}{endpoint}",
                        sha=self.DIGEST,
                    )
                    self.assertEqual(res.returncode, 0, res.stderr)
                    self.assertEqual(victim.read_bytes(), b"keep me untouched")
                    self.assertTrue(part_link.is_symlink())
                    entries = sorted(os.listdir(vad_dir))
                    if expect_staged:
                        self.assertEqual(entries, ["silero_vad.onnx", "silero_vad.onnx.part"])
                        self.assertEqual((vad_dir / "silero_vad.onnx").read_bytes(), self.DATA)
                    else:
                        self.assertEqual(entries, ["silero_vad.onnx.part"])

    def test_fifo_directory_and_oversized_sparse_model_never_hang_or_hash(self):
        base, hits = self._start_server({"/model": {"body": self.DATA}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # 1. FIFO (with no writer AND with a writer carrying valid pinned
            #    bytes) must not block vad_model_ok, must be refused, and is
            #    replaced cleanly.
            p_fifo = root / "p_fifo"
            (p_fifo / "vad").mkdir(parents=True)
            fifo_model = p_fifo / "vad" / "silero_vad.onnx"
            os.mkfifo(fifo_model)
            check_fifo = self._run_vad_script(
                p_fifo, 'vad_model_ok "$VAD_MODEL"', sha=self.DIGEST, timeout=5
            )
            self.assertNotEqual(check_fifo.returncode, 0)
            fifo_fd = os.open(fifo_model, os.O_RDWR | os.O_NONBLOCK)
            try:
                os.write(fifo_fd, self.DATA)
                check_fifo_loaded = self._run_vad_script(
                    p_fifo, 'vad_model_ok "$VAD_MODEL"', sha=self.DIGEST, timeout=5
                )
                self.assertNotEqual(
                    check_fifo_loaded.returncode,
                    0,
                    "FIFO carrying valid pinned bytes must still be refused",
                )
            finally:
                os.close(fifo_fd)
            res_fifo = self._run_vad_script(
                p_fifo, "stage_silero_vad", url=f"{base}/model", sha=self.DIGEST, timeout=10
            )
            self.assertEqual(res_fifo.returncode, 0, res_fifo.stderr)
            self.assertTrue(fifo_model.is_file() and not stat.S_ISFIFO(os.lstat(fifo_model).st_mode))
            self.assertEqual(fifo_model.read_bytes(), self.DATA)

            # 1b. Character device (/dev/null) with empty-string SHA-256 must
            #     still be rejected by the regular-file check.
            empty_sha = hashlib.sha256(b"").hexdigest()
            check_devnull = self._run_vad_script(
                root / "unused", 'vad_model_ok "/dev/null"', sha=empty_sha, timeout=5
            )
            self.assertNotEqual(
                check_devnull.returncode, 0, "/dev/null must be refused even when SHA matches empty"
            )

            # 2. Real directory at $VAD_MODEL must fail cleanly without fetching or moving inside.
            hits_before = len(hits)
            p_dir = root / "p_dir"
            dir_model = p_dir / "vad" / "silero_vad.onnx"
            dir_model.mkdir(parents=True)
            res_dir = self._run_vad_script(
                p_dir, "stage_silero_vad", url=f"{base}/model", sha=self.DIGEST, timeout=10
            )
            self.assertEqual(res_dir.returncode, 0, res_dir.stderr)
            self.assertIn("Silero VAD model path is a directory", res_dir.stdout)
            self.assertEqual(len(hits), hits_before, "directory target must not trigger download")
            self.assertTrue(dir_model.is_dir())
            self.assertEqual(os.listdir(dir_model), [])
            self.assertEqual(os.listdir(p_dir / "vad"), ["silero_vad.onnx"])

            # 3. Directory created mid-download (race): mv -Tf refuses and cleans up temp file.
            p_race = root / "p_race"
            (p_race / "vad").mkdir(parents=True)
            payload = root / "race_payload.bin"
            payload.write_bytes(self.DATA)
            stub_bin = root / "race_bin"
            stub_bin.mkdir()
            stub_curl = stub_bin / "curl"
            stub_curl.write_text(
                "#!/usr/bin/env bash\n"
                f"mkdir -p {shlex.quote(str(p_race / 'vad' / 'silero_vad.onnx'))}\n"
                f"cat {shlex.quote(str(payload))}\n"
            )
            stub_curl.chmod(0o755)
            res_race = self._run_vad_script(
                p_race,
                "stage_silero_vad",
                url="http://127.0.0.1:1/unused",
                sha=self.DIGEST,
                extra_env={"PATH": f"{stub_bin}:{os.environ['PATH']}"},
                timeout=10,
            )
            self.assertEqual(res_race.returncode, 0, res_race.stderr)
            self.assertIn("Silero VAD model download failed", res_race.stdout)
            self.assertEqual(os.listdir(p_race / "vad" / "silero_vad.onnx"), [])
            self.assertEqual(os.listdir(p_race / "vad"), ["silero_vad.onnx"])

            # 4. Huge sparse regular file (1 GiB logical, 0 blocks on disk) is rejected
            #    instantly via stat size check without hashing.
            p_sparse = root / "p_sparse"
            (p_sparse / "vad").mkdir(parents=True)
            sparse_model = p_sparse / "vad" / "silero_vad.onnx"
            with open(sparse_model, "wb") as handle:
                handle.truncate(1 << 30)
            check_sparse = self._run_vad_script(
                p_sparse, 'vad_model_ok "$VAD_MODEL"', sha=self.DIGEST, timeout=5
            )
            self.assertNotEqual(check_sparse.returncode, 0)
            res_sparse = self._run_vad_script(
                p_sparse, "stage_silero_vad", url=f"{base}/model", sha=self.DIGEST, timeout=10
            )
            self.assertEqual(res_sparse.returncode, 0, res_sparse.stderr)
            self.assertEqual(sparse_model.read_bytes(), self.DATA)

    def test_exact_size_boundary_unreadable_and_special_path_characters(self):
        base, _ = self._start_server({"/model": {"body": self.DATA}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Prefix with spaces, backslash, leading dash, and an embedded newline.
            prefix = root / "-odd dir\\line1\nline2 with spaces"
            res_ok = self._run_vad_script(
                prefix,
                "stage_silero_vad",
                url=f"{base}/model",
                sha=self.DIGEST,
                max_bytes=len(self.DATA),
            )
            self.assertEqual(res_ok.returncode, 0, res_ok.stderr)
            self.assertIn("Silero VAD model staged", res_ok.stdout)
            model = prefix / "vad" / "silero_vad.onnx"
            self.assertEqual(model.read_bytes(), self.DATA)

            # One byte below the file's size must reject it before hashing.
            check_over = self._run_vad_script(
                prefix,
                'vad_model_ok "$VAD_MODEL"',
                sha=self.DIGEST,
                max_bytes=len(self.DATA) - 1,
            )
            self.assertNotEqual(check_over.returncode, 0)

            # Relative path beginning with '-' passed directly to vad_model_ok.
            rel_dir = root / "rel_dash"
            rel_dir.mkdir()
            (rel_dir / "-silero_vad.onnx").write_bytes(self.DATA)
            res_rel = self._run_vad_script(
                root / "unused",
                f"cd {shlex.quote(str(rel_dir))}\nvad_model_ok '-silero_vad.onnx'",
                sha=self.DIGEST,
                max_bytes=len(self.DATA),
            )
            self.assertEqual(res_rel.returncode, 0, res_rel.stderr)

            # Unreadable regular file (non-root) must fail vad_model_ok cleanly
            # with empty stderr (even if pinned to the empty-string SHA-256)
            # and be atomically replaced by stage_silero_vad.
            if os.geteuid() != 0:
                p_unr = root / "unreadable"
                (p_unr / "vad").mkdir(parents=True)
                unr = p_unr / "vad" / "silero_vad.onnx"
                unr.write_bytes(b"secret")
                unr.chmod(0)
                self.addCleanup(lambda: unr.exists() and unr.chmod(0o600))
                empty_sha = hashlib.sha256(b"").hexdigest()
                chk_unr = self._run_vad_script(
                    p_unr,
                    'vad_model_ok "$VAD_MODEL"',
                    sha=empty_sha,
                )
                self.assertNotEqual(chk_unr.returncode, 0)
                self.assertEqual(chk_unr.stderr, "")
                rep_unr = self._run_vad_script(
                    p_unr,
                    "stage_silero_vad",
                    url=f"{base}/model",
                    sha=self.DIGEST,
                )
                self.assertEqual(rep_unr.returncode, 0, rep_unr.stderr)
                self.assertIn("Silero VAD model staged", rep_unr.stdout)
                self.assertEqual(unr.read_bytes(), self.DATA)

    def test_oversized_downloads_and_redirects_across_transfer_modes(self):
        cap = 4096
        over = b"A" * (cap + 1)
        over_large = b"B" * (cap + 65536)
        over_sha = hashlib.sha256(over).hexdigest()
        base, _ = self._start_server({
            "/declared": {"mode": "fixed", "body": over},
            "/chunked": {"mode": "chunked", "body": over_large, "chunk_size": 1024},
            "/close": {"mode": "close", "body": over},
            "/redirect-ok": {
                "mode": "fixed",
                "status": 302,
                "headers": {"Location": "/ok"},
                "body": b"moved",
            },
            "/redirect-over-body": {
                "mode": "fixed",
                "status": 302,
                "headers": {"Location": "/ok"},
                "body": over,
            },
            "/redirect-to-declared-over": {
                "mode": "fixed",
                "status": 302,
                "headers": {"Location": "/declared"},
                "body": b"moved",
            },
            "/redirect-to-chunked-over": {
                "mode": "fixed",
                "status": 302,
                "headers": {"Location": "/chunked"},
                "body": b"moved",
            },
            "/ok": {"mode": "fixed", "body": self.DATA},
        })
        # Valid 302 redirect to a valid final response must stage cleanly.
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "p-redir-ok"
            res_ok = self._run_vad_script(
                prefix,
                "stage_silero_vad",
                url=f"{base}/redirect-ok",
                sha=self.DIGEST,
                max_bytes=cap,
            )
            self.assertEqual(res_ok.returncode, 0, res_ok.stderr)
            self.assertIn("Silero VAD model staged", res_ok.stdout)
            self.assertEqual((prefix / "vad" / "silero_vad.onnx").read_bytes(), self.DATA)

        for endpoint in (
            "/declared",
            "/chunked",
            "/close",
            "/redirect-over-body",
            "/redirect-to-declared-over",
            "/redirect-to-chunked-over",
        ):
            with self.subTest(endpoint=endpoint), tempfile.TemporaryDirectory() as tmp:
                prefix = Path(tmp) / "p"
                res = self._run_vad_script(
                    prefix,
                    "stage_silero_vad",
                    url=f"{base}{endpoint}",
                    sha=over_sha if endpoint in ("/declared", "/close", "/redirect-to-declared-over") else self.DIGEST,
                    max_bytes=cap,
                )
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertIn("Silero VAD model download failed", res.stdout)
                self.assertFalse((prefix / "vad" / "silero_vad.onnx").exists())
                self.assertEqual(os.listdir(prefix / "vad"), [])

        # Also simulate a curl that ignores --max-filesize on unknown-length
        # bodies: head -c (cap + 1) + vad_model_ok must still cap and refuse
        # exact cap+1 (even if VAD_PIN_SHA matches the first cap bytes) and cut
        # off a multi-megabyte finite stream before the writer finishes.
        cap_prefix_sha = hashlib.sha256(b"Z" * cap).hexdigest()
        for stream_len, pin_sha, expect_truncated in (
            (cap + 1, hashlib.sha256(b"Z" * (cap + 1)).hexdigest(), False),
            (cap + 1, cap_prefix_sha, False),
            (2 << 20, hashlib.sha256(b"Z" * (2 << 20)).hexdigest(), True),
        ):
            with self.subTest(stub_curl_bytes=stream_len, pin=pin_sha[:8]), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prefix = root / "p"
                finished_marker = root / "wrote-all"
                stub_bin = root / "bin"
                stub_bin.mkdir()
                stub_curl = stub_bin / "curl"
                stub_curl.write_text(
                    "#!/usr/bin/env bash\n"
                    f"python3 - {stream_len} {shlex.quote(str(finished_marker))} <<'PY'\n"
                    "import pathlib, sys\n"
                    "rem = int(sys.argv[1])\n"
                    "marker = pathlib.Path(sys.argv[2])\n"
                    'chunk = b"Z" * 4096\n'
                    "try:\n"
                    "    while rem > 0:\n"
                    "        n = min(rem, len(chunk))\n"
                    "        sys.stdout.buffer.write(chunk[:n])\n"
                    "        sys.stdout.buffer.flush()\n"
                    "        rem -= n\n"
                    '    marker.write_text("done")\n'
                    "except BrokenPipeError:\n"
                    "    sys.exit(23)\n"
                    "PY\n"
                )
                stub_curl.chmod(0o755)
                res = self._run_vad_script(
                    prefix,
                    "stage_silero_vad",
                    url="http://127.0.0.1:1/unused",
                    sha=pin_sha,
                    max_bytes=cap,
                    extra_env={"PATH": f"{stub_bin}:{os.environ['PATH']}"},
                )
                self.assertEqual(res.returncode, 0, res.stderr)
                self.assertIn("Silero VAD model download failed", res.stdout)
                self.assertFalse((prefix / "vad" / "silero_vad.onnx").exists())
                self.assertEqual(os.listdir(prefix / "vad"), [])
                if expect_truncated:
                    self.assertFalse(
                        finished_marker.exists(),
                        "head -c must cut off an unbounded curl stream before it finishes",
                    )

        # A curl invocation that exits non-zero even after emitting valid bytes
        # (e.g. truncated/error exit) must fail via the subshell's own pipefail
        # (even when the caller disabled pipefail) and never stage.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "p"
            args_log = root / "curl-args"
            payload = root / "valid_payload.bin"
            payload.write_bytes(self.DATA)
            stub_bin = root / "bin"
            stub_bin.mkdir()
            stub_curl = stub_bin / "curl"
            stub_curl.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$@\" > {shlex.quote(str(args_log))}\n"
                f"cat {shlex.quote(str(payload))}\n"
                "exit 22\n"
            )
            stub_curl.chmod(0o755)
            res = self._run_vad_script(
                prefix,
                "set +o pipefail\nstage_silero_vad",
                url="http://127.0.0.1:1/unused",
                sha=self.DIGEST,
                extra_env={"PATH": f"{stub_bin}:{os.environ['PATH']}"},
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("Silero VAD model download failed", res.stdout)
            self.assertEqual(os.listdir(prefix / "vad"), [])
            logged = args_log.read_text().splitlines()
            self.assertIn("--max-filesize", logged)
            self.assertIn("--max-time", logged)

    def test_bad_sha_and_misplaced_or_symlinked_vad_dir_fail_cleanly(self):
        base, hits = self._start_server({
            "/ok": {"body": self.DATA},
            "/bad": {"body": b"corrupt payload"},
        })
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Absent target + bad sha -> stays absent, no temp files.
            p1 = root / "p1"
            res1 = self._run_vad_script(
                p1, "stage_silero_vad", url=f"{base}/bad", sha=self.DIGEST
            )
            self.assertEqual(res1.returncode, 0, res1.stderr)
            self.assertIn("Silero VAD model download failed", res1.stdout)
            self.assertEqual(os.listdir(p1 / "vad"), [])

            # Existing valid target + bad server -> untouched, never replaced.
            p2 = root / "p2"
            (p2 / "vad").mkdir(parents=True)
            valid = p2 / "vad" / "silero_vad.onnx"
            valid.write_bytes(self.DATA)
            res2 = self._run_vad_script(
                p2, "stage_silero_vad", url=f"{base}/bad", sha=self.DIGEST
            )
            self.assertEqual(res2.returncode, 0, res2.stderr)
            self.assertEqual(res2.stdout, "")
            self.assertEqual(valid.read_bytes(), self.DATA)
            self.assertEqual(os.listdir(p2 / "vad"), ["silero_vad.onnx"])

            # Regular file at $PREFIX/vad -> skips staging cleanly without fetching.
            hits_before = len(hits)
            p3 = root / "p3"
            p3.mkdir()
            (p3 / "vad").write_bytes(b"not a directory")
            res3 = self._run_vad_script(
                p3, "stage_silero_vad", url=f"{base}/ok", sha=self.DIGEST
            )
            self.assertEqual(res3.returncode, 0, res3.stderr)
            self.assertIn("skipping VAD model staging", res3.stdout)
            self.assertEqual((p3 / "vad").read_bytes(), b"not a directory")
            self.assertEqual(len(hits), hits_before)

            # Symlinked $PREFIX/vad -> external empty dir: must never stage there.
            ext_empty = root / "ext_empty"
            ext_empty.mkdir()
            p4 = root / "p4"
            p4.mkdir()
            os.symlink(ext_empty, p4 / "vad")
            res4 = self._run_vad_script(
                p4, "stage_silero_vad", url=f"{base}/ok", sha=self.DIGEST
            )
            self.assertEqual(res4.returncode, 0, res4.stderr)
            self.assertIn("skipping VAD model staging", res4.stdout)
            self.assertEqual(os.listdir(ext_empty), [])
            self.assertEqual(len(hits), hits_before)

            # Symlinked $PREFIX/vad -> external dir holding valid silero_vad.onnx:
            # must be refused (neither accepted as staged nor overwritten).
            ext_valid = root / "ext_valid"
            ext_valid.mkdir()
            (ext_valid / "silero_vad.onnx").write_bytes(self.DATA)
            p5 = root / "p5"
            p5.mkdir()
            os.symlink(ext_valid, p5 / "vad")
            res5 = self._run_vad_script(
                p5, "stage_silero_vad", url=f"{base}/ok", sha=self.DIGEST
            )
            self.assertEqual(res5.returncode, 0, res5.stderr)
            self.assertIn("skipping VAD model staging", res5.stdout)
            self.assertEqual((ext_valid / "silero_vad.onnx").read_bytes(), self.DATA)
            self.assertEqual(os.listdir(ext_valid), ["silero_vad.onnx"])
            self.assertEqual(len(hits), hits_before)

            # Symlinked $PREFIX/vad -> regular file and dangling symlink.
            ext_file = root / "ext_file"
            ext_file.write_bytes(b"preserve")
            for label, target in (("file", ext_file), ("dangling", root / "missing_dir")):
                with self.subTest(vad_dir_symlink=label):
                    p_sym = root / f"p_sym_{label}"
                    p_sym.mkdir()
                    os.symlink(target, p_sym / "vad")
                    res_sym = self._run_vad_script(
                        p_sym, "stage_silero_vad", url=f"{base}/ok", sha=self.DIGEST
                    )
                    self.assertEqual(res_sym.returncode, 0, res_sym.stderr)
                    self.assertIn("skipping VAD model staging", res_sym.stdout)
                    self.assertEqual(ext_file.read_bytes(), b"preserve")
                    self.assertFalse((root / "missing_dir").exists())
                    self.assertEqual(len(hits), hits_before)

    def test_preserves_caller_traps_and_cancellation_reaps_download_without_publishing(self):
        base, _ = self._start_server({"/model": {"body": self.DATA}})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Parent EXIT trap and custom INT/TERM traps must survive, and
            #    cancel_silero_vad must not leak into global function scope.
            p1 = root / "p1"
            marker = root / "parent trap ran"
            res = self._run_vad_script(
                p1,
                "\n".join([
                    'trap \'touch "$MARKER_PATH"\' EXIT',
                    "trap 'echo MYINT' INT",
                    "trap 'echo MYTERM' TERM",
                    "stage_silero_vad",
                    "trap -p INT",
                    "trap -p TERM",
                    "if declare -F cancel_silero_vad >/dev/null; then echo LEAKED_HELPER; fi",
                ]),
                url=f"{base}/model",
                sha=self.DIGEST,
                extra_env={"MARKER_PATH": str(marker)},
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue(marker.exists(), "stage_silero_vad clobbered parent EXIT trap")
            self.assertIn("echo MYINT", res.stdout)
            self.assertIn("echo MYTERM", res.stdout)
            self.assertNotIn("LEAKED_HELPER", res.stdout)

            # 2. SIGINT and SIGTERM delivered to the staging subshell, to the
            #    parent installer PID alone, or to the process group must
            #    terminate the download worker (preventing stderr pipe hangs),
            #    prevent mv from publishing, clean up the hidden temp file, and
            #    preserve the parent EXIT trap.
            payload = root / "cancel_payload.bin"
            payload.write_bytes(self.DATA)
            defs = extract_vad_staging_defs()

            for target, sig in (
                ("subshell", signal.SIGINT),
                ("subshell", signal.SIGTERM),
                ("parent", signal.SIGTERM),
                ("parent", signal.SIGINT),
                ("pg", signal.SIGTERM),
                ("pg", signal.SIGINT),
            ):
                with self.subTest(target=target, sig=sig.name):
                    case_dir = root / f"cancel_{target}_{sig.name}"
                    prefix = case_dir / "prefix"
                    stub_bin = case_dir / "bin"
                    stub_bin.mkdir(parents=True)
                    ready = case_dir / "ready"
                    curl_pid_file = case_dir / "curl-pid"
                    mv_called = case_dir / "mv-called"
                    parent_marker = case_dir / "parent-exit"

                    # Wrap mv to detect any publication attempt.
                    real_mv = shutil.which("mv") or "/usr/bin/mv"
                    stub_mv = stub_bin / "mv"
                    stub_mv.write_text(
                        "#!/usr/bin/env bash\n"
                        f"touch {shlex.quote(str(mv_called))}\n"
                        f'exec {shlex.quote(real_mv)} "$@"\n'
                    )
                    stub_mv.chmod(0o755)

                    stub_curl = stub_bin / "curl"
                    if target == "subshell":
                        # Signal the staging subshell (grandparent of curl via
                        # the inner download pipeline subshell) and emit valid
                        # bytes cleanly so only signal handling stops mv.
                        stub_curl.write_text(
                            "#!/usr/bin/env bash\n"
                            f'echo "$$" > {shlex.quote(str(curl_pid_file))}\n'
                            "sub_pid=$(ps -o ppid= -p \"$PPID\" | tr -d ' ')\n"
                            f"kill -{sig.name} \"$sub_pid\"\n"
                            "sleep 30\n"
                            f"cat {shlex.quote(str(payload))}\n"
                        )
                    else:
                        stub_curl.write_text(
                            "#!/usr/bin/env bash\n"
                            f'echo "$$" > {shlex.quote(str(curl_pid_file))}\n'
                            f"touch {shlex.quote(str(ready))}\n"
                            "sleep 30\n"
                            f"cat {shlex.quote(str(payload))}\n"
                        )
                    stub_curl.chmod(0o755)

                    script = "\n".join([
                        "set -euo pipefail",
                        'say() { printf "SAY: %s\\n" "$*"; }',
                        'trap \'touch "$MARKER_PATH"\' EXIT',
                        f"PREFIX={shlex.quote(str(prefix))}",
                        defs,
                        f"VAD_PIN_SHA={shlex.quote(self.DIGEST)}",
                        'VAD_URL="http://127.0.0.1:1/slow"',
                        "stage_silero_vad",
                    ]) + "\n"
                    env = {
                        **os.environ,
                        "PATH": f"{stub_bin}:{os.environ['PATH']}",
                        "MARKER_PATH": str(parent_marker),
                    }
                    proc = subprocess.Popen(
                        ["bash", "-c", script],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                    try:
                        if target != "subshell":
                            deadline = time.monotonic() + 5.0
                            while not ready.exists() and time.monotonic() < deadline:
                                time.sleep(0.02)
                            self.assertTrue(ready.exists(), "slow curl stub never started")
                            if target == "parent":
                                os.kill(proc.pid, sig)
                            else:
                                os.killpg(proc.pid, sig)
                        # Must exit promptly with stderr=PIPE (no orphan holding pipe open).
                        stdout, _stderr = proc.communicate(timeout=5)
                        curl_pid = int(curl_pid_file.read_text().strip())
                        reap_deadline = time.monotonic() + 1.0
                        curl_dead = False
                        while time.monotonic() < reap_deadline:
                            try:
                                os.kill(curl_pid, 0)
                            except ProcessLookupError:
                                curl_dead = True
                                break
                            try:
                                stat_text = Path(f"/proc/{curl_pid}/stat").read_text()
                                if stat_text.rsplit(")", 1)[1].split()[0] == "Z":
                                    curl_dead = True
                                    break
                            except OSError:
                                curl_dead = True
                                break
                            time.sleep(0.02)
                        self.assertTrue(
                            curl_dead,
                            f"orphaned curl pid {curl_pid} survived cancellation",
                        )
                    finally:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        if proc.poll() is None:
                            proc.wait(timeout=5)
                        if proc.stdout:
                            proc.stdout.close()
                        if proc.stderr:
                            proc.stderr.close()

                    self.assertFalse(mv_called.exists(), "cancelled staging must never reach mv")
                    self.assertTrue(parent_marker.exists(), "parent EXIT trap must still run")
                    self.assertEqual(os.listdir(prefix / "vad"), [], "cancelled staging left temp files")
                    if target == "subshell":
                        self.assertEqual(proc.returncode, 0)
                        self.assertIn("Silero VAD model download failed", stdout)
                    else:
                        self.assertEqual(proc.returncode, -int(sig))

            # 3. Early staging failure (before download launch) must never kill
            #    an unrelated caller background job inherited via $!.
            if os.geteuid() != 0:
                ro_prefix = root / "ro_prefix"
                ro_prefix.mkdir()
                ro_prefix.chmod(0o555)
                self.addCleanup(lambda: ro_prefix.exists() and ro_prefix.chmod(0o755))
                res_bg = self._run_vad_script(
                    ro_prefix,
                    "\n".join([
                        "sleep 0.5 &",
                        "caller_bg=$!",
                        "stage_silero_vad",
                        "wait \"$caller_bg\"",
                        "echo CALLER_BG_RC=$?",
                    ]),
                    url=f"{base}/model",
                    sha=self.DIGEST,
                )
                self.assertEqual(res_bg.returncode, 0, res_bg.stderr)
                self.assertIn("CALLER_BG_RC=0", res_bg.stdout)
                self.assertIn("Silero VAD model download failed", res_bg.stdout)

            # 4. Cancellation with a returning (non-exiting) caller TERM trap
            #    must invoke the caller trap, never stage, and never print
            #    "Silero VAD model staged".
            ret_dir = root / "ret_trap"
            ret_prefix = ret_dir / "prefix"
            ret_bin = ret_dir / "bin"
            ret_bin.mkdir(parents=True)
            ret_ready = ret_dir / "ready"
            ret_curl = ret_bin / "curl"
            ret_curl.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {shlex.quote(str(ret_ready))}\n"
                "sleep 30\n"
                f"cat {shlex.quote(str(payload))}\n"
            )
            ret_curl.chmod(0o755)
            ret_script = "\n".join([
                "set -euo pipefail",
                'say() { printf "SAY: %s\\n" "$*"; }',
                "trap 'echo CALLER_TERM_RETURNED' TERM",
                f"PREFIX={shlex.quote(str(ret_prefix))}",
                defs,
                f"VAD_PIN_SHA={shlex.quote(self.DIGEST)}",
                'VAD_URL="http://127.0.0.1:1/slow"',
                "stage_silero_vad",
                "echo AFTER_STAGE",
            ]) + "\n"
            ret_proc = subprocess.Popen(
                ["bash", "-c", ret_script],
                env={**os.environ, "PATH": f"{ret_bin}:{os.environ['PATH']}"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 5.0
                while not ret_ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ret_ready.exists(), "slow curl stub never started")
                os.kill(ret_proc.pid, signal.SIGTERM)
                ret_out, ret_err = ret_proc.communicate(timeout=5)
            finally:
                try:
                    os.killpg(ret_proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if ret_proc.poll() is None:
                    ret_proc.wait(timeout=5)
                if ret_proc.stdout:
                    ret_proc.stdout.close()
                if ret_proc.stderr:
                    ret_proc.stderr.close()
            self.assertEqual(ret_proc.returncode, 0, ret_err)
            self.assertIn("CALLER_TERM_RETURNED", ret_out)
            self.assertIn("AFTER_STAGE", ret_out)
            self.assertNotIn("Silero VAD model staged", ret_out)
            self.assertEqual(os.listdir(ret_prefix / "vad"), [])

            # 5. A first TERM arriving right before restoring old_int or between
            #    restoring old_int and old_term must never be dropped, must
            #    invoke the caller's TERM handler, and must preserve the
            #    restored INT handler.
            for label, anchor in (
                ("before_old_int", '  eval "${old_int:-trap - INT}"'),
                ("between_restores", '  eval "${old_term:-trap - TERM}"'),
            ):
                with self.subTest(restore_boundary=label):
                    self.assertIn(anchor, defs)
                    defs_injected = defs.replace(
                        anchor,
                        '  kill -TERM "$BASHPID"\n' + anchor,
                        1,
                    )
                    bp = root / f"restore_{label}"
                    bscript = "\n".join([
                        "set -euo pipefail",
                        'say() { printf "SAY: %s\\n" "$*"; }',
                        "trap 'echo RESTORED_INT_OK' INT",
                        "trap 'echo RESTORED_TERM_OK' TERM",
                        f"PREFIX={shlex.quote(str(bp))}",
                        defs_injected,
                        f"VAD_PIN_SHA={shlex.quote(self.DIGEST)}",
                        f"VAD_URL={shlex.quote(f'{base}/model')}",
                        "stage_silero_vad",
                        "trap -p INT",
                        "trap -p TERM",
                    ]) + "\n"
                    bres = subprocess.run(
                        ["bash", "-c", bscript],
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                    self.assertEqual(bres.returncode, 0, bres.stderr)
                    self.assertIn("RESTORED_TERM_OK", bres.stdout.splitlines())
                    self.assertNotIn("Silero VAD model staged", bres.stdout)
                    self.assertIn("echo RESTORED_INT_OK", bres.stdout)
                    self.assertIn("echo RESTORED_TERM_OK", bres.stdout)


if __name__ == "__main__":
    unittest.main()
