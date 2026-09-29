"""End-to-end bootstrap: install.sh sections -> prefix layout -> launchers -> CLI.

This is not an argparse test and not a text-inventory test. It runs the real
installer sections — launchers, the serve/laya/bonsai2 packages, the MLX Chat
assistant package with its static assets, the desktop entry, and the omarchy
command-center registration — verbatim under a fake HOME, then exercises the
installed launchers as child processes.

Only divergence from a real install: the release-tag curl fetches are
replaced by a local curl shim that copies the same release-tag paths out of
this checkout (tests never touch the network, and the release tag for this
commit does not exist while the branch is unmerged). A file the installer
fetches that the checkout does not ship aborts the install, exactly like a
failed download on a real machine. The wheel itself is not installed: the
venv is created without pip, so nothing here imports mlx, and the assistant
--help smoke proves the installed CLI runs without it.
"""

import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALLER = REPO_ROOT / "install.sh"

# Section boundaries in install.sh: everything the install stages, from the
# launchers through the assistant package, up to the GPU smoke test (which
# needs the real wheel). The launchers section also runs the wheel's
# mlx-omarchy-info lookup, which only the bootstrap class shims; the
# package-only slice skips it.
SECTION_START = "# 5. Demo and launchers."
PACKAGES_ONLY_START = "# 5b. Serve CLI"
SECTION_END = "# 6. Smoke test"


def installer_text():
    return INSTALLER.read_text()


def install_section(start=SECTION_START):
    """The real installation text from `start` to the smoke test."""
    text = installer_text()
    return text[text.index(start):text.index(SECTION_END)]


def checkout_files(package):
    """Every installable file of a package in this checkout, package-relative."""
    root = REPO_ROOT / "serve" / package
    if not root.is_dir():
        return None
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )


def assistant_source_dir():
    return REPO_ROOT / "serve" / "mlx_omarchy_assistant"


# Local curl replacement: serve release-tag paths from the checkout. A file
# the installer fetches that the checkout does not ship is a failed download
# and aborts the install (the sections run under set -e).
CURL_SHIM = """curl() {
  local url="" out=""
  while (( $# )); do
    case "$1" in
      -o) out="$2"; shift 2 ;;
      -*) shift ;;
      *) url="$1"; shift ;;
    esac
  done
  local rel="${url#*/"$VERSION"/}"
  local src="$MLX_OMARCHY_TEST_TREE/$rel"
  [[ -f "$src" ]] || { echo "curl: release tag has no $rel" >&2; return 22; }
  mkdir -p "$(dirname "$out")"
  cp "$src" "$out"
}
"""


def bootstrap_script(prefix, bin_dir, apps_dir, start=SECTION_START, with_python_shim=False):
    """A bash script running the real installer section text verbatim."""
    parts = [
        "set -euo pipefail\n",
        'say() { printf "== %s\\n" "$*"; }\n',
        'die() { echo "error: $*" >&2; exit 1; }\n',
        f'PREFIX="{prefix}"\n',
        f'VENV="{prefix}/venv"\n',
        f'BIN="{bin_dir}"\n',
        f'APPS="{apps_dir}"\n',
        'mkdir -p "$PREFIX" "$APPS"\n',
    ]
    if with_python_shim:
        # The real wheel provides mlx and mlx/bin/mlx-omarchy-info; this
        # host has neither. The info lookup runs during the launchers
        # section, so a stand-in python answers it and hands everything
        # else to the real interpreter.
        parts += [
            'mkdir -p "$VENV/bin"\n',
            "cat >\"$VENV/bin/python\" <<'SHIM'\n",
            "#!/bin/sh\n",
            'for arg in "$@"; do\n',
            '  case "$arg" in -I) echo /nonexistent/mlx/bin/mlx-omarchy-info; exit 0 ;; esac\n',
            "done\n",
            'exec python3 "$@"\n',
            "SHIM\n",
            'chmod +x "$VENV/bin/python"\n',
        ]
    parts.append(CURL_SHIM)
    parts.append(install_section(start))
    return "".join(parts)


class BootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        home = Path(cls.tmp.name)
        cls.home = home
        prefix = home / ".local/share/mlx-omarchy"
        bin_dir = home / ".local/bin"
        apps_dir = home / ".local/share/applications"
        bin_dir.mkdir(parents=True)
        omarchy_bin = bin_dir / "fake-omarchy-bin"
        omarchy_bin.mkdir()
        (omarchy_bin / "omarchy").write_text("#!/bin/sh\nexit 0\n")
        (omarchy_bin / "omarchy").chmod(0o755)
        # A stale entry from a pre-upgrade install: the new desktop entry
        # must replace it, not pile up next to it.
        apps_dir.mkdir(parents=True)
        (apps_dir / "mlx-omarchy-demo.desktop").write_text("[Desktop Entry]\n")
        script = bootstrap_script(prefix, bin_dir, apps_dir, with_python_shim=True)
        result = subprocess.run(
            ["bash", "-c", script],
            env=cls.install_env(home, prefix),
            capture_output=True, text=True, timeout=120)
        cls.install_output = result.stdout + result.stderr
        if result.returncode != 0:
            raise AssertionError(f"install section failed:\n{cls.install_output}")
        # The real venv replaces the section-5 python shim.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                        str(prefix / "venv")], check=True, timeout=120)
        cls.prefix = prefix
        cls.bin_dir = bin_dir
        cls.apps_dir = apps_dir
        cls.omarchy_bin = omarchy_bin
        cls.assistant_merged = assistant_source_dir().is_dir()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def install_env(cls, home, prefix):
        return {**os.environ,
                "HOME": str(home),
                "PATH": f"{home / '.local/bin/fake-omarchy-bin'}:{os.environ['PATH']}",
                "MLX_OMARCHY_HOME": str(prefix),
                "MLX_OMARCHY_TEST_TREE": str(REPO_ROOT),
                "REPO": "local.test/does-not-matter",
                "VERSION": "test"}

    def launcher_env(self):
        # HF_HOME is pinned inside the fake HOME so a real model cache on
        # the host can never satisfy (or leak into) a bootstrap run.
        return {**os.environ,
                "HOME": str(self.home),
                "HF_HOME": str(self.home / ".cache/huggingface"),
                "HF_HUB_OFFLINE": "",
                "MLX_OMARCHY_OFFLINE": "1"}

    def run_launcher(self, launcher, *args, cwd=None):
        return subprocess.run([str(launcher), *args], env=self.launcher_env(),
                              capture_output=True, text=True, timeout=120,
                              cwd=cwd)

    @classmethod
    def installed_files(cls, package):
        root = cls.prefix / package
        if not root.is_dir():
            return None
        return sorted(
            p.relative_to(cls.prefix).as_posix()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )

    def test_packages_land_in_prefix(self):
        # serve/laya/bonsai2 intentionally keep in-repo-only files (docs,
        # CONTRACT.md) out of the staged package, so the contract here is
        # installed subset of checkout: no staged file may be a phantom, and
        # a module missing from the prefix breaks the CLI tests below at
        # import time.
        for package in ("mlx_omarchy_serve", "mlx_omarchy_laya", "mlx_omarchy_bonsai2"):
            installed = self.installed_files(package)
            self.assertIsNotNone(installed, f"{package} missing from prefix")
            for rel in installed:
                self.assertTrue((REPO_ROOT / "serve" / rel).is_file(),
                                f"{rel} staged from a file the checkout does not ship")

    def test_assistant_tree_is_fully_installed(self):
        expected = checkout_files("mlx_omarchy_assistant")
        if expected is None:
            self.skipTest("assistant package not merged into this checkout yet")
        installed = self.installed_files("mlx_omarchy_assistant")
        self.assertIsNotNone(installed, "assistant package missing from prefix")
        for rel in expected:
            self.assertIn(f"mlx_omarchy_assistant/{rel}", installed,
                          f"installer never staged {rel}")
            if "/static/" in rel:
                self.assertTrue((self.prefix / "mlx_omarchy_assistant" / rel).stat().st_size > 0,
                                rel)

    def test_missing_module_file_breaks_the_install(self):
        """A fetched file absent from the checkout must abort the install.

        Negative control for the curl shim: the package-only section re-runs
        with one assistant module missing from the tree, and the install
        must die like it would on a failed download instead of staging a
        broken package.
        """
        source = assistant_source_dir()
        if not source.is_dir():
            self.skipTest("assistant package not merged into this checkout yet")
        victim = source / "pairs.py"
        backup = victim.read_bytes()
        victim.unlink()
        try:
            result = subprocess.run(
                ["bash", "-c",
                 bootstrap_script(self.prefix, self.bin_dir, self.apps_dir,
                                  start=PACKAGES_ONLY_START)],
                env=self.install_env(self.home, self.prefix),
                capture_output=True, text=True, timeout=120, check=False)
        finally:
            victim.write_bytes(backup)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("curl: release tag has no serve/mlx_omarchy_assistant/pairs.py",
                      result.stdout + result.stderr)

    def test_installed_assistant_launcher_help_needs_no_mlx(self):
        if not self.assistant_merged:
            self.skipTest("assistant package not merged into this checkout yet")
        # --help runs the installed CLI from an unrelated cwd: the launcher's
        # PYTHONPATH must resolve the staged package, and the import chain
        # must complete on a host without mlx.
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-chat", "--help",
                                   cwd=self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: mlx-omarchy-assistant", result.stdout)

    def test_installed_demo_launcher_help(self):
        if not self.assistant_merged:
            self.skipTest("demo attaches to the coordinator; package not merged yet")
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-demo", "--help",
                                   cwd=self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: mlx-omarchy-demo", result.stdout)

    def test_installed_demo_forwards_into_the_real_coordinator_parser(self):
        if not self.assistant_merged:
            self.skipTest("assistant package not merged into this checkout yet")
        # --yes without --pair must die in the real coordinator's parser with
        # its cross-validation message: proof the demo maps argv onto the
        # installed mlx_omarchy_assistant CLI, not a local stub.
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-demo", "--yes")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--yes requires an explicit --pair", result.stderr)

    def test_desktop_entry_replaces_the_stale_terminal_entry(self):
        entry = self.apps_dir / "mlx-omarchy-chat.desktop"
        self.assertTrue(entry.is_file(), "chat desktop entry missing")
        body = entry.read_text()
        exec_lines = [line for line in body.splitlines() if line.startswith("Exec=")]
        self.assertEqual(exec_lines, [f"Exec={self.bin_dir / 'mlx-omarchy-chat'}"])
        self.assertNotIn("omarchy-launch-floating-terminal-with-presentation", body)
        self.assertFalse((self.apps_dir / "mlx-omarchy-demo.desktop").exists(),
                         "stale terminal-demo entry survived the upgrade")
        unit = self.home / ".config" / "systemd" / "user" / "mlx-omarchy-chat.service"
        self.assertTrue(unit.is_file(), "login unit missing")
        self.assertIn("--resume --no-browser", unit.read_text())

    def test_launchers_are_executable(self):
        for name in ("mlx-omarchy-serve", "mlx-omarchy-demo", "mlx-omarchy-chat"):
            path = self.bin_dir / name
            self.assertTrue(path.is_file(), name)
            self.assertTrue(os.access(path, os.X_OK), name)
        registered = self.omarchy_bin / "omarchy-mlx-serve"
        self.assertTrue(registered.is_file())
        self.assertTrue(stat.S_IMODE(registered.stat().st_mode) & stat.S_IXUSR)

    def test_registered_binary_lives_beside_omarchy_with_metadata(self):
        registered = self.omarchy_bin / "omarchy-mlx-serve"
        body = registered.read_text()
        self.assertIn("# omarchy:group=mlx", body)
        self.assertIn("# omarchy:name=serve", body)
        self.assertIn("mlx_omarchy_serve", body)

    def test_installed_entrypoint_serves_the_bundled_catalog(self):
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "catalog", "list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("qwen3.8-27b-4bit", result.stdout)

    def test_installed_entrypoint_recommends_and_plans(self):
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve", "recommend")
        self.assertEqual(result.returncode, 0, result.stderr)
        # The bundled seed is honest: generation-qualified, HTTP-untested —
        # so recommend lists it but the auto-serve gate holds it back.
        self.assertIn("qwen3.8-27b-4bit", result.stdout)
        self.assertIn("pick:", result.stdout)
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "plan", "qwen3.8-27b-4bit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verdict:", result.stdout)

    def test_installed_entrypoint_refuses_unbudgetable_context(self):
        # Whatever the seed says (kv unknown, or a bounded max_tokens), an
        # absurd context must be refused at admission, never waved through.
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "plan", "qwen3.8-27b-4bit",
                                   "--context", "1000000000")
        self.assertTrue(
            result.returncode == 2 or "DOES NOT FIT" in result.stdout,
            f"expected refusal, got rc={result.returncode}: {result.stdout} {result.stderr}",
        )

    def test_omarchy_route_launcher_reaches_the_same_cli(self):
        result = self.run_launcher(self.omarchy_bin / "omarchy-mlx-serve",
                                   "catalog", "status")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bundled fallback", result.stdout)

    def test_refusal_path_fails_closed_offline(self):
        # Offline (env set by the launcher test env) + nothing complete on
        # disk: the gate must refuse rather than download. Which refusal
        # fires first depends on the host (memory-fit vs offline), so
        # assert the fail-closed contract, not the machine's memory.
        result = self.run_launcher(self.bin_dir / "mlx-omarchy-serve",
                                   "serve", "qwen3.8-27b-4bit", "--yes")
        self.assertEqual(result.returncode, 1)
        self.assertTrue(
            "refusing to download" in result.stderr
            or "does not fit" in result.stderr,
            f"expected a refusal, got: {result.stderr[-400:]}",
        )


class PatchFetchTests(unittest.TestCase):
    def test_installer_fetches_every_patch_the_apply_script_applies(self):
        # v0.7.4 and v0.7.5 both shipped an installer whose patch list lagged
        # the apply script, so fresh installs died with "patch file missing".
        # Run the real 4b fetch text under the release-tag curl shim, then ask
        # the fetched apply script which files it needs.
        text = installer_text()
        section = text[text.index("# 4b."):text.index('MLX_OMARCHY_CONV_RING="${MLX_OMARCHY_CONV_RING')]
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            script = "".join([
                "set -euo pipefail\n",
                'say() { :; }\n',
                f'PREFIX="{prefix}"\n',
                CURL_SHIM,
                section,
            ])
            env = {**os.environ, "MLX_OMARCHY_TEST_TREE": str(REPO_ROOT),
                   "REPO": "local.test/does-not-matter", "VERSION": "test"}
            result = subprocess.run(["bash", "-c", script], env=env, capture_output=True,
                                    text=True, timeout=60, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            applied = (prefix / "apply-mlx-lm-patches.sh").read_text()
            needed = set(re.findall(r"^\s*apply (\S+\.patch)", applied, re.MULTILINE))
            self.assertGreaterEqual(len(needed), 4)
            fetched = {p.name for p in (prefix / "patches").iterdir()}
            self.assertEqual(needed - fetched, set())


if __name__ == "__main__":
    unittest.main()
