"""mlx-omarchy-retire-legacy contract in a disposable HOME.

Dry-run by default; --yes removes the legacy venv, launchers, user unit,
desktop entries, and marker-matched command-center shims; user data and
~/.cache/huggingface survive unless --purge-data; reruns are no-ops.
"""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RETIRE = REPO / "packaging" / "mlx-omarchy-retire-legacy"
LAUNCHERS = (
    "mlx-omarchy",
    "mlx-omarchy-demo",
    "mlx-omarchy-chat",
    "mlx-omarchy-info",
    "mlx-omarchy-serve",
    "omarchy-mlx-serve",
)
UNIT = "mlx-omarchy-chat.service"
DESKTOP = "mlx-omarchy-chat.desktop"


class RetireLegacyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name) / "home"
        self.prefix = self.home / ".local" / "share" / "mlx-omarchy"
        self.bin_dir = self.home / ".local" / "bin"
        self.apps = self.home / ".local" / "share" / "applications"
        self.unit_dir = self.home / ".config" / "systemd" / "user"
        self.center = Path(self._tmp.name) / "center"  # fake command-center dir
        for directory in (self.prefix / "venv", self.prefix / "assistant",
                          self.bin_dir, self.apps, self.unit_dir, self.center):
            directory.mkdir(parents=True)
        (self.prefix / "venv" / "pyvenv.cfg").write_text("[config]\n")
        (self.prefix / "assistant" / "history.json").write_text("{}\n")
        for launcher in LAUNCHERS:
            (self.bin_dir / launcher).write_text("#!/bin/sh\n")
        (self.apps / DESKTOP).write_text("[Desktop Entry]\n")
        (self.unit_dir / UNIT).write_text("[Unit]\n")
        self.center_shim = self.center / "omarchy-mlx-serve"
        self.center_shim.write_text(
            "#!/usr/bin/env bash\nexport PYTHONPATH=.../mlx_omarchy_serve\n"
        )
        self.center_shim.chmod(0o755)
        self.cache = self.home / ".cache" / "huggingface"
        self.cache.mkdir(parents=True)
        (self.cache / "weights.bin").write_text("weights\n")

    def tearDown(self):
        self._tmp.cleanup()

    def run_retire(self, *args, extra_env=None):
        env = {
            **os.environ,
            "HOME": str(self.home),
            "PATH": f"{self.center}:{os.environ['PATH']}",
            "MLX_OMARCHY_HOME": str(self.prefix),
        }
        env.update(extra_env or {})
        return subprocess.run(
            ["bash", str(RETIRE), *args],
            env=env, capture_output=True, text=True,
        )

    def test_dry_run_lists_everything_and_removes_nothing(self):
        result = self.run_retire()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"would remove venv: {self.prefix}/venv", result.stdout)
        self.assertIn(f"{self.bin_dir}/mlx-omarchy-chat", result.stdout)
        self.assertIn(f"would remove user unit: {self.unit_dir / UNIT}", result.stdout)
        self.assertIn(f"would remove file: {self.apps / DESKTOP}", result.stdout)
        self.assertIn(f"would remove shim: {self.center_shim}", result.stdout)
        self.assertIn("Dry run", result.stdout)
        self.assertTrue((self.prefix / "venv" / "pyvenv.cfg").exists())
        self.assertTrue((self.bin_dir / "mlx-omarchy").exists())
        self.assertTrue(self.center_shim.exists())

    def test_yes_removes_install_and_keeps_data_and_cache(self):
        result = self.run_retire("--yes")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.prefix / "venv").exists())
        for launcher in LAUNCHERS:
            self.assertFalse((self.bin_dir / launcher).exists())
        self.assertFalse((self.unit_dir / UNIT).exists())
        self.assertFalse((self.apps / DESKTOP).exists())
        self.assertFalse(self.center_shim.exists())
        # User data and the model cache survive.
        self.assertTrue((self.prefix / "assistant" / "history.json").exists())
        self.assertTrue((self.cache / "weights.bin").exists())
        self.assertIn("User data kept", result.stdout)
        self.assertIn("~/.cache/huggingface", result.stdout)

    def test_rerun_is_a_noop(self):
        self.run_retire("--yes")
        again = self.run_retire("--yes")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("already retired", again.stdout)

    def test_purge_data_needs_yes_and_then_removes_the_root(self):
        dry = self.run_retire("--purge-data")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertTrue(self.prefix.exists())
        purged = self.run_retire("--yes", "--purge-data")
        self.assertEqual(purged.returncode, 0, purged.stderr)
        self.assertFalse(self.prefix.exists())
        self.assertTrue((self.cache / "weights.bin").exists())

    def test_foreign_command_center_shim_is_left_alone(self):
        foreign = self.center / "omarchy-mlx-serve"
        foreign.write_text("#!/bin/sh\nexec some-other-serve\n")
        foreign.chmod(0o755)
        result = self.run_retire("--yes")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(foreign.exists())
        self.assertIn("not mlx-omarchy's", result.stdout)

    def test_custom_data_home_is_honored(self):
        alt = Path(self._tmp.name) / "alt-root" / "venv"
        alt.mkdir(parents=True)
        result = self.run_retire("--yes", extra_env={"MLX_OMARCHY_HOME": str(alt.parent)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(alt.exists())
        self.assertTrue(self.prefix.exists())  # default root untouched

    def test_unknown_option_fails(self):
        result = self.run_retire("--explode")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown option", result.stderr)

    def test_no_install_is_reported_cleanly(self):
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        clean_path = Path(empty.name) / "bin"
        clean_path.mkdir()
        result = self.run_retire(
            extra_env={
                "HOME": empty.name,
                "MLX_OMARCHY_HOME": str(Path(empty.name) / "none"),
                "PATH": f"{clean_path}:{os.environ['PATH']}",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No legacy", result.stdout)

    def test_script_never_removes_the_system_prefix(self):
        text = RETIRE.read_text()
        # Removal paths are always HOME- or data-root-relative; the system
        # prefix is only ever read for the informational message.
        for line in text.splitlines():
            if "rm -rf" in line or "rm -f" in line:
                self.assertNotIn("SYSTEM_PREFIX", line)


if __name__ == "__main__":
    unittest.main()
