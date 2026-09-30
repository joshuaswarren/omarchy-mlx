"""The one name table and venv discovery contract.

serve/mlx_omarchy_paths.py is the single source for every install name;
packaging/paths.sh is generated output, and the wheel-side discovery
fallback literals in overlay/tools/coreml/parakeet_dictation.py must
never drift from the module. Discovery precedence is
$OMARCHY_MLX_VENV > /usr/lib/omarchy-mlx/venv > the legacy home venv,
with the one-line retire hint whenever the legacy venv wins.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "serve"
if str(SERVE) not in sys.path:
    sys.path.insert(0, str(SERVE))

import mlx_omarchy_paths as paths  # noqa: E402


def _makedirs(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class NameTableContractTest(unittest.TestCase):
    def test_shell_projection_matches_generated_file(self):
        generated = subprocess.run(
            [sys.executable, str(SERVE / "mlx_omarchy_paths.py"), "--shell"],
            check=True, capture_output=True, text=True,
        ).stdout
        self.assertEqual(generated, (REPO / "packaging" / "paths.sh").read_text())

    def test_shell_table_sources_cleanly(self):
        result = subprocess.run(
            ["bash", "-c", 'source packaging/paths.sh && printf "%s" "$RETIRE_CMD"'],
            cwd=REPO, check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout, paths.RETIRE_CMD)

    def test_wheel_fallback_literals_match_module(self):
        text = (REPO / "overlay" / "tools" / "coreml" / "parakeet_dictation.py").read_text()
        self.assertIn(paths.VENV_ENV, text)
        # The fallback spells the legacy root as path components.
        self.assertIn('".local" / "share" / "mlx-omarchy"', text)
        retire = (REPO / "packaging" / "mlx-omarchy-retire-legacy").read_text()
        self.assertIn(paths.SYSTEM_SHARE_PREFIX, retire)


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self._env = {
            key: os.environ.get(key) for key in (paths.VENV_ENV, paths.DATA_HOME_ENV)
        }
        for key in (paths.VENV_ENV, paths.DATA_HOME_ENV):
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _layout(self, tmp: Path) -> dict:
        home = tmp / "home"
        return {
            "home": home,
            "env": _makedirs(tmp / "env-venv"),
            "system": _makedirs(_makedirs(tmp / "sys-prefix") / paths.VENV_DIR_NAME),
            "legacy": _makedirs(
                _makedirs(home / ".local" / "share" / paths.HOME_PREFIX_NAME)
                / paths.VENV_DIR_NAME
            ),
        }

    def test_precedence_env_system_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp))
            os.environ[paths.VENV_ENV] = str(layout["env"])
            self.assertEqual(
                paths.venv_roots(system_prefix=layout["system"].parent, home=layout["home"]),
                [layout["env"], layout["system"], layout["legacy"]],
            )

    def test_missing_roots_are_filtered_and_legacy_is_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp))
            layout["env"].rmdir()
            found, origin = paths.discover_venv(
                system_prefix=layout["system"].parent, home=layout["home"]
            )
            self.assertEqual((found, origin), (layout["system"], "system"))

    def test_legacy_resolution_prints_retire_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp))
            layout["system"].rmdir()
            capture = io.StringIO()
            with contextlib.redirect_stderr(capture):
                found, origin = paths.discover_venv(
                    system_prefix=layout["system"].parent, home=layout["home"]
                )
            self.assertEqual((found, origin), (layout["legacy"], "legacy"))
            self.assertIn(paths.RETIRE_CMD, capture.getvalue())

    def test_no_venv_resolves_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._layout(Path(tmp))
            for key in ("env", "system", "legacy"):
                layout[key].rmdir()
            self.assertEqual(
                paths.discover_venv(system_prefix=layout["system"].parent, home=layout["home"]),
                (None, "none"),
            )
            self.assertEqual(paths.venv_roots(system_prefix=layout["system"].parent, home=layout["home"]), [])

    def test_data_home_env_wins_for_default_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom = Path(tmp) / "custom-root"
            os.environ[paths.DATA_HOME_ENV] = str(custom)
            self.assertEqual(paths.default_data_home(), custom)
            self.assertEqual(paths.legacy_venv(), custom / paths.VENV_DIR_NAME)

    def test_data_home_default_is_legacy_prefix(self):
        self.assertEqual(
            paths.default_data_home(),
            Path.home() / ".local" / "share" / paths.HOME_PREFIX_NAME,
        )


class RecognitionDelegationTest(unittest.TestCase):
    def test_installed_roots_glob_through_the_name_module(self):
        from mlx_omarchy_assistant import recognition

        with tempfile.TemporaryDirectory() as tmp:
            root = (
                Path(tmp)
                / "venv"
                / "lib"
                / "python3.14"
                / "site-packages"
                / "mlx"
            )
            _makedirs(root)
            previous = os.environ.get(paths.VENV_ENV)
            os.environ[paths.VENV_ENV] = str(Path(tmp) / "venv")
            try:
                self.assertIn(root, recognition._installed_roots())
            finally:
                if previous is None:
                    os.environ.pop(paths.VENV_ENV, None)
                else:
                    os.environ[paths.VENV_ENV] = previous


if __name__ == "__main__":
    unittest.main()
