# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the mlx-omarchy-parakeet downloader CLI (section 13)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.reference import ReferenceLock, default_cache_root, model_cache_dir

_CLI = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "mlx-omarchy-parakeet"
    / "mlx_omarchy_parakeet.py"
)


class ParakeetCliTest(unittest.TestCase):
    def test_download_on_verifying_cache_emits_true_receipt(self):
        lock = ReferenceLock.load()
        cache_dir = model_cache_dir(
            default_cache_root(), lock.model_repo, lock.model_revision
        )
        if not cache_dir.is_dir():
            self.skipTest("parakeet reference cache not present (offline)")

        run = subprocess.run(
            [sys.executable, str(_CLI), "download", "--json"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        receipt = json.loads(run.stdout)
        self.assertEqual(
            receipt["schema"], "mlx-omarchy.parakeet-download-receipt.v1"
        )
        self.assertEqual(receipt["model_revision"], lock.model_revision)
        self.assertEqual(
            receipt["reference_commit"], lock.reference_commit
        )
        # Every pinned file reports as verified against its lock hash.
        self.assertTrue(receipt["files"])
        for entry in receipt["files"]:
            self.assertTrue(entry["verified"], entry["path"])
        paths = {entry["path"] for entry in receipt["files"]}
        self.assertIn("tokenizer.json", paths)
        for package in ("encoder", "decoder", "joint"):
            self.assertTrue(
                any(p.startswith(f"{package}.mlpackage/") for p in paths)
            )

    def test_verify_reports_ok(self):
        lock = ReferenceLock.load()
        cache_dir = model_cache_dir(
            default_cache_root(), lock.model_repo, lock.model_revision
        )
        if not cache_dir.is_dir():
            self.skipTest("parakeet reference cache not present (offline)")
        run = subprocess.run(
            [sys.executable, str(_CLI), "verify"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("verified", run.stdout)


class ParakeetDependencyProbeTest(unittest.TestCase):
    """The transcribe dependency probe: message, self-heal, import closure."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "mlx_omarchy_parakeet_under_test", _CLI
        )
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_missing_deps_message_names_pip_line_and_interpreter(self):
        message = self.mod._missing_deps_message(["numpy", "google.protobuf"])
        self.assertTrue(message.startswith(
            "missing runtime dependencies for transcribe: "
            "numpy, google.protobuf"
        ))
        self.assertIn(
            f"{sys.executable} -m pip install numpy protobuf", message
        )

    def test_missing_deps_message_points_at_the_system_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "bin" / "python"
            fake.parent.mkdir()
            fake.write_text("#!/bin/sh\n")
            original = self.mod._SYSTEM_VENV_PYTHON
            self.mod._SYSTEM_VENV_PYTHON = fake
            try:
                message = self.mod._missing_deps_message(["numpy"])
            finally:
                self.mod._SYSTEM_VENV_PYTHON = original
        self.assertIn(f"the system package's runtime interpreter is {fake}",
                      message)

    def test_probe_refuses_without_a_reexec_target(self):
        original = self.mod._installing_venv_python
        self.mod._installing_venv_python = lambda: None
        # The dev interpreter usually has numpy; force the missing state
        # this probe exists to report.
        original_missing = self.mod._missing_runtime_deps
        self.mod._missing_runtime_deps = lambda: ["numpy", "google.protobuf"]
        try:
            with self.assertRaises(self.mod.TranscribeRefusal) as ctx:
                self.mod._check_runtime_deps()
        finally:
            self.mod._installing_venv_python = original
            self.mod._missing_runtime_deps = original_missing
        self.assertIn("missing runtime dependencies for transcribe",
                      str(ctx.exception))

    def test_installing_venv_python_none_in_dev_checkout(self):
        # The dev layout (overlay/tools/mlx-omarchy-parakeet/...) has no
        # owning venv five parents up; discovery must say so instead of
        # fabricating a path.
        self.assertIsNone(self.mod._installing_venv_python())

    def test_installing_venv_python_finds_owning_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            site = env / "lib" / "python3.x" / "site-packages" / "mlx" / "bin"
            site.mkdir(parents=True)
            installed = site / "mlx_omarchy_parakeet.py"
            installed.write_text("# sentinel\n")
            stub = env / "bin" / "python"
            stub.parent.mkdir()
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)
            self.assertEqual(self.mod._installing_venv_python(installed), stub)

    def test_installing_venv_python_accepts_symlinked_venv_python(self):
        """A venv python symlinks to the same CPython binary the system
        python3 may use (jw16: /usr/bin/python3 -> python3.14). The
        environments differ even when the binary does not, so discovery
        must compare raw paths and still return the candidate — the
        resolve()-based comparison disabled the re-exec and was the one
        real bug this feature shipped.
        """
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            site = env / "lib" / "python3.x" / "site-packages" / "mlx" / "bin"
            site.mkdir(parents=True)
            installed = site / "mlx_omarchy_parakeet.py"
            installed.write_text("# sentinel\n")
            (env / "bin").mkdir()
            linked = env / "bin" / "python"
            linked.symlink_to(sys.executable)
            self.assertEqual(self.mod._installing_venv_python(installed),
                             linked)

    def test_venv_has_deps_probes_the_candidate_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            yes = Path(tmp) / "yes"
            yes.write_text("#!/bin/sh\nexit 0\n")
            yes.chmod(0o755)
            no = Path(tmp) / "no"
            no.write_text("#!/bin/sh\nexit 1\n")
            no.chmod(0o755)
            self.assertTrue(self.mod._venv_has_deps(yes, ["numpy"]))
            self.assertFalse(self.mod._venv_has_deps(no, ["numpy"]))

    def test_reexec_self_heal_runs_transcribe_help_under_a_bare_interpreter(self):
        """End to end: `transcribe --help` under an interpreter without the
        deps re-execs into the owning venv and exits 0 instead of
        refusing. This is the jwm1 defect, staged in a fake install."""
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            mlx_root = env / "lib" / "python3.x" / "site-packages" / "mlx"
            site = mlx_root / "bin"
            site.mkdir(parents=True)
            # The wheel ships the CLI with the coreml tree beside it; the
            # top-level imports must resolve under the bare interpreter.
            shutil.copytree(_CLI.parents[1] / "coreml", mlx_root / "coreml")
            installed = site / "mlx-omarchy-parakeet"
            installed.write_text(_CLI.read_text())
            wrapper = env / "bin" / "python"
            wrapper.parent.mkdir()
            wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
            wrapper.chmod(0o755)
            saved_guard = os.environ.pop(self.mod._REEXEC_GUARD, None)
            try:
                run = subprocess.run(
                    [sys.executable, "-S", str(installed), "transcribe",
                     "--help"],
                    capture_output=True, text=True, timeout=120,
                )
            finally:
                if saved_guard is not None:
                    os.environ[self.mod._REEXEC_GUARD] = saved_guard
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertNotIn("missing runtime dependencies", run.stderr)

    def test_transcribe_import_closure_is_declared(self):
        """Static import closure of the CLI plus the vendored coreml tree.

        Every third-party root must be one of: the wheel itself (mlx),
        the declared transcribe deps (numpy, google.protobuf), or the
        declared-optional soundfile. Keeps _TRANSCRIBE_DEPS from drifting
        away from what the transcribe path actually imports.
        """
        import ast

        tools = _CLI.parent.parent          # overlay/tools
        coreml = tools / "coreml"
        extra = tools / "ane-export"        # bundle_cache sys.path import

        def local_path(name):
            for base in (tools, coreml, extra):
                candidate = base / f"{name}.py"
                if candidate.is_file():
                    return candidate
                package = base / name / "__init__.py"
                if package.is_file():
                    return package
            return None

        seen: set[Path] = set()
        external: dict[str, set[str]] = {}
        queue = [_CLI, *sorted(coreml.rglob("*.py"))]
        while queue:
            path = queue.pop()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                        and node.module:
                    names = [node.module]
                for name in names:
                    resolved = local_path(name) or local_path(name.split(".")[0])
                    if resolved:
                        queue.append(resolved)
                    else:
                        root = name.split(".")[0]
                        if root not in sys.stdlib_module_names:
                            external.setdefault(root, set()).add(path.name)

        # Exact set: the wheel itself, the declared transcribe deps, and
        # the declared-optional soundfile decoder. Anything else is an
        # undeclared import this contract test refuses.
        self.assertEqual(
            set(external),
            {"mlx", "soundfile"}
            | {module.split(".")[0] for module in self.mod._TRANSCRIBE_DEPS},
            f"import closure drifted: {sorted(external)}",
        )
        # The declared deps are not decorative: the transcribe path must
        # really import both.
        self.assertIn("numpy", external)
        self.assertIn("google", external)


if __name__ == "__main__":
    unittest.main()
