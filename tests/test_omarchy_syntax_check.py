import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/ci/omarchy_syntax_check.py"


@unittest.skipUnless(shutil.which("c++"), "requires a C++ compiler")
class OmarchySyntaxCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="syntax gate ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = self.root / "mlx/backend/omarchy"
        self.backend.mkdir(parents=True)

    def entry(self, name, source, *, arguments=False):
        path = self.backend / name
        path.write_text(source)
        command = [
            shutil.which("c++"), "-DVALUE=7", "-c", str(path),
            "-o", str(self.root / "output file.o"),
            "-MD", "-MT", "target name", "-MF", str(self.root / "output file.d"),
        ]
        return {
            "directory": str(self.root), "file": str(path),
            **({"arguments": command} if arguments else {"command": shlex.join(command)}),
        }

    def run_checker(self, entries):
        (self.root / "compile_commands.json").write_text(json.dumps(entries))
        return subprocess.run(
            [sys.executable, str(CHECKER), str(self.root), "--jobs", "2"],
            capture_output=True, text=True, timeout=20, check=False,
        )

    def test_quoted_command_checks_source_without_build_artifacts(self):
        result = self.run_checker([self.entry("valid file.cpp", "static_assert(VALUE == 7);\n")])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1/1 translation units pass", result.stdout)
        self.assertFalse((self.root / "output file.o").exists())
        self.assertFalse((self.root / "output file.d").exists())

    def test_argument_array_checks_source_without_build_artifacts(self):
        result = self.run_checker([
            self.entry("valid.cpp", "static_assert(VALUE == 7);\n", arguments=True),
        ])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1/1 translation units pass", result.stdout)
        self.assertFalse((self.root / "output file.o").exists())
        self.assertFalse((self.root / "output file.d").exists())

    def test_reports_compiler_failure_and_ignores_other_backends(self):
        valid = self.entry("valid.cpp", "static_assert(VALUE == 7);\n")
        invalid = self.entry("invalid.cpp", "static_assert(VALUE == 8);\n")
        unrelated = {**invalid, "file": str(self.root / "mlx/backend/cpu/invalid.cpp")}
        result = self.run_checker([valid, invalid, unrelated])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("FAIL invalid.cpp", result.stdout)
        self.assertIn("1/2 translation units pass", result.stdout)

    def test_empty_backend_selection_fails_closed(self):
        result = self.run_checker([])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no omarchy translation units", result.stderr)

    def test_missing_database_requests_configuration(self):
        result = subprocess.run(
            [sys.executable, str(CHECKER), str(self.root)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("configure first", result.stderr)


if __name__ == "__main__":
    unittest.main()
