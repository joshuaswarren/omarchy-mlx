import resource
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "tests/qmm_drain_check.cpp"
HEADER = ROOT / "overlay/mlx/backend/omarchy/shaders/qmm_drain.h"


class QmmDrainTests(unittest.TestCase):
    def run_check(self, header):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler is not installed")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / CHECK.relative_to(ROOT)
            source.parent.mkdir(parents=True)
            source.write_text(CHECK.read_text())
            include = root / HEADER.relative_to(ROOT)
            include.parent.mkdir(parents=True)
            include.write_text(header)
            binary = root / "drain_check"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-UNDEBUG", str(source), "-o", str(binary)],
                check=True, capture_output=True, text=True, timeout=30,
            )
            return subprocess.run(
                [str(binary)], check=False, capture_output=True, text=True, timeout=10,
                preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)),
            )

    def test_row_major_layouts(self):
        result = self.run_check(HEADER.read_text())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("216 drain layouts", result.stdout)

    def test_transposed_drain_is_rejected(self):
        header = HEADER.read_text()
        mutant = header.replace("row * 8u + column", "column * 8u + row")
        self.assertNotEqual(mutant, header)
        result = self.run_check(mutant)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("output[i] == i + 1u", result.stderr)


if __name__ == "__main__":
    unittest.main()
