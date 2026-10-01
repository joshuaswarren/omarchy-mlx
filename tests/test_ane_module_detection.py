from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "overlay/tools/mlx-omarchy-info/ane_module.h"


class AneModuleDetectionTests(unittest.TestCase):
    def test_t6021_and_generic_modules_are_detected_from_fake_sysfs(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler is not installed")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            modules = root / "sys/module"
            t6021 = modules / "ane_t6021"
            t6021.mkdir(parents=True)
            (t6021 / "version").write_text("0.4.0\n")
            source = root / "module_check.cpp"
            source.write_text(
                '''#include <fstream>
#include <iostream>
#include <iterator>
#include "ane_module.h"
int main(int, char** argv) {
  const auto path = omarchy_info::find_ane_module(argv[1]);
  if (path.empty()) return 1;
  std::ifstream version(path / "version");
  std::string value((std::istreambuf_iterator<char>(version)), {});
  if (!value.empty() && value.back() == '\\n') value.pop_back();
  std::cout << path.filename().string() << " " << value << "\\n";
}
'''
            )
            binary = root / "module_check"
            subprocess.run(
                [compiler, "-std=c++17", "-I", str(HEADER.parent),
                 str(source), "-o", str(binary)],
                check=True, capture_output=True, text=True, timeout=30,
            )

            result = subprocess.run(
                [str(binary), str(root)],
                check=False, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "ane_t6021 0.4.0")

            generic = modules / "ane"
            generic.mkdir()
            (generic / "version").write_text("0.2.0\n")
            result = subprocess.run(
                [str(binary), str(root)],
                check=False, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "ane 0.2.0")


if __name__ == "__main__":
    unittest.main()
