"""Check the production copy shader's value expression on CPU, not GPU execution."""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / "overlay/mlx/backend/omarchy/shaders/copy_general.comp"


class CopyBf16Bits(unittest.TestCase):
    def test_every_storage_pattern_survives_same_dtype_copy(self):
        source = "\n".join(
            line for line in SHADER.read_text().splitlines()
            if not line.startswith(("#version", "#extension"))
        )
        preprocessed = subprocess.run(
            ["c++", "-E", "-P", "-x", "c++", "-DUSE_BF16=1", "-"],
            input=source, text=True, capture_output=True, check=True,
        ).stdout
        expression = re.search(
            r"output_data\.values\[out_index\]\s*=\s*(.*?);",
            preprocessed, re.DOTALL,
        )
        assert expression is not None, "production copy assignment not found"
        expression = expression.group(1).replace("input_a.values[in_index]", "value")
        helper = re.search(
            r"uint16_t bf16_store\(float value\)\s*\{.*?\n\}",
            preprocessed, re.DOTALL,
        )
        program = """
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
using uint = uint32_t;
using std::isnan;
float uintBitsToFloat(uint value) { return std::bit_cast<float>(value); }
uint floatBitsToUint(float value) { return std::bit_cast<uint>(value); }
""" + (helper.group(0) if helper else "") + "\nuint16_t copy_value(uint16_t value) { return " + expression + "; }\n" + r"""
int main() {
  uint mismatches = 0;
  for (uint bits = 0; bits < 65536; ++bits) {
    if (copy_value(uint16_t(bits)) != bits) ++mismatches;
  }
  std::cout << "patterns=65536 mismatches=" << mismatches << '\n';
  return mismatches != 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / "copy.cpp"
            binary = Path(directory) / "copy-check"
            cpp.write_text(program)
            subprocess.run(
                ["c++", "-std=c++20", "-O0", str(cpp), "-o", str(binary)],
                capture_output=True, text=True, check=True,
            )
            result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
