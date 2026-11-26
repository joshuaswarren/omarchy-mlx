import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class DecodeProfileEnumTests(unittest.TestCase):
    def test_cli_maps_commented_explicit_enum_and_next_implicit_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            header = root / "compute.h"
            header.write_text(
                "enum class Other { Wrong, };\n"
                "enum class ComputeKernel : uint16_t {\n"
                "  Before,\n"
                "  Commented, // retained id with {braces}\n"
                "  Explicit = 2, // fixed id\n"
                "  After,\n"
                "  Count,\n"
                "};\n",
                encoding="utf-8",
            )
            windows = root / "windows.jsonl"
            windows.write_text(
                '{"prompt":true,"start_monotonic":0}\n'
                + "".join(f'{{"t1":{i * 10}}}\n' for i in range(1, 10)),
                encoding="utf-8",
            )
            profile = root / "profile.jsonl"
            profile.write_text(
                '{"k":"meta","period_ns":1,"valid_bits":64}\n'
                '{"k":"s","s":1,"t":40}\n'
                '{"k":"d","s":1,"e":3,"n":1,"op":"op","t0":1,"t1":2}\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools/analyze_decode_profile.py"),
                 "--profile", str(profile), "--windows", str(windows),
                 "--compute-h", str(header)],
                check=True, capture_output=True, text=True,
            )
        output = json.loads(result.stdout)
        self.assertEqual(output["top_kernels_by_count"], [["After|n=1|op=op", 1, 0.0]])


if __name__ == "__main__":
    unittest.main()
