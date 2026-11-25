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
        self.assertEqual(output["timing_mode"], "t1-t0")

    def test_apple_honeykrisp_uses_dt1_within_submit_and_resets_across_submits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            header = root / "compute.h"
            header.write_text(
                "enum class ComputeKernel : uint16_t {\n"
                "  First,\n"
                "  Heavy,\n"
                "  Count,\n"
                "};\n",
                encoding="utf-8",
            )
            windows = root / "windows.jsonl"
            windows.write_text(
                '{"prompt":true,"start_monotonic":0}\n'
                + "".join(f'{{"t1":{i * 10000000}}}\n' for i in range(1, 10)),
                encoding="utf-8",
            )
            profile = root / "profile.jsonl"
            # On Honeykrisp, Heavy's real GPU time (2.0 ms = 2_000_000 ns) sits in
            # t0[1] - t1[0]; t1[1] - t0[1] is only 20 us. Next submit resets prev_t1.
            profile.write_text(
                '{"k":"meta","device":"Apple M1 Max (G13C C0)","period_ns":1.0,"valid_bits":64}\n'
                '{"k":"s","s":1,"t":45000000}\n'
                '{"k":"d","s":1,"e":0,"n":1,"op":0,"t0":1000000,"t1":1010000}\n'
                '{"k":"d","s":1,"e":1,"n":1,"op":4,"t0":3010000,"t1":3030000}\n'
                '{"k":"s","s":2,"t":55000000}\n'
                '{"k":"d","s":2,"e":0,"n":1,"op":0,"t0":9000000,"t1":9020000}\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools/analyze_decode_profile.py"),
                 "--profile", str(profile), "--windows", str(windows),
                 "--compute-h", str(header)],
                check=True, capture_output=True, text=True,
            )
        output = json.loads(result.stdout)
        self.assertEqual(output["timing_mode"], "dt1")
        by_key = {k: ms for k, ms, _ in output["top_kernels_by_gpu_ms"]}
        self.assertAlmostEqual(by_key["Heavy|n=1|op=4"], 2.02, places=3)
        # Submit 1 First = 0.01 ms, Submit 2 First = 0.02 ms (reset across submit boundary)
        self.assertAlmostEqual(by_key["First|n=1|op=0"], 0.03, places=3)

    def test_dt1_counter_wrap_and_untimed_dispatch_break(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            header = root / "compute.h"
            header.write_text(
                "enum class ComputeKernel : uint16_t {\n"
                "  First,\n"
                "  Wrapped,\n"
                "  AfterGap,\n"
                "  Count,\n"
                "};\n",
                encoding="utf-8",
            )
            windows = root / "windows.jsonl"
            windows.write_text(
                '{"prompt":true,"start_monotonic":0}\n'
                + "".join(f'{{"t1":{i * 10000000}}}\n' for i in range(1, 10)),
                encoding="utf-8",
            )
            # Submit 1: 8-bit counter wraps (240,245) -> (4,8) at 1 us/tick:
            # First = 5 us, Wrapped = (8 - 245) mod 256 = 19 us -> total 24 us.
            # Submit 2: untimed dispatch between (100,105) and (125,130) breaks
            # the chain -> AfterGap gets t1-t0 = 5 us (not 25 us).
            profile = root / "profile.jsonl"
            profile.write_text(
                '{"k":"meta","device":"Apple M1 Max (G13C C0)","period_ns":1000.0,"valid_bits":8}\n'
                '{"k":"s","s":1,"t":45000000}\n'
                '{"k":"d","s":1,"e":0,"n":1,"op":0,"t0":240,"t1":245}\n'
                '{"k":"d","s":1,"e":1,"n":1,"op":0,"t0":4,"t1":8}\n'
                '{"k":"s","s":2,"t":55000000}\n'
                '{"k":"d","s":2,"e":0,"n":1,"op":0,"t0":100,"t1":105}\n'
                '{"k":"d","s":2,"e":0,"n":1,"op":0}\n'
                '{"k":"d","s":2,"e":2,"n":1,"op":0,"t0":125,"t1":130}\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools/analyze_decode_profile.py"),
                 "--profile", str(profile), "--windows", str(windows),
                 "--compute-h", str(header)],
                check=True, capture_output=True, text=True,
            )
        output = json.loads(result.stdout)
        by_key = {k: ms for k, ms, _ in output["top_kernels_by_gpu_ms"]}
        self.assertAlmostEqual(by_key["Wrapped|n=1|op=0"], 0.019, places=6)
        self.assertAlmostEqual(by_key["AfterGap|n=1|op=0"], 0.005, places=6)
        self.assertAlmostEqual(output["steady"]["gpu_ms_total"], 0.034, places=6)


if __name__ == "__main__":
    unittest.main()
