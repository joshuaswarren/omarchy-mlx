"""Cap-footer handling in the remaining GPU-profile NDJSON readers.

The profiler appends a complete {"k":"cap","bytes":N,"cap":C} record when
it exhausts its byte cap and then stops permanently. Every consumer that
reads these streams must refuse to treat the truncated stream as a
complete profile. Stdlib unittest only; fixtures replicate the real
MLX_OMARCHY_GPU_PROFILE event contract (gpu_profiler.h).
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

META = {"k": "meta", "device": "llvmpipe", "period_ns": 1.0,
        "valid_bits": 64, "pool": 65536, "label": "t", "host_t0": 100}
DISPATCH = {"k": "d", "s": 1, "e": 0, "op": 0, "n": 1, "gx": 1, "gy": 1,
            "gz": 1, "h": 10, "t0": 100, "t1": 200, "b": [[1, 0, 256]]}
SUBMIT = {"k": "s", "s": 1, "dur": 50, "t": 150}
CAP = {"k": "cap", "bytes": 1073741824, "cap": 1073741824}

HEADER = (
    "enum class ComputeKernel : uint16_t {\n"
    "  Alpha,\n"
    "  Count,\n"
    "};\n")


def load_module(relpath, name):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_lines(dirpath, filename, records):
    # The C++ encoder writes compact NDJSON; analyze_w1.load() filters on
    # the literal {"k":"d" prefix, so fixtures must match that byte shape.
    path = Path(dirpath) / filename
    path.write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in records)
        + "\n", encoding="utf-8")
    return path


class AnalyzeW1LoadTests(unittest.TestCase):
    def test_cap_line_raises(self):
        mod = load_module("tools/dfuse/analyze_w1.py", "analyze_w1")
        with tempfile.TemporaryDirectory() as td:
            prof = write_lines(td, "prof-x.ndjson", [META, DISPATCH, CAP])
            with self.assertRaises(SystemExit) as cm:
                mod.load(str(prof))
        self.assertIn("byte cap", str(cm.exception))

    def test_dispatch_only_file_unchanged(self):
        mod = load_module("tools/dfuse/analyze_w1.py", "analyze_w1")
        with tempfile.TemporaryDirectory() as td:
            prof = write_lines(td, "prof-x.ndjson", [DISPATCH])
            self.assertEqual(len(mod.load(str(prof))), 1)


class SubcapCalibrateTests(unittest.TestCase):
    def test_cap_raises_runtime_error(self):
        mod = load_module("scripts/subcap_calibrate.py", "subcap_calibrate")
        with tempfile.TemporaryDirectory() as td:
            prof = write_lines(td, "sc.jsonl", [META, DISPATCH, CAP])
            with self.assertRaises(RuntimeError) as cm:
                mod.ms_per_submit(str(prof), grace_seconds=0.1)
        self.assertIn("byte cap", str(cm.exception))

    def test_uncapped_dispatches_still_summed(self):
        mod = load_module("scripts/subcap_calibrate.py", "subcap_calibrate")
        with tempfile.TemporaryDirectory() as td:
            prof = write_lines(td, "sc.jsonl", [META, DISPATCH])
            vals = mod.ms_per_submit(str(prof), grace_seconds=0.1)
        # (200-100) ticks * 1.0 ns = 100 ns = 0.0001 ms
        self.assertEqual(vals, [0.0001])


class KvStateViewsTests(unittest.TestCase):
    def test_cap_raises_systemexit(self):
        mod = load_module("scripts/kv_state_views.py", "kv_state_views")
        with tempfile.TemporaryDirectory() as td:
            prof = write_lines(td, "p.jsonl", [META, DISPATCH, CAP])
            mark = write_lines(td, "m.jsonl", [{"t": 1, "p": "s1_start"}])
            with self.assertRaises(SystemExit) as cm:
                mod.window_counters(str(prof), str(mark), ["Alpha"])
        self.assertIn("byte cap", str(cm.exception))

    def test_uncapped_windows_unchanged(self):
        mod = load_module("scripts/kv_state_views.py", "kv_state_views")
        with tempfile.TemporaryDirectory() as td:
            prof = write_lines(td, "p.jsonl",
                               [META, SUBMIT, DISPATCH])
            mark = write_lines(td, "m.jsonl", [{"t": 1, "p": "s1_start"}])
            per, total, unresolved = mod.window_counters(
                str(prof), str(mark), ["Alpha"])
        self.assertEqual(total, 1)
        self.assertEqual(unresolved, 0)
        self.assertEqual(per["s1_start"], {"Alpha/n=1": 1})


class CliCapTests(unittest.TestCase):
    """CLI consumers fail nonzero with the cap named on stderr."""

    def run_cli(self, script, *args):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / script), *map(str, args)],
            capture_output=True, text=True, timeout=60)

    def test_chain_census_cap_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            header = Path(td) / "compute.h"
            header.write_text(HEADER, encoding="utf-8")
            prof = write_lines(td, "p.jsonl", [META, DISPATCH, CAP])
            r = self.run_cli("scripts/chain_census.py", prof,
                             "--compute-h", header)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("byte cap", r.stderr)

    def test_sdpa_counts_cap_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            header = Path(td) / "compute.h"
            header.write_text(HEADER, encoding="utf-8")
            prof = write_lines(td, "p.jsonl", [META, DISPATCH, CAP])
            markers = write_lines(td, "m.jsonl", [{"t": 1, "p": "tok"}])
            r = self.run_cli("scripts/sdpa_counts.py", prof,
                             "--compute-h", header, "--markers", markers)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("byte cap", r.stderr)

    def test_analyze_decode_profile_cap_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            header = Path(td) / "compute.h"
            header.write_text(HEADER, encoding="utf-8")
            prof = write_lines(td, "p.jsonl", [META, DISPATCH, CAP])
            windows = write_lines(
                td, "w.jsonl",
                [{"prompt": "x", "start_monotonic": 100}, {"t1": 300}])
            r = self.run_cli("tools/analyze_decode_profile.py",
                             "--profile", prof, "--windows", windows,
                             "--compute-h", header)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("byte cap", r.stderr)
        # No JSON summary may be printed for a capped stream.
        self.assertNotIn('"steady"', r.stdout)


if __name__ == "__main__":
    unittest.main()
