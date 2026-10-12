"""Tests for profile_analyze.py: delayed-record phase attribution.

Stdlib unittest only. The fixtures replicate the real MLX_OMARCHY_GPU_PROFILE
event contract (gpu_profiler.h): {"k":"meta"|"d"|"s"|"j"} NDJSON plus the
markers file written by scripts/profile_generate.py. Dispatch ("k":"d")
records are emitted late - at join or ring-slot reuse - so the fixtures put
them in flushed order, NOT submission order: a correct analyzer must
attribute by the submission id d["s"] against the submit record's host time,
never by file order.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "profile_analyze.py"

MARKERS = [
    {"t": 1000, "p": "load_start"},
    {"t": 2000, "p": "load_done"},
    {"t": 3000, "p": "prefill_start"},
    {"t": 9000, "p": "prefill_done"},
    {"t": 9500, "p": "decode_start"},
    {"t": 12000, "p": "tok"},
    {"t": 20000, "p": "tok"},
    {"t": 25000, "p": "decode_done"},
]

# Flushed order: decode submission 3 first, prefill submission 1 last.
# Host submit times decide windows: s1@3500=prefill, s2@11000=decode,
# s3@15000=decode; s4@2500 falls between load_done and prefill_start
# (submit outside named phases); s99 has no submit record at all. The
# two unknowables must stay distinct.
PROFILE = [
    {"k": "meta", "device": "llvmpipe", "period_ns": 33.333,
     "valid_bits": 64, "pool": 65536, "label": "t", "host_t0": 100},
    {"k": "d", "s": 3, "e": 7, "op": 0, "n": 1, "gx": 1, "gy": 1, "gz": 1,
     "h": 120, "t0": 5200, "t1": 8200, "b": [[1, 0, 256]]},
    {"k": "d", "s": 2, "e": 7, "op": 0, "n": 1, "gx": 1, "gy": 1, "gz": 1,
     "h": 100, "t0": 4200, "t1": 5200, "b": [[1, 0, 256]]},
    {"k": "d", "s": 1, "e": 5, "op": 0, "n": 1, "gx": 32, "gy": 1, "gz": 1,
     "h": 150, "t0": 3500, "t1": 4200, "b": [[1, 0, 256]]},
    {"k": "d", "s": 4, "e": 5, "op": 0, "n": 1, "gx": 1, "gy": 1, "gz": 1,
     "h": 10, "t0": 300, "t1": 400, "b": []},
    {"k": "d", "s": 99, "e": 5, "op": 0, "n": 1, "gx": 1, "gy": 1, "gz": 1,
     "h": 10, "t0": 100, "t1": 200, "b": []},
    {"k": "s", "s": 1, "dur": 800, "t": 3500},
    {"k": "s", "s": 2, "dur": 300, "t": 11000},
    {"k": "s", "s": 3, "dur": 300, "t": 15000},
    {"k": "s", "s": 4, "dur": 50, "t": 2500},
    {"k": "j", "s": 1, "wait": 500, "inval": 100, "t": 8000},
    {"k": "j", "s": 3, "wait": 900, "inval": 100, "t": 21000},
    {"k": "end", "t": 30000, "dispatches": 5, "dropped": 0,
     "submissions": 4, "joins": 2},
]


def write_fixture(dirpath, records, markers=None):
    profile = Path(dirpath) / "profile.jsonl"
    profile.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    markers_path = None
    if markers is not None:
        markers_path = Path(dirpath) / "markers.jsonl"
        markers_path.write_text(
            "\n".join(json.dumps(m) for m in markers) + "\n",
            encoding="utf-8")
    return profile, markers_path


def run_analyzer(profile, markers_path=None):
    cmd = [sys.executable, str(SCRIPT), str(profile)]
    if markers_path:
        cmd += ["--markers", str(markers_path)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


class AttributionTests(unittest.TestCase):
    def test_phases_follow_submission_not_file_order(self):
        with tempfile.TemporaryDirectory() as td:
            profile, markers_path = write_fixture(td, PROFILE, MARKERS)
            r = run_analyzer(profile, markers_path)
        self.assertEqual(r.returncode, 0, r.stderr)
        # Submission 1's dispatch record appears LAST in the file, but its
        # submit time lands in prefill: attribution must say prefill.
        self.assertIn("prefill: dispatches=1", r.stdout)
        self.assertIn("decode: dispatches=2", r.stdout)
        # Two DISTINCT unknowables, never conflated:
        self.assertIn("unknown (no matching submit record): dispatches=1",
                      r.stdout)
        self.assertIn("unknown (submit outside named phases): dispatches=1",
                      r.stdout)
        # Host-marker windows are labeled as windows, not semantic phases.
        self.assertIn("per host-marker window", r.stdout)
        self.assertIn("windows are NOT", r.stdout)

    def test_no_zero_fabrication_for_unknown_dispatch(self):
        with tempfile.TemporaryDirectory() as td:
            profile, markers_path = write_fixture(td, PROFILE, MARKERS)
            r = run_analyzer(profile, markers_path)
        # Both unknown dispatches carry real GPU time (100 ticks each);
        # neither may vanish into a named phase or a zero.
        unknown_rows = [ln for ln in r.stdout.splitlines()
                        if "unknown (" in ln and "dispatches=1" in ln]
        self.assertEqual(len(unknown_rows), 2)
        for ln in unknown_rows:
            self.assertNotIn("gpu_busy=0.000ms", ln)
            self.assertNotIn("gpu_busy=0ns", ln)

    def test_decode_rates_use_interval_denominator(self):
        with tempfile.TemporaryDirectory() as td:
            profile, markers_path = write_fixture(td, PROFILE, MARKERS)
            r = run_analyzer(profile, markers_path)
        # 2 tok markers = 1 inter-token interval; the yield count must
        # not be the denominator (39879/64 vs /63 class of error).
        self.assertIn("dispatches/decode-interval: 2.0 "
                      "(2 over 1 inter-token intervals", r.stdout)
        self.assertIn("submissions/decode-interval: 2.0 (2 over 1 intervals)",
                      r.stdout)
        self.assertNotIn("decode-token", r.stdout)

    def test_dependency_proxy_labeled_as_proxy_only(self):
        with tempfile.TemporaryDirectory() as td:
            profile, markers_path = write_fixture(td, PROFILE, MARKERS)
            r = run_analyzer(profile, markers_path)
        # Disjoint compute bindings say nothing about transfer/fill/alias
        # dependencies or which barriers are removable.
        self.assertNotIn("barrier-free", r.stdout)
        self.assertIn("compute-binding", r.stdout)
        self.assertIn("not a barrier-correctness bound", r.stdout)

    def test_kernel_enum_grouping_without_header(self):
        with tempfile.TemporaryDirectory() as td:
            profile, markers_path = write_fixture(td, PROFILE, MARKERS)
            r = run_analyzer(profile, markers_path)
        # No --compute-h: enum 7 must surface as kernel_7, not crash or
        # group by anything else.
        self.assertIn("kernel_7", r.stdout)

    def test_enum_parse_keeps_enumerators_with_trailing_comments(self):
        # A retired enumerator keeps its id (append-only) and is written
        # `Name, // retired ...`; dropping it shifts every later name by one.
        header = (
            "enum class ComputeKernel : uint16_t {\n"
            "  Alpha,\n"
            "  Retired, // retired: no shader; the id stays\n"
            "  // a comment line\n"
            "  Gamma,\n"
            "  Count,\n"
            "};\n")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "compute.h"
            path.write_text(header, encoding="utf-8")
            spec = importlib.util.spec_from_file_location("profile_analyze", SCRIPT)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            names = module.parse_kernel_names(str(path))
        self.assertEqual(names, ["Alpha", "Retired", "Gamma", "Count"])


class MalformedInputTests(unittest.TestCase):
    def test_malformed_json_line_exits_2_with_line_number(self):
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, PROFILE[:3])
            with open(profile, "a", encoding="utf-8") as fh:
                fh.write("{not json}\n")
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 2)
        self.assertIn("malformed profile line 4", r.stderr)

    def test_missing_meta_exits_1(self):
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, PROFILE[1:])
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no meta line", r.stderr)

    def test_meta_without_period_ns_exits_2(self):
        bad = [{"k": "meta", "device": "llvmpipe", "valid_bits": 64}]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, bad)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 2)
        self.assertIn("period_ns", r.stderr)

    def test_missing_file_fails_nonzero(self):
        r = run_analyzer(Path("/nonexistent/profile.jsonl"))
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
