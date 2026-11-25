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

    def test_dispatches_on_empty_submission_exit_2(self):
        # Real M2 capture shape: submission 6 closed the buffer, then an
        # empty submit 7 (close=0) took the slot and the flush tagged the
        # buffer's dispatch with 7.
        bad = PROFILE[:1] + [
            {"k": "d", "s": 7, "e": 5, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 300, "t1": 400, "b": []},
            {"k": "q", "s": 6, "close": 900, "queue_t0": 910,
             "queue_t1": 950},
            {"k": "s", "s": 6, "dur": 50, "t": 960},
            {"k": "q", "s": 7, "close": 0, "queue_t0": 980,
             "queue_t1": 990},
            {"k": "s", "s": 7, "dur": 20, "t": 995},
        ]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, bad)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 2)
        self.assertIn("submission 7 closed no command buffer", r.stderr)

    def test_empty_submission_without_dispatches_is_valid(self):
        ok = PROFILE + [
            {"k": "q", "s": 3, "close": 14000, "queue_t0": 14100,
             "queue_t1": 14900},
            {"k": "q", "s": 5, "close": 0, "queue_t0": 16000,
             "queue_t1": 16100},
            {"k": "s", "s": 5, "dur": 20, "t": 16200},
        ]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, ok)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 0, r.stderr)


class CalibrationRecordTests(unittest.TestCase):
    def test_cal_floor_reported(self):
        rec = PROFILE + [
            {"k": "cal", "floor_ns": 152000, "samples": 32, "warmup": 2},
        ]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, rec)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("calibrated floor=152000ns", r.stdout)
        self.assertIn("32 lone-submit samples", r.stdout)

    def test_cal_gpu_epoch_tolerated_when_present(self):
        rec = PROFILE + [
            {"k": "cal", "floor_ns": 90000, "samples": 32,
             "gpu_epoch": [1234567, 7654321]},
        ]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, rec)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("gpu_epoch=[1234567, 7654321]", r.stdout)

    def test_cal_missing_floor_exits_2(self):
        rec = PROFILE + [{"k": "cal", "samples": 32}]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, rec)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 2)
        self.assertIn("malformed cal record", r.stderr)

    def test_cal_non_numeric_samples_exits_2(self):
        rec = PROFILE + [
            {"k": "cal", "floor_ns": 90000, "samples": "thirty-two"},
        ]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, rec)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 2)
        self.assertIn("malformed cal record", r.stderr)


class HoneykrispDt1Tests(unittest.TestCase):
    def test_apple_device_auto_uses_dt1_within_submit(self):
        rec = [
            {"k": "meta", "device": "Apple M1 Max (G13C C0)",
             "period_ns": 1.0, "valid_bits": 64, "pool": 65536,
             "label": "hk", "host_t0": 0},
            {"k": "d", "s": 1, "e": 0, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 1000000, "t1": 1010000, "b": []},
            {"k": "d", "s": 1, "e": 1, "op": 4, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 3010000, "t1": 3030000, "b": []},
            {"k": "d", "s": 2, "e": 0, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 9000000, "t1": 9020000, "b": []},
            {"k": "s", "s": 1, "dur": 100, "t": 1000},
            {"k": "s", "s": 2, "dur": 100, "t": 2000},
            {"k": "end", "t": 3000, "dispatches": 3, "dropped": 0,
             "submissions": 2, "joins": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, rec)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("timing=dt1", r.stdout)
        # Heavy kernel_1 gets dt1 = 3030000 - 1010000 = 2.020ms, not 20us.
        self.assertIn("kernel_1                   n=1      total=2.020ms", r.stdout)
        # First kernel_0 gets 10us (submit 1) + 20us (submit 2 reset) = 30.0us.
        self.assertIn("kernel_0                   n=2      total=30.0us", r.stdout)
        # Intra-submit raw interval is labeled as included in dt1 busy, not double-counted as gap.
        self.assertIn("raw intra-submission timestamp intervals (included in dt1 busy):", r.stdout)
        self.assertNotIn("gaps intra-submission:", r.stdout)

    def test_dt1_counter_wrap_and_untimed_dispatch_break(self):
        # 8-bit wrap: (240,245) -> (4,8) at 1 us/tick => 5 us + 19 us = 24 us.
        wrap_rec = [
            {"k": "meta", "device": "Apple M1 Max (G13C C0)",
             "period_ns": 1000.0, "valid_bits": 8, "pool": 65536,
             "label": "wrap", "host_t0": 0},
            {"k": "d", "s": 1, "e": 0, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 240, "t1": 245, "b": []},
            {"k": "d", "s": 1, "e": 1, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 4, "t1": 8, "b": []},
            {"k": "s", "s": 1, "dur": 100, "t": 1000},
            {"k": "end", "t": 2000, "dispatches": 2, "dropped": 0,
             "submissions": 1, "joins": 0},
        ]
        # Untimed dispatch between (100,105) and (125,130) breaks dt1 chain:
        # kernel_2 gets t1-t0 = 5 us, not 25 us.
        gap_rec = [
            {"k": "meta", "device": "Apple M1 Max (G13C C0)",
             "period_ns": 1000.0, "valid_bits": 64, "pool": 65536,
             "label": "gap", "host_t0": 0},
            {"k": "d", "s": 1, "e": 0, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 100, "t1": 105, "b": []},
            {"k": "d", "s": 1, "e": 1, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "b": []},
            {"k": "d", "s": 1, "e": 2, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 125, "t1": 130, "b": []},
            {"k": "s", "s": 1, "dur": 100, "t": 1000},
            {"k": "end", "t": 2000, "dispatches": 3, "dropped": 0,
             "submissions": 1, "joins": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            p1, _ = write_fixture(td, wrap_rec)
            r1 = run_analyzer(p1)
            p2, _ = write_fixture(td, gap_rec)
            r2 = run_analyzer(p2)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertIn("busy 24.0us / span 24.0us", r1.stdout)
        self.assertIn("kernel_1                   n=1      total=19.0us", r1.stdout)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("kernel_2                   n=1      total=5.0us", r2.stdout)
        self.assertIn("gaps intra-submission: n=1 total=20.0us", r2.stdout)

    def test_split_submit_counter_wrap_preserves_inter_submit_gap(self):
        # 8-bit wrap split across s=1 (240,245) then s=2 (4,8) at 1 us/tick:
        # s1=5us, s2=4us, inter-submit gap = (260 - 245) = +15.0us (not -24us).
        rec = [
            {"k": "meta", "device": "Apple M1 Max (G13C C0)",
             "period_ns": 1000.0, "valid_bits": 8, "pool": 65536,
             "label": "split-wrap", "host_t0": 0},
            {"k": "d", "s": 1, "e": 0, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 240, "t1": 245, "b": []},
            {"k": "d", "s": 2, "e": 1, "op": 0, "n": 1, "gx": 1, "gy": 1,
             "gz": 1, "h": 10, "t0": 4, "t1": 8, "b": []},
            {"k": "s", "s": 1, "dur": 100, "t": 1000},
            {"k": "s", "s": 2, "dur": 100, "t": 2000},
            {"k": "end", "t": 3000, "dispatches": 2, "dropped": 0,
             "submissions": 2, "joins": 0},
        ]
        with tempfile.TemporaryDirectory() as td:
            profile, _ = write_fixture(td, rec)
            r = run_analyzer(profile)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("busy 9.0us / span 24.0us", r.stdout)
        self.assertIn("gaps inter-submission: n=1 total=15.0us", r.stdout)


if __name__ == "__main__":
    unittest.main()
