# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the repeated-inference driver (sections 44-45)."""

import unittest

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.inference_driver import run_inference_loop


class _FakeRun:
    """Scripted inference: deterministic outputs, counter ticks."""

    def __init__(self, *, diverge_at=None, fail_at=None):
        self.calls = 0
        self.diverge_at = diverge_at
        self.fail_at = fail_at
        self._counters = {field: 0 for field in (
            "ane_models_loaded", "ane_packages_compiled",
            "ane_package_cache_hits", "ane_worker_starts",
            "ane_submissions", "ane_timeouts", "ane_input_bytes",
            "ane_output_bytes", "ane_exec_ns",
        )}

    def run_once(self, iteration: int):
        self.calls += 1
        self._counters["ane_worker_starts"] += 1
        self._counters["ane_submissions"] += 2
        self._counters["ane_output_bytes"] += 128
        self._counters["ane_exec_ns"] += 1000
        if self.fail_at is not None and iteration == self.fail_at:
            return "DeviceFailed", {}
        outputs = {"y": b"\x42" * 64}
        if self.diverge_at is not None and iteration >= self.diverge_at:
            outputs["y"] = bytes([0xFF]) * 64
        return "Completed", dict(outputs)

    def snapshot(self):
        return dict(self._counters)


class InferenceDriverTest(unittest.TestCase):
    def test_clean_loop_reports_all_matching_with_counter_totals(self):
        fake = _FakeRun()
        report = run_inference_loop(
            fake.run_once, fake.snapshot, iterations=10
        )
        self.assertEqual(fake.calls, 10)
        self.assertTrue(report.all_matched_baseline)
        self.assertEqual(report.median_ms, report.runs[0].elapsed_ms)
        self.assertEqual(
            report.aggregate_counters["ane_worker_starts"], 10
        )
        self.assertEqual(report.aggregate_counters["ane_submissions"], 20)
        self.assertEqual(report.aggregate_counters["ane_output_bytes"], 1280)
        self.assertTrue(all(not run.violations for run in report.runs))
        document = report.to_dict()
        self.assertEqual(
            document["schema"],
            "mlx-omarchy.coreml.inference-report.v1",
        )
        self.assertEqual(len(document["runs"]), 10)

    def test_divergence_is_flagged_per_run(self):
        fake = _FakeRun(diverge_at=5)
        report = run_inference_loop(
            fake.run_once, fake.snapshot, iterations=8
        )
        self.assertFalse(report.all_matched_baseline)
        for run in report.runs[:5]:
            self.assertTrue(run.outputs_match_baseline)
        for run in report.runs[5:]:
            self.assertFalse(run.outputs_match_baseline)

    def test_timeout_counter_is_an_invariant_violation(self):
        fake = _FakeRun()
        original = fake.run_once

        def failing(iteration):
            if iteration == 3:
                fake._counters["ane_timeouts"] += 1
                return "DeadlineExceeded", {}
            return original(iteration)

        report = run_inference_loop(
            failing, fake.snapshot, iterations=5
        )
        violating = [r for r in report.runs if r.violations]
        self.assertEqual(len(violating), 1)
        self.assertIn("ane_timeouts", violating[0].violations[0])

    def test_positive_iteration_count_required(self):
        with self.assertRaises(ValueError):
            run_inference_loop(
                lambda i: ("Completed", {}), lambda: {}, iterations=0
            )


if __name__ == "__main__":
    unittest.main()
