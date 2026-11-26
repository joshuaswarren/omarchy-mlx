# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Repeated-inference driver and stage-attribution skeleton (44-45).

Drives N bounded worker runs against one bundle and produces one
per-run receipt plus an aggregate report: median/p95 latency, first-
divergence tracking (every run's outputs are compared against the first
run's), and the per-run trace-counter deltas that attribute work to the
ANE path. Host tests use the mock device seam; the hardware path plugs
the same driver into the real worker.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .encoder_parity import ANE_COUNTER_FIELDS, evaluate_invariants


@dataclass(frozen=True)
class InferenceRunResult:
    iteration: int
    elapsed_ms: int
    status: str
    outputs_match_baseline: bool
    counter_deltas: dict[str, int]
    violations: tuple[str, ...]


@dataclass(frozen=True)
class InferenceReport:
    runs: tuple[InferenceRunResult, ...]
    median_ms: int
    p95_ms: int
    all_matched_baseline: bool
    aggregate_counters: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "schema": "mlx-omarchy.coreml.inference-report.v1",
            "runs": [
                {
                    "iteration": run.iteration,
                    "elapsed_ms": run.elapsed_ms,
                    "status": run.status,
                    "outputs_match_baseline": run.outputs_match_baseline,
                    "counter_deltas": run.counter_deltas,
                    "violations": list(run.violations),
                }
                for run in self.runs
            ],
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "all_matched_baseline": self.all_matched_baseline,
            "aggregate_counters": self.aggregate_counters,
        }


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1, int(len(ordered) * fraction + 0.5)
    )
    return ordered[index]


def run_inference_loop(
    run_once: Callable[[int], tuple[str, dict]],
    snapshot_counters: Callable[[], dict[str, int]],
    *,
    iterations: int = 100,
) -> InferenceReport:
    """Drive ``run_once`` N times and attribute each run.

    ``run_once(iteration)`` returns ``(status, outputs)`` — the worker
    status string and a dict of output-tensor name → payload bytes.
    ``snapshot_counters()`` returns the current trace-counter values;
    per-run deltas are the difference between snapshots. Run 0's
    outputs are the baseline; every later run must match them exactly.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    results: list[InferenceRunResult] = []
    baseline: dict | None = None
    before = snapshot_counters()

    for iteration in range(iterations):
        started = time.monotonic()
        status, outputs = run_once(iteration)
        elapsed = int((time.monotonic() - started) * 1000)

        after = snapshot_counters()
        deltas = {
            key: after.get(key, 0) - before.get(key, 0)
            for key in ANE_COUNTER_FIELDS
        }
        before = after

        if baseline is None:
            baseline = outputs
            matched = True
        else:
            matched = outputs == baseline

        report = {**deltas}
        violations = tuple(evaluate_invariants(report))
        results.append(
            InferenceRunResult(
                iteration=iteration,
                elapsed_ms=elapsed,
                status=status,
                outputs_match_baseline=matched,
                counter_deltas=deltas,
                violations=violations,
            )
        )

    latencies = [run.elapsed_ms for run in results]
    aggregate = {
        key: sum(run.counter_deltas.get(key, 0) for run in results)
        for key in ANE_COUNTER_FIELDS
    }
    return InferenceReport(
        runs=tuple(results),
        median_ms=_percentile(latencies, 0.5),
        p95_ms=_percentile(latencies, 0.95),
        all_matched_baseline=all(
            run.outputs_match_baseline for run in results
        ),
        aggregate_counters=aggregate,
    )
