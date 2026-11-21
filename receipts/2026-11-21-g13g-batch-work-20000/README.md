# 13-inch M1 (G13G) submission budget: 8000 to 20000

Date: 2026-11-21. Machine: 13-inch M1 (T8103), Linux 7.3.0-2-20261110.1-lab-ARCH, Mesa vulkan-asahi-omarchy 26.2.4.git69bde15340, wheel 0.32.4.dev202611210550+bf5631d (v0.7.49, libmlx sha256 b9d70d47...). All runs went through the lab GPU queue (one job at a time, GPU lock held). Firmware pstate during decode: 6 (top) in every 2 ms sample.

## Question

A decode step evaluates once (1 finalize and 867 dispatches per Qwen3.8-2B token). At the G13G work budget 8000, the evaluator splits that step into 16 submissions. Each submission is a separate kernel job, and it waits for the previous job to complete. Is there a budget that decodes faster and keeps the submission tail inside the rule the G13S, G14G and G17P rows use (p99 at or under about 40 ms, no submission near the 78 ms stutter of issue 19)?

## Verdict

20000. It is the largest swept budget that keeps the Qwen3-4B submission p99 under 40 ms.

| Budget | Qwen3.8-2B decode tok/s (median) | vs 8000 | 2B submits/token | 2B submission p50 / p99 / max | Qwen3-4B decode (median) | vs 8000 | 4B submission p50 / p99 / max |
|---|---|---|---|---|---|---|---|
| 8000 | 27.94 | 0 | 16 | 2.30 / 5.03 / 5.36 ms | 22.46 | 0 | 8.38 / 10.84 / 11.81 ms |
| 20000 | 29.06 | +4.0% | 7 | 5.03 / 8.74 / 9.18 ms | 22.74 | +1.3% | 20.23 / 24.11 / 24.49 ms |
| 40000 | 29.52 | +5.7% | 4 | 9.95 / 13.44 / 13.97 ms | 22.81 | +1.5% | p99 45.53 / max 46.50 ms |
| 60000 | 29.66 | +6.2% | 3 | 14.81 / 18.62 / 18.85 ms | 22.84 | +1.7% | 43.45 / 44.28 / 44.44 ms |
| 80000 | 29.86 | +6.9% | 2 | 15.67 / 20.41 / 21.80 ms | 22.89 | +1.9% | 43.46 / 45.03 / 48.95 ms |
| 0 (cap off) | 29.15 | +4.5% | 1 | 33.25 / 34.24 / 35.76 ms | | | |

- Greedy ids are identical across every arm of each model: Qwen3.8-2B ids sha256 cdf6b652..., 128 tokens, 27 runs; Qwen3-4B ids sha256 0d3c0f29..., 96 tokens, 10 runs.
- 40000 and above put the 4B p99 over 40 ms, so those values fail the tail rule.
- The cap-off row decodes slower than 4 submissions per token. With one submission the GPU waits for the host to record the whole step; with several, recording overlaps execution.
- The 4B p50 at 8000 is already 8.4 ms on this stack, so the old 2 to 6 ms p50 band no longer holds at the current row either. The tail rule is the binding constraint, as for G13S, G14G and G17P.

## Runs

- runs/qwen38-2b-a: 8000 / 40000 / 0, 4 runs each, balanced order (tools/run.sh).
- runs/qwen38-2b-sweep: 8000 / 20000 / 40000 / 60000 / 80000, 3 runs each, balanced order (tools/run2.sh).
- runs/qwen3-4b: Qwen3-4B 4-bit (the calibration model), the same budgets, 2 runs each (tools/run3.sh).
- runs/submission-gpu-time.json: per-submission GPU time (drm_sched job run to done, read-only ftrace of gpu_scheduler events) over 64 decode tokens per setting. The raw traces are not committed because they list every process on the host.

Decode is mlx-lm stream_generate with a fixed prompt of about 600 tokens and greedy sampling (tools/decode_sub.py). Submissions per token come from the backend trace counters (mlx_omarchy_trace_snapshot).
