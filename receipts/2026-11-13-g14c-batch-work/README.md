# G14C submission budget 40000 -> 100000 (2026-11-13; swept 120000, re-picked, see REPICK.md)

Board: 14-inch M2 Max (G14C), Linux, Mesa honeykrisp, `MLX_OMARCHY_BATCH_WORK` unset (table) vs `120000` vs `200000` (the shipped value is 100000, see REPICK.md).
600-token decode, runs interleaved, 20 s gaps, a 60 Hz submit-latency probe beside each run.
First 64 generated token ids identical across arms in every pair (hash `ids_sha`); logprobs were not recorded in these runs. Scheduling only.

Shipped value 100000 vs default 40000 (REPICK.md, 10 pairs on 4B, 9 on 2B, 3 on 9B in total): small but consistent gains, 4B about +0.5%, 2B about +1.3%, 9B about +1.4%, every rerun pair positive. The first 2B run read +0.3% with one negative pair; the 6-pair rerun read +1.27%, 6 of 6 positive.

## Speed (tok/s; first-campaign rows are means, rerun rows list each run)

| Model | default (40000) | 120000 | gain | pairs positive | script |
|---|---|---|---|---|---|
| 4B | 67.97 | 68.97 | +1.5% | all (6 procs) | `w9c-linux-batchwork24.sh` |
| 4B rerun | 68.75 / 68.72 / 68.77 | 69.65 / 69.56 / 69.55 | +1.2% | 3 of 3 | `w9c-linux-batchwork4b.sh` |
| 2B | 105.99 | 106.84 | +0.8% | all | `w9c-linux-batchwork24.sh` |
| 2B rerun | 108.24 / 107.84 / 105.90 / 105.56 | 109.61 / 109.36 / 106.31 / 106.75 | +1.05% mean | 4 of 4 | `w9c-linux-batchwork2b.sh` |
| 9B | 37.95 | 38.59 | +1.7% | all (6 procs) | `w9c-linux-batchwork9.sh` |
| 9B rerun | 38.08 / 38.05 / 37.69 | 38.69 / 38.33 / 38.25 | +1.3% mean | 3 of 3 | `w9c-linux-batchwork9b3.sh` |

200000 was slower than 120000 on 4B (+0.8% vs +1.5%) and 2B (+0.4% vs +0.8%) and slightly faster on 9B (+2.1% vs +1.7%). 120000 is the best value on two of three models and within 0.4% on the third.
Submits per step: 4B 3.19 -> 1.07, 2B 2.11 -> 1.04, 9B 5.25 -> 2.09.

## Frames (same probe and criterion as #419)

Probe: `scripts/submit-latency.c`, 60 Hz, 256-byte fill.

| Arm | Frames over 33.3 ms | p99 | max |
|---|---|---|---|
| default, 120000, 200000, all models (first campaign) | 0 of 1436 (2B), 0 of 2154 (4B, 9B) | 0.7-1.5 ms | 0.8-2.0 ms |
| 4B rerun, both arms | 0 of 2154 | 0.66-0.81 ms | up to 2.07 ms |
| 2B rerun, both arms | 0 of 2872 | up to 0.83 ms | up to 1.26 ms |
| 9B rerun, both arms | 0 of 2154 | up to 1.19 ms | up to 1.98 ms |

## Per-submission time (the omarchy-mlx #407 rule: p50 <= 6 ms, p99 <= about 40 ms, no sample at 78 ms)

`scripts/subcap_calibrate.py` on a diagnostics wheel of this PR (GPU profiling harness on, so decode runs slower than normal), Qwen3-4B, 64 decode tokens after a 512-token prefill, one process per budget. Each sample is the summed GPU dispatch time of one submission (`runs/subcap/`):

| Budget | n | p50 ms | p95 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| 40000 | 287 | 1.52 | 1.71 | 3.12 | 3.39 |
| 50000 | 285 | 1.83 | 2.07 | 3.37 | 4.10 |
| 60000 | 215 | 2.33 | 2.53 | 4.53 | 5.14 |
| 80000 | 144 | 2.88 | 3.34 | 5.29 | 6.68 |
| 120000 | 142 | 4.54 | 4.86 | 5.93 | 10.06 |

120000 passes the rule's own metric, the summed GPU dispatch time per submission, but it fails the queue-interval p50 below (see REPICK.md; the shipped value is 100000). The 4B wall time of one decode step over submits per step (13.5 ms) is not that metric: it also counts CPU encode time and gaps between submissions. The samples cover prefill and decode submissions together (not split). Only Qwen3-4B was sampled; the #407 rule takes 4B as the limiting model.

The raw profile has a second per-submission figure, the queue interval (time from queue start to queue end). At 120000 its p50 is about 6.2 ms (p99 about 11.5 ms, max about 13.2 ms); at 40000 its p50 is about 2.7 ms. By that figure the p50 sits just over 6 ms. The raw `subcap.jsonl.*` files (43 MB) are not in the repo; the lab keeps them under `artifacts/w9C/g14c-subcap/`.

## Limits

- Wall time per submission at 120000 is longer than the GPU time above: one decode step divided by submits per step is about 9 ms (2B), 12.4 ms (9B) and 13.5 ms (4B).
- The probe cannot rank budgets on the M2. A 98 ms eval (cap off) left the probe p99 at about 1 ms (`runs/probe-control`). It shows no frame cost; it does not show that none exists.
- NOT tested on a real compositor.
- Only G14C changes. Other rows keep their values; the test ceiling is per row (G14C 100000, others 40000).
