# G14C submission budget 40000 -> 120000 (2026-11-13)

Board: 14-inch M2 Max (G14C), Linux, Mesa honeykrisp, `MLX_OMARCHY_BATCH_WORK` unset (table) vs `120000` vs `200000`.
600-token decode, runs interleaved, 20 s gaps, a 60 Hz submit-latency probe beside each run.
First 64 generated token ids identical across arms in every pair (hash `ids_sha`); logprobs were not recorded in these runs. Scheduling only.

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

## Limits

- Submissions at 120000 take about 9 ms (2B), 12.4 ms (9B) and 13.5 ms (4B), computed as one decode step divided by submits per step. That is above the 2 to 6 ms band the doc comment describes.
- The probe cannot rank budgets on the M2. A 98 ms eval (cap off) left the probe p99 at about 1 ms (`runs/probe-control`). It shows no frame cost; it does not show that none exists.
- NOT tested on a real compositor.
- Only G14C changes. Other rows keep their values; the test ceiling is per row (G14C 120000, others 40000).
