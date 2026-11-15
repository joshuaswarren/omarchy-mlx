# Re-pick: 120000 -> 100000 (2026-11-13)

Rule (omarchy-mlx #407): the largest swept budget where Qwen3-4B decode has submission p50 at or under 6 ms and p99 near 40 ms or less, with no sample near 78 ms. It is applied here on BOTH the GPU dispatch sum and the queue interval (`queue_interval.py`, `queue_interval_90k_110k.py`; field `dur` of the `"k":"s"` records).

## Both metrics, Qwen3-4B decode, ms (p50 / p99 / max)

| Budget | GPU dispatch sum | Queue interval | passes p50 at or under 6 |
|---|---|---|---|
| 40000 | 1.52 / 3.12 / 3.39 | 2.65 / 5.50 / 6.45 | yes |
| 60000 | 2.33 / 4.53 / 5.14 | 4.12 / 6.43 / 8.75 | yes |
| 80000 | 2.88 / 5.29 / 6.68 | 4.98 / 8.85 / 9.94 | yes |
| 90000 | 3.11 / 5.83 / 7.46 | 4.70 / 9.85 / 9.85 | yes |
| 100000 | 3.87 / 6.32 / 8.55 | 4.23 / 10.58 / 12.78 | yes |
| 110000 | 4.20 / 5.91 / 9.08 | 5.75 / 11.00 / 14.61 | yes, margin 0.25 ms |
| 120000 | 4.54 / 5.93 / 10.06 | 6.29 / 11.52 / 13.24 | NO |

One run per budget. 100000 reads lower than 90000 on the queue interval, so run noise is about 0.5 ms. 110000 passes by 0.25 ms, inside that noise. 100000 is the pick: p50 margin 1.8 ms. Main approved it.

## Speed at 100000 vs default 40000 (`batchwork100k`, interleaved pairs, tok/s)

| Model | default | 100000 | gain | pairs |
|---|---|---|---|---|
| 4B | 68.65 / 68.90 / 67.99 / 68.02 | 69.21 / 69.41 / 68.06 / 68.30 | +0.5% mean (+0.8, +0.7, +0.1, +0.4) | 4 of 4 positive |
| 2B | 108.00 / 108.25 / 107.44 | 109.32 / 108.78 / 106.48 | +0.3% mean (+1.2, +0.5, -0.9) | 2 of 3 |
| 9B | 37.70 / 38.02 / 37.98 | 38.33 / 38.69 / 38.31 | +1.4% mean (+1.7, +1.8, +0.9) | 3 of 3 |

First 64 token ids identical in all 20 runs (logprobs not recorded); 0 of 20 runs had a frame over 33.3 ms beside decode (probe, not a real compositor).
My prediction for 4B was +0.8 to 1.4%; it came in at +0.5%, so the gain is smaller than at 120000 (+1.2 to 1.5%). Submissions per step on 4B: 3.19 default, 2.09 at 100000 (1.07 at 120000).
At 80000 (`batchwork80k`, 3 pairs): 4B +0.1%, 2B +1.5%, 9B +1.2%. w7P's data agrees: 80000 vs 40000 is 1.02.
The earlier 120000 tables stay in the README as the swept comparison.
Method: GPU rows use inclusive quantiles (as `subcap_calibrate.py` does); queue-interval rows use nearest-rank percentiles. They can differ by up to 1.5 ms at p99. The p50 verdicts do not change. The queue-interval scripts read the jsonl files in `~/.local/share/apple-silicon-lab/artifacts/w9C/g14c-subcap*` (not in this repo, 8 MB each).

## Reruns (`runs/*-rerun`, 6 interleaved pairs each, first 64 ids identical, 0 frames over 33.3 ms)

| Row | Gain | Pairs |
|---|---|---|
| 4B at 100000 | +0.48% (+0.96, +0.22, +0.59, +0.69, +0.12, +0.32) | 6 of 6 |
| 2B at 100000 | +1.27% (+1.79, +1.21, +1.37, +0.85, +1.18, +1.21) | 6 of 6 |
| 4B at 80000 | +0.83% (+0.80, +0.41, +0.47, +0.82, +1.27, +1.23) | 6 of 6 |

Pooled with the first run at 100000: 4B +0.49% over 10 pairs, 2B +0.9% over 9 pairs. The first-run 2B row above (+0.3%) and 4B at 80000 (+0.1%) were low draws.
