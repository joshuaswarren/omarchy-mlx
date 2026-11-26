# 2026-11-18 G14C: default-on MLX_OMARCHY_WAVE_SCHED

M2 Max (G14C B1), omarchy-mlx 0.7.41-2 +17a9718a. Harness: tools/wavesched-g14c-ship.sh.

## Qwen3-4B-4bit (three pairs, 128 greedy tokens)

| pair | off tok/s | on tok/s | ids sha16 |
|-----:|----------:|---------:|-----------|
| 1 | 6.42 | 10.81 | 3396d7dbb4b5bf28 |
| 2 | 9.95 | 9.82 | 3396d7dbb4b5bf28 |
| 3 | 9.49 | 7.68 | 3396d7dbb4b5bf28 |

numcmp off vs on: max_abs_dlogprob_full=0.0, ids_equal=True (32 full steps).

## Qwen3.8-2B-4bit

Decode ~125 tok/s every arm (flat). numcmp: max_abs_dlogprob_full=0.0, ids 49e83a4dba5b1356.

## Ship

wave_sched_default_part adds G14C; G13C/G13G/G17C stay off until their perf rows.
