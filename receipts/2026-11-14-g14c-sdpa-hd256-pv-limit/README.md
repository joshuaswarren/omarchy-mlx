# G14C width-256 SDPA value-phase limit: 68 workgroups

Date: 2026-11-14. Part: Apple M2 Max (G14C), Mesa main ICD, wheel from main 6bf610a155, `MLX_OMARCHY_BATCH_WORK=100000`.
Rule changed: `sdpa_split_pv_wide_max_workgroups` in `overlay/mlx/backend/omarchy/device.h` returns 68 for G14C (it returned 0, so the width-256 value-phase pass 1 never ran by default there).
Method: `receipts/2026-11-12-sdpa-hd256-kv-heads-budget`. The count is key-value heads times 128-key blocks. Raw logs are in `logs/`, the scripts that made them in `scripts/` (the lab home directory is written `<HOME>`; set it before a rerun). A kernel-log line (`GPU-TIMEOUTS after=0 delta=0`) is appended into some log lines by the host; it is not part of a row.

## Op rows (`logs/run-g14.log`, `logs/run-g14b.log`)

`mx.fast.scaled_dot_product_attention`, random bf16, one decode query, 4 query heads per key-value head, split route forced, flag `0` against `1`, ABBA blocks of 36 chained calls, 64 reps per arm.

| workgroups | key-value heads | keys | variant against stock |
|---|---|---|---|
| 32 | 1 / 2 / 4 | 4096 / 2048 / 1024 | +24.83 / +27.64 / +29.25 percent |
| 60 | 1 / 2 / 4 | 7680 / 3840 / 1920 | +20.43 / +25.19 / +26.87 percent |
| 64 | 2 / 4 | 4096 / 2048 | +23.86 / +26.53 percent |
| 68 | 1 / 4 | 8704 / 2176 | +19.03 / +25.47 percent |
| 69 to 75 | 1 | 8832 to 9600 | -12.79 to -11.32 percent |
| 76 | 2 / 4 | 4864 / 2432 | -13.88 / -15.55 percent |
| 96 to 122 | 1 / 2 / 4 | 6144 to 12288 | -10.11 to -15.41 percent |

The edge is one block wide at 1 key-value head (68 wins, 69 loses, eight consecutive counts 68 to 75 sort once). At 2 and 4 key-value heads the edge lies between the last win (68) and the first loss tried (76).

## Float64 gate (`logs/run-f64.log`)

bf16 q, k and v, 8 shapes (8/2 heads at 512 to 4096 keys, 16/4 at 2048 and 4096, 4/1 at 4096), score sigma 0.5 and 2.0, each arm against a numpy float64 reference computed from the same inputs. The max abs error of the variant equals the error of the stock pass 1 in all 16 rows (0.00006 to 0.0039, the bf16 output rounding). The two arms differ by at most one bf16 ULP (0.00098 at 2176 keys, sigma 2.0, value near 1).

## Model rows (4-bit weights, bfloat16 activations, 128 decoded tokens, ABBA)

| model | context | key-value heads x blocks | default (stock) | variant | change | ids |
|---|---|---|---|---|---|---|
| Qwen3.8-9B | 512 | 4 x 5 = 20 | 42.42 | 43.08 | +1.57 percent | equal |
| Qwen3.8-9B | 2048 | 4 x 17 = 68 | 42.12 | 42.74 | +1.47 percent | equal |
| Qwen3.8-9B | 4096 | 4 x 33 = 132 | 41.79 | 41.28 | -1.23 percent | equal |
| Qwen3.8-2B | 512 | 2 x 5 = 10 | 114.33 | 118.21 | +3.40 percent | equal |
| Qwen3.8-2B | 2048 | 2 x 17 = 34 | 111.74 | 116.94 | +4.66 percent | differ (rounding order) |
| Qwen3.8-2B | 4096 | 2 x 33 = 66 | 110.26 | 112.97 | +2.45 percent | differ (rounding order) |

Logs: `logs/run-wd.log` (9B at 512 and 2048), `logs/run-wd2.log` (2B at 512 and 2048), `logs/run-wd3b.log` (9B and 2B at 4096). The 9B at 4096 has 132 workgroups, above the limit, so the new default keeps the stock pass 1 there; the row is the forced variant and shows the limit is not too high. The 2B ids differ at 2048 and 4096 because the variant adds the four partial sums in another order; the float64 gate above shows the error is the same. A first run of the 4096 row (`run-wd3`) was discarded: a wheel build ran on the same CPU (9B arm 25.0 tok/s against 41.9).

## Tests

`overlay/tests/omarchy/test_sdpa_split.cpp` checks the G14C limit and its edge at 1, 2, 3, 4 and 8 key-value heads. `scripts/gate-harness.sh new` (the harness of the 2026-11-12 receipt): 54 checks, 0 fails; with the limit mutated to 67 it fails 4 checks and to 69 it fails 3.
