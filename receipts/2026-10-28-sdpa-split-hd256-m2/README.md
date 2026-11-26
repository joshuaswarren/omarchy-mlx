# Split-K decode attention for head width 256 on the 14-inch M2 Max (2026-10-28)

Follow-up to `receipts/2026-10-28-sdpa-split-hd256/` (13-inch M1), merged as #293. The same protocol ran on the 14-inch M2 Max (G14C B1, 96 GB), Honeykrisp, Release build of the branch head `ff4b52ac` (its shaders are byte-identical to the merged ones at width 256). One build serves both arms: arm A sets `MLX_OMARCHY_SDPA_DECODE_SPLIT=0`, arm B leaves it unset (default threshold on this part: 320 keys). Arms interleaved A, B, A, B, A, B, one run each.

## Decode tokens per second, three interleaved pairs

| Cell | route off | route on | ratio per pair |
| --- | --- | --- | --- |
| Qwen3.5-9B 4-bit, 8192 tokens | 19.57, 19.04, 19.51 | 35.16, 35.91, 35.76 | 1.80, 1.89, 1.83 |
| Qwen3.8-2B 4-bit, 8192 tokens | 43.75, 43.76, 44.08 | 94.39, 93.77, 95.50 | 2.16, 2.14, 2.17 |
| Qwen3.8-2B 4-bit, 32768 tokens | 16.65, 16.68, 16.57 | 64.33, 64.52, 63.62 | 3.86, 3.87, 3.84 |

Prefill is unchanged (for example 1077 to 1088 tokens per second on the 2B at 32768).

## Short context (the pre-registered check), two interleaved repetitions

| Model, context | route off | route on | B over A |
| --- | --- | --- | --- |
| Qwen3.8-2B, 128 | 98.91, 99.75 | 98.29, 100.00 | 0.994, 1.003 |
| Qwen3.8-2B, 256 | 95.77, 96.04 | 95.18, 95.80 | 0.994, 0.998 |
| Qwen3.5-9B, 128 | 37.55, 37.71 | 37.48, 37.61 | 0.998, 0.997 |
| Qwen3.5-9B, 256 | 37.02, 37.15 | 36.96, 37.12 | 0.998, 0.999 |

The route adds 0 dispatches per token here (254 on the 2B and 384 on the 9B in both arms at these contexts, below the threshold). Nothing measurable at short contexts, against 1 to 2 percent on the 2B on the 13-inch M1.

## Attention alone, Qwen3.5-9B shape (head width 256, 16 and 4 heads), bf16

| Keys | route off: microseconds per call | route on |
| --- | --- | --- |
| 2048 | 344 | 157 |
| 8192 | 2282 | 183 |
| 32768 | 8666 | 612 |

(Eight chained calls on distinct caches. The copy bandwidth of this machine is 331 GB/s in the same run; the route off reads the cache at 15 GB/s, the route on at 183 GB/s at 8192 keys.)

## Correctness

- `omarchy_sdpa_split_tests` on the M2 Max GPU: 11 of 11 cases, 791 assertions, route taken.
- Greedy decode of a fixed 419-token prompt, 48 new tokens, route off against on: 48 of 48 token ids equal for the 2B and the 9B.

## Not run on this machine

The teacher-forced numerics (run on the 13-inch M1: mean NLL difference within 0.002 nats per token), the width-128 before/after check, and contexts above 32768 tokens. Logs are in `logs/` with `SHA256SUMS`.
