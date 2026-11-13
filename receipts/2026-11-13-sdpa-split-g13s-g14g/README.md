# Split-K decode attention on G13S and G14G (2026-11-13)

This receipt backs two new rows of `sdpa_split_min_keys` in `device.h`: the M1 Pro (G13S) and the M2 (G14G) take the split-K decode attention route from 192 keys. Before this change both parts kept the fused kernel at every length, because only G13G, G13C and G14C had rows.

## Machines and stack

| Board | Machine | OS | Vulkan device |
|---|---|---|---|
| G13S | MacBook Pro 16-inch 2021, M1 Pro (`apple,j316s`) | Fedora 45, Linux 7.3.0 | `Apple M1 Pro (G13S C0)`, Borealis, driver ID 35 |
| G14G | MacBook Air 13-inch 2022, M2 (`apple,j413`) | Fedora 44, Linux 7.3.0 | `Apple M2 (G14G B0)`, Borealis, driver ID 35 |

Release wheel v0.7.41 (`0.32.4.dev202611111431+8380f002`), Python 3.14, mlx-lm 0.31.3 with the `install.sh --patch-venv` series, `schedutil` governor, desktop session with no other GPU work. `MLX_OMARCHY_BATCH_WORK` was 20000 (M1 Pro) and 10000 (M2) in every arm. The new default is the route `MLX_OMARCHY_SDPA_DECODE_SPLIT=1` forces, so the release wheel with and without that flag measures the default before and after this change on these lengths.

## Break-even sweep

`tools/sdpa_bench_fine.py` is the script from `receipts/2026-10-23-sdpa-split-decode`, unchanged: single-query attention, 32 query heads over 8 key-value heads of width 128, bf16, 36 independent calls and a 36-call chain, median of 7. Arms: default (the fused kernel on these parts), `split` (`MLX_OMARCHY_SDPA_DECODE_SPLIT=1`), and `pv` (split plus `MLX_OMARCHY_SDPA_SPLIT_PV=1`). Raw output is in `sweeps/`.

Microseconds per call, chain / independent:

| Keys | M1 Pro fused | M1 Pro split | M2 fused | M2 split |
|---:|---|---|---|---|
| 128 | 55.6 / 55.3 | 96.4 / 91.5 | 86.7 / 86.6 | 88.1 / 80.8 |
| 192 | 114.4 / 117.2 | 91.2 / 90.6 | 154.8 / 153.7 | 89.2 / 88.3 |
| 256 | 149.6 / 151.6 | 92.3 / 91.3 | 201.5 / 202.4 | 88.2 / 88.3 |
| 512 | 305.1 / 292.7 | 120.9 / 116.4 | 374.9 / 374.1 | 94.5 / 93.3 |
| 1024 | 550.1 / 553.7 | 115.4 / 117.9 | 720.3 / 719.5 | 118.0 / 122.8 |

The threshold follows the rule in `receipts/2026-10-23-sdpa-split-decode`: the smallest measured length where the split route is faster in both shapes. That is 192 on both parts. At 128 keys the fused kernel still wins the chain on both (55.6 against 96.4, 86.7 against 88.1).

The value-phase variant (`pv` sweeps) was faster than the plain split up to 512 keys on both parts. Its default is not changed here, because on G13G the same microbenchmark win did not carry into decode (`sdpa_split_pv_max_keys`).

## Decode

`tools/longctx.py`: the llmstep.py prompt text from `receipts/2026-10-23-sdpa-split-decode`, repeated to 443 tokens, then 128 greedy tokens, one warm-up and two measured requests per process. Two rounds, default then split, for each model. Records are in `decode/` (M2 JSON; the M1 Pro log carries the same lines). Decode tokens per second, mean of the four requests per arm:

| Board | Model | Fused | Split | Ratio | Tokens |
|---|---|---:|---:|---:|---|
| M1 Pro | Qwen3-4B-Instruct-2507 4-bit | 32.38 | 41.52 | 1.282 | identical (`9e20669f96a7c584`) |
| M2 | Qwen3-4B-Instruct-2507 4-bit | 17.56 | 21.07 | 1.200 | identical (`9e20669f96a7c584`) |
| M1 Pro | Qwen3.8-2B 4-bit | 74.43 | 73.97 | 0.994 | differ |
| M2 | Qwen3.8-2B 4-bit | 39.39 | 39.16 | 0.994 | differ |

Qwen3.8-2B has only a few full-attention layers (head width 256), so the attention route barely moves its decode. Its greedy digest changes (`c4ed7eeb93e2eda6` to `91d5d220bf5b2c1d`, the same pair on both machines), as expected for a route that sums in a different order. `receipts/2026-10-23-sdpa-split-decode` covers this under its numerics section. Prefill is unchanged (the route is decode-only).

## Tests

`overlay/tests/omarchy/test_sdpa_split.cpp`, case "split-K decode SDPA default: the part rule and its key thresholds", pins both rows with the device names these drivers report, at 191 and 192 keys. The existing check that `Apple M2 Pro (G14S B1)` keeps the fused kernel still holds: "G14G" does not match G14S.

## Files

`sweeps/`, `decode/`, `tools/sdpa_bench_fine.py`, `tools/longctx.py`, `SHA256SUMS`.
