# G14C (Apple M2 Max): GatedDelta conv + delta + gated norm default

Date: 2026-11-14. Part: Apple M2 Max (G14C), Mesa main Vulkan manifest, wheel from main 6bf610a155, mlx-lm with the gdn-conv-delta patch, `MLX_OMARCHY_BATCH_WORK=100000`.
Rule changed: `gdn_conv_delta_norm_default_part` in `overlay/mlx/backend/omarchy/device.h` now returns true for G14C as well as G13G, so an unset `MLX_OMARCHY_GDN_CONV_DELTA` resolves to 2 on the M2 Max (conv, norm and delta step as one dispatch instead of three per GatedDelta layer).

## Method

Decode, 128 tokens, bfloat16 activations, 4-bit weights, one process per arm, order default, flag 2, flag 2, default (ABBA, 2 runs per arm), prompt of 512 and 2048 tokens. `logs/run-gcd.log` has every arm; `scripts/` has the runner and the driver (the lab home directory is written `<HOME>`, and the runner reads the Mesa manifest from `VK_ICD_FILENAMES`). The host appended a kernel-log line (`GPU-TIMEOUTS after=0 delta=0`) into one summary line (9B at 2048); its two arms per side are readable above it.

## Rows

| model | context | default tok/s | flag 2 tok/s | change | ids |
|---|---|---|---|---|---|
| Qwen3.8-2B (18 GatedDelta layers) | 512 | 114.24 | 116.53 | +2.00 percent | equal |
| Qwen3.8-2B | 2048 | 113.01 | 114.39 | +1.22 percent | equal |
| Qwen3.8-9B (24 GatedDelta layers) | 512 | 42.38 | 42.34 | -0.08 percent | equal |
| Qwen3.8-9B | 2048 | 42.05 | 41.99 | -0.14 percent | equal |

The 2B gains 0.17 ms and 0.11 ms per token. The 9B is flat: the arms of one side differ by about 0.2 percent among themselves. The prediction was +0.5 to +3 percent on the 2B (held) and +0.3 to +2 percent on the 9B (not held: flat, no loss beyond the spread).
