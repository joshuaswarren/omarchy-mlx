# RTX 5060 Ti decode: compiled tape A/B is a null — batch-work 40000 already owns the gain

Date: 2026-11-27. Device: NVIDIA GeForce RTX 5060 Ti 16 GB (PCI 0x2d04) on Thunderbolt, NVIDIA driver 615.78.08 (its Vulkan ICD, apiVersion 1.4.351, DRIVER_ID_NVIDIA_PROPRIETARY), no display attached. Host: a 16-inch M1 Max under Linux kernel `7.1.12-2-20261103.2-lab-ARCH`, one boot for every run here. Runtime env on both sides: `MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_VRAM_ACTIVATIONS=1 MLX_OMARCHY_VRAM_WEIGHTS=<model dir>`. Wheel `mlx 0.32.4.dev202611261500+e1f6a24`, provenance verified against the installed RECORD; `e1f6a24` carries `{"RTX 5060 Ti", 40000}` in `kBatchWorkCalibration`, so every run here submitted decode at the calibrated budget, not the 5000 fallback.

## Verdict

The `MLX_DISABLE_COMPILE` flag does not move decode on this path. Five interleaved pairs (ABBA), 64 greedy tokens each:

| Qwen3-4B-Instruct-2507 4-bit, 64 greedy tokens | median tok/s | run order |
|---|---|---|
| a1 `MLX_DISABLE_COMPILE=1` | 114.27 | 1 |
| a2 | 115.92 | 4 |
| a3 | 114.41 | 5 |
| a4 | 114.56 | 8 |
| a5 | 113.87 | 9 |
| b1 compile on | 114.80 | 2 |
| b2 | 112.41 | 3 |
| b3 | 113.96 | 6 |
| b4 | 113.91 | 7 |
| b5 | 113.37 | 10 |
| b6bw compile on, `MLX_OMARCHY_BATCH_WORK=40000` | 108.89 | 11 |

A median of medians 114.41, B 113.91; every per-arm range (112.4–115.9) overlaps. All 11 runs produced the same 64 token ids (sha256 `a6fad159dfa9`). `b6bw` is config-identical to the b-arms (the env override equals the table row) and ran last; its −4.4% is drift, not an effect.

## Why it is a null

Two independent reasons, both verified in source at `e1f6a24`:

1. **Nothing on this decode path calls `mx.compile`.** `tools/profile_decode_driver.py` drives `generate_step` directly; `MLX_DISABLE_COMPILE` only gates the compiled-tape/FusedChain route, which this workload never enters. The flag is inert here.
2. **There is nothing left to coalesce.** The merged `{"RTX 5060 Ti", 40000}` calibration row already runs one command buffer per decode token at this wheel. FusedChain's win is submission coalescing; at ~1 submission/token the tape can at most tie.

The workload is still submission-bound when a budget forces splitting: the earlier probe arm `MLX_OMARCHY_BATCH_WORK=1` (every dispatch its own submission) collapsed decode to **36.47 tok/s** on the same wheel and ids — the per-submission fixed cost remains the dominant lever, and it is already pulled.

## Artifact caution

A same-session probe earlier in the day measured `MLX_DISABLE_COMPILE=1` at 102.81 vs compile-on 113.96 (+10.8%) — the interleaved ABBA falsifies that. The probe's base arm was the first MLX process after several minutes of CUDA `llama-server` runs; the gap was cold-start order, not the flag. First-in-session arms cannot be compared against later arms; the pairs above interleave from a warm state.

## Reference arms

Same card, same model family, llama.cpp `llama-server` (build 614, CUDA backend, `--flash-attn`), 128 tokens, single measurement each — CUDA graphs are already engaged and near-saturated:

| arm | predicted_per_second |
|---|---|
| graphs on (default) | 99.62 |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 98.14 |
| `GGML_CUDA_GRAPH_OPT=1` | 98.40 |

MLX-Vulkan on the same card (113–116 tok/s) leads llama.cpp CUDA (~98–100) on the same model class.

## Method

`ab.sh` (in `runs/`) runs each arm as a fresh process: 8 warmup tokens on a throwaway cache, 512-token prompt, then 64 greedy tokens through `tools/profile_decode_driver.py`, one host sync per token, per-token `t0/t1` monotonic_ns in the `.jsonl`. Arms interleave a,b,b,a,a,b,b,a,a,b then the info arm. The card ran alone under the `GPU_TURN_DEVICE=5060` gpu-turn ticket. `runs/` holds every arm's `.jsonl`, `.log`, the probe's four `mlx-*.jsonl`, and the three llama.cpp `srv-*.log`/`resp-*.json`.

## Model

`mlx-community/Qwen3-4B-Instruct-2507-4bit` snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`: `model.safetensors` sha256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f` (full digest; receipt table above truncates at 16 hex for readability).
