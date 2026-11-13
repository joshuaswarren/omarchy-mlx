# Width-256 value-phase SDPA: default by key-value heads times keys (2026-11-12)

`MLX_OMARCHY_SDPA_SPLIT_PV` selects the value-phase pass 1 of the split-K decode attention. For head width 256 (the Qwen3.5 and Qwen3.8 full-attention layers) the default was a key limit per chip: 768 keys on the Apple M1 (G13G), none on the Apple M1 Max (G13C). This receipt measures the variant against the old pass 1 on both chips at four head shapes and many context lengths, and replaces the key limit with a budget on key-value heads times keys: **1536 on G13G (768 keys of the Qwen3.8-2B) and 6144 on G13C (3072 keys of the 2B, 1536 keys of the 9B)**.

## What was measured

Op level: `mx.fast.scaled_dot_product_attention` on random bf16, head width 256, one decode query, K and V sliced from a larger buffer (the KV cache layout), split route forced, the two arms (`MLX_OMARCHY_SDPA_SPLIT_PV` 0 and 1) interleaved in ABBA blocks of 36 chained calls, median per call, bootstrap 95 percent interval of the gain. Probes are in `scripts/` (`sdpa-op-probe.py` is 8 query and 2 key-value heads; `-h4`, `-h12`, `-h16`, `-h32` are 4/1, 12/3, 16/4 and 32/8; `-cold` cycles 24 distinct K and V pairs through the chain, 8/2).

Model level (`sdpa-hd256-probe-chunk.py`): decode of a real model with the two arms interleaved step by step in one process, 128-token prefill chunks, per-arm median ms per token and the bootstrap interval of the median, plus the largest logit difference and whether the argmax matched.

Row names: `o<keys>` op level 8/2; `o<keys>h<q/kv>` op level at another head shape (h4 is 4 query and 1 key-value head, h12 12/3, h16 16/4, h32 32/8); `oc<keys>` cache-cold op level; `n<keys>` Qwen3.8-2B (8 query, 2 key-value heads) decode with other work running beside it; `k<keys>` the same with the neural engine idle; `m9` or `m9b`/`m9j` in the name is the Qwen3.8-9B (16 query, 4 key-value heads). Rows on the 13-inch G13G carry `r<n>` or `b<n>` repeat numbers. Rows named `q` ran with the neural engine idle.

## Result

Op level, the variant wins or loses by key-value heads times keys on a chip:

| Chip | Wins at (key-value heads x keys) | Loses at |
|---|---|---|
| G13G | 1536 (1 x 1536, 2 x 768), 1920 (4 x 480, 3 x 640), 2048 (2 x 1024, 4 x 512, 1 x 2048) | 2304 (3 x 768), 2560 (2 x 1280, 1 x 2560), 3072 (4 x 768, 3 x 1024), 6144 (8 x 768) |
| G13C | 6144 (8 x 768), 8192 (2 x 4096, 4 x 2048) | 12288 (3 x 4096, 4 x 3072, 8 x 1536, 1 x 12288), 16384 (4 x 4096) |

Model level decides the default, because it stops earlier than op level: on G13G the 2B gains up to 768 keys (1536) and loses from 1024 (2048); on G13C both models gain at 6144 and lose at 8192. That gives the budgets 1536 (G13G) and 6144 (G13C). The cause of the limit (cache footprint of K and V) is a hypothesis; no counter was read.

All rows (one run each unless the name carries a repeat number; `Per arm` is the number of timed repetitions of each arm):

#### G13G

| Row | Level | Keys | Stock | Variant | Gain % | CI95 | Per arm |
|---|---|---:|---:|---:|---:|---|---:|
| n1024r1 | model | 1024 | 60.44 ms | 60.516 ms | -0.13 | -0.59 to 0.29 | 128 |
| n1024r2 | model | 1024 | 61.131 ms | 61.57 ms | -0.72 | -0.9 to -0.37 | 64 |
| n1024r3 | model | 1024 | 60.146 ms | 60.26 ms | -0.19 | -0.59 to 0.09 | 64 |
| n1280r1 | model | 1280 | 63.86 ms | 64.457 ms | -0.93 | -3.75 to 2.01 | 64 |
| n1400r1 | model | 1400 | 60.18 ms | 60.331 ms | -0.25 | -0.38 to -0.06 | 64 |
| n1400r2 | model | 1400 | 63.455 ms | 63.497 ms | -0.07 | -1.26 to 1.25 | 64 |
| n1536r1 | model | 1536 | 62.425 ms | 62.589 ms | -0.26 | -4.18 to 2.86 | 64 |
| n1792r1 | model | 1792 | 60.23 ms | 60.281 ms | -0.09 | -0.22 to 0.01 | 64 |
| n2048r1 | model | 2048 | 60.169 ms | 60.438 ms | -0.45 | -0.72 to -0.22 | 64 |
| n2048r2 | model | 2048 | 64.91 ms | 65.738 ms | -1.28 | -1.78 to -0.96 | 64 |
| n256r1 | model | 256 | 61.688 ms | 61.456 ms | +0.38 | -1.48 to 2.01 | 128 |
| n480r1 | model | 480 | 62.002 ms | 61.798 ms | +0.33 | 0.13 to 0.61 | 128 |
| n480r2 | model | 480 | 60.142 ms | 59.814 ms | +0.55 | 0.14 to 1.13 | 128 |
| n480r3 | model | 480 | 60.527 ms | 60.786 ms | -0.43 | -1.91 to 2.21 | 64 |
| n640r1 | model | 640 | 62.393 ms | 61.162 ms | +1.97 | -2.96 to 3.86 | 64 |
| n768m9b2 | model | 768 | 197.046 ms | 197.23 ms | -0.09 | -0.26 to 0.08 | 64 |
| n768r1 | model | 768 | 60.173 ms | 59.904 ms | +0.45 | 0.12 to 0.76 | 64 |
| n768r2 | model | 768 | 60.222 ms | 59.892 ms | +0.55 | 0.06 to 1.08 | 64 |
| n768r3 | model | 768 | 61.428 ms | 61.088 ms | +0.55 | -3.29 to 4.45 | 64 |
| n832r2 | model | 832 | 61.152 ms | 60.835 ms | +0.52 | 0.15 to 1.13 | 64 |
| n900r1 | model | 900 | 61.154 ms | 61.015 ms | +0.23 | 0.04 to 0.6 | 64 |
| n900r2 | model | 900 | 61.214 ms | 60.787 ms | +0.70 | 0.36 to 0.97 | 64 |
| n900r3 | model | 900 | 61.343 ms | 60.978 ms | +0.59 | 0.07 to 0.9 | 64 |
| n960r1 | model | 960 | 61.541 ms | 62.396 ms | -1.39 | -3.84 to 2.48 | 64 |
| o1024 | op | 1024 | 156.79 us | 110.47 us | +29.55 | 29.32 to 29.87 | 200 |
| o1024h12 | op | 1024 | 174.2 us | 193.64 us | -11.16 | -11.21 to -11.12 | 3000 |
| o1280 | op | 1280 | 167.85 us | 192.18 us | -14.49 | -14.54 to -14.45 | 3000 |
| o1536 | op | 1536 | 174.76 us | 196.77 us | -12.60 | -12.65 to -12.55 | 3000 |
| o1536h4 | op | 1536 | 166.65 us | 119.27 us | +28.43 | 28.38 to 28.48 | 3000 |
| o1792 | op | 1792 | 184.08 us | 207.32 us | -12.62 | -12.68 to -12.57 | 3000 |
| o2048 | op | 2048 | 178.65 us | 204.75 us | -14.61 | -15.17 to -13.99 | 200 |
| o2048h4 | op | 2048 | 176.45 us | 129.36 us | +26.68 | 26.63 to 26.72 | 3000 |
| o2560h4 | op | 2560 | 186.37 us | 209.99 us | -12.67 | -12.75 to -12.61 | 2000 |
| o4096 | op | 4096 | 351.47 us | 519.22 us | -47.73 | -48.75 to -46.98 | 150 |
| o480 | op | 480 | 150.05 us | 101.99 us | +32.03 | 31.93 to 32.09 | 200 |
| o480h16 | op | 480 | 152.54 us | 105.31 us | +30.96 | 30.89 to 31.04 | 3000 |
| o512h16 | op | 512 | 155.17 us | 108.98 us | +29.77 | 29.74 to 29.79 | 3000 |
| o640h12 | op | 640 | 155.61 us | 108.67 us | +30.16 | 30.13 to 30.2 | 3000 |
| o768 | op | 768 | 156.49 us | 109.13 us | +30.27 | 30.15 to 30.41 | 200 |
| o768h12 | op | 768 | 155.42 us | 180.54 us | -16.16 | -16.25 to -16.08 | 3000 |
| o768h16 | op | 768 | 168.8 us | 190.57 us | -12.90 | -12.96 to -12.85 | 3000 |
| o768h32 | op | 768 | 212.75 us | 305.08 us | -43.40 | -43.51 to -43.33 | 3000 |
| o768h4 | op | 768 | 147.96 us | 100.09 us | +32.35 | 32.25 to 32.44 | 3000 |
| o768h8 | op | 768 | 156.89 us | 109.58 us | +30.15 | 30.12 to 30.19 | 3000 |
| o900 | op | 900 | 161.23 us | 114.28 us | +29.12 | 28.99 to 29.23 | 200 |
| oc1024 | op | 1024 | 185.79 us | 146.49 us | +21.16 | 21.1 to 21.23 | 3000 |
| oc768 | op | 768 | 178.9 us | 132.45 us | +25.97 | 25.91 to 26.01 | 3000 |

#### G13C

| Row | Level | Keys | Stock | Variant | Gain % | CI95 | Per arm |
|---|---|---:|---:|---:|---:|---|---:|
| k1024 | model | 1024 | 20.218 ms | 19.88 ms | +1.68 | 1.47 to 1.89 | 128 |
| k2048 | model | 2048 | 20.312 ms | 20.004 ms | +1.52 | 0.96 to 1.86 | 64 |
| k256 | model | 256 | 25.516 ms | 25.348 ms | +0.66 | 0.25 to 1.29 | 128 |
| k3072q | model | 3072 | 20.264 ms | 20.007 ms | +1.27 | 1.01 to 1.54 | 64 |
| k4096q | model | 4096 | 20.315 ms | 20.561 ms | -1.21 | -1.6 to -0.87 | 64 |
| k480 | model | 480 | 19.885 ms | 19.602 ms | +1.43 | 1.29 to 1.72 | 128 |
| k640 | model | 640 | 20.287 ms | 19.942 ms | +1.70 | 0.51 to 3.03 | 128 |
| n1024m9j | model | 1024 | 44.878 ms | 44.405 ms | +1.05 | 0.9 to 1.22 | 64 |
| n1536m9j | model | 1536 | 45.056 ms | 44.67 ms | +0.86 | 0.62 to 1.03 | 64 |
| n2048m9j | model | 2048 | 44.911 ms | 45.143 ms | -0.52 | -0.76 to -0.34 | 64 |
| n4096m9j | model | 4096 | 45.314 ms | 46.207 ms | -1.97 | -2.13 to -1.82 | 64 |
| n768m9j | model | 768 | 44.809 ms | 44.353 ms | +1.02 | 0.9 to 1.1 | 64 |
| o1024 | op | 1024 | 143.83 us | 96.1 us | +33.19 | 32.97 to 33.51 | 200 |
| o1024h16 | op | 1024 | 155.9 us | 105.65 us | +32.23 | 32.21 to 32.26 | 3000 |
| o12288h4 | op | 12288 | 303.71 us | 332.46 us | -9.47 | -9.51 to -9.43 | 1000 |
| o1536h32 | op | 1536 | 169.29 us | 195.9 us | -15.72 | -15.81 to -15.65 | 2000 |
| o2048 | op | 2048 | 149.04 us | 101.03 us | +32.21 | 32.06 to 32.3 | 200 |
| o2048h16 | op | 2048 | 170.65 us | 124.73 us | +26.91 | 26.89 to 26.93 | 3000 |
| o3072h16 | op | 3072 | 182.9 us | 211.73 us | -15.77 | -15.8 to -15.73 | 2000 |
| o4096 | op | 4096 | 177.2 us | 130.68 us | +26.26 | 26.15 to 26.49 | 150 |
| o4096h12 | op | 4096 | 196.72 us | 224.02 us | -13.88 | -13.92 to -13.85 | 1500 |
| o4096h16 | op | 4096 | 203.81 us | 230.36 us | -13.03 | -13.06 to -13.0 | 1500 |
| o4096h32 | op | 4096 | 318.45 us | 466.49 us | -46.49 | -46.52 to -46.46 | 1500 |
| o480 | op | 480 | 152.44 us | 99.73 us | +34.58 | 34.54 to 34.64 | 200 |
| o640 | op | 640 | 136.29 us | 88.32 us | +35.20 | 35.15 to 35.25 | 200 |
| o768 | op | 768 | 156.46 us | 103.05 us | +34.14 | 34.06 to 34.2 | 200 |
| o768h12 | op | 768 | 154.65 us | 103.04 us | +33.37 | 33.28 to 33.46 | 3000 |
| o768h16 | op | 768 | 153.61 us | 102.43 us | +33.32 | 33.27 to 33.36 | 3000 |
| o768h32 | op | 768 | 155.53 us | 108.78 us | +30.06 | 30.01 to 30.11 | 3000 |
| o900 | op | 900 | 158.22 us | 106.01 us | +33.00 | 32.94 to 33.06 | 200 |
| oc4096 | op | 4096 | 207.6 us | 161.8 us | +22.06 | 22.03 to 22.09 | 1500 |
| oc768 | op | 768 | 160.6 us | 107.49 us | +33.07 | 33.05 to 33.09 | 3000 |

## Environment

| | G13G (Apple M1, 13-inch) | G13C (Apple M1 Max, 16-inch) |
|---|---|---|
| Kernel | 7.3.0-2-20261110.1-lab-ARCH | 7.1.12-2-20261103.2-lab-ARCH |
| Mesa | 439dc032d7 | 439dc032d7 |
| mlx-omarchy wheel | 0.32.4.dev202611120448+8937c93e | 0.32.4.dev202611120448+8937c93e |
| mlx-lm | 0.32.2 | 0.32.2 |
| Gate source | `overlay/mlx/backend/omarchy/device.h`, this branch (not in the wheel above) | same |

Models: Qwen3.8-2B 4-bit (8 query and 2 key-value heads) and Qwen3.8-9B-Distill 4-bit (16 query and 4 key-value heads, 32 layers). On the 16-inch the stage copies hash to `c202656db3c08107` (2B) and `c05a1b77c4e175eb` (9B): the first 16 hex digits of the sha256 of `sha256sum *.safetensors config.json` run in the model directory. The 2B hash on the 13-inch was not recomputed.

## Commands

Op row: `SDPA_CTX=<keys> SDPA_NTOK=<reps> timeout 280 python scripts/<probe>.py` (reps are total, half per arm). Model row: `SDPA_CTX=<keys> SDPA_NTOK=128 SDPA_MODEL_DIR=<dir> python scripts/sdpa-hd256-probe-chunk.py`; the quiet rows ran with no neural engine work. Every run went through the lab's single GPU queue with a kernel, boot id and file-hash preflight.

Gate unit test: `scripts/gate-harness.sh` compiles the real `device.h` against the gate test body of `overlay/tests/omarchy/test_sdpa_split.cpp`. Result: against the old header it does not compile (the signature gained `kv_heads`); against the new header 31 checks, 0 failures; three mutations (G13C budget 8192, G13G budget 2048, product replaced by keys alone) fail 3, 3 and 6 checks.

## Not in this receipt

- The gate as built into a library and run on the two chips: the timing rows force the flag per arm (`0` and `1`), so they measure the two arms the gate chooses between, not the gate itself.
- A backend dispatch trace. The variant does not change the dispatch count, so a count would not show which pass 1 ran.
- Thermal logs, Vulkan device reopen after a run, and Vulkan firmware identity.
- The 9B on the 13-inch below 768 keys at model level (the 384 key limit for 4 key-value heads there rests on the op rows).
- Rows from earlier sessions in the same scratch directories (names a*, d*, m*) are left out: their protocol is not recorded.
