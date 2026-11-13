# Width-256 value-phase SDPA: default by pass 1 workgroup count (2026-11-12)

`MLX_OMARCHY_SDPA_SPLIT_PV` selects the value-phase pass 1 of the split-K decode attention. For head width 256 (the Qwen3.5 and Qwen3.8 full-attention layers) the default was a key limit per chip: 768 keys on the Apple M1 (G13G), none on the Apple M1 Max (G13C). Pass 1 launches one workgroup per key-value head and 128-key block. This receipt measures the variant against the old pass 1 on both chips at five head counts and many lengths, and sets the default by that workgroup count: **on while key-value heads times ceil(keys / 128) is at most 16 on G13G and at most 64 on G13C**, which is two workgroups per GPU core (8 and 32 cores). Width 128 follows the same rule at twice the count (32 and 128).

## What was measured

Op level: `mx.fast.scaled_dot_product_attention` on random bf16, head width 256, one decode query, K and V sliced from a larger buffer (the KV cache layout), split route forced, the two arms (`MLX_OMARCHY_SDPA_SPLIT_PV` 0 and 1) interleaved in ABBA blocks of 36 chained calls, median per call, bootstrap 95 percent interval of the gain. Probes are in `scripts/` (`sdpa-op-probe.py` is 8 query and 2 key-value heads; `-h4`, `-h12`, `-h16`, `-h32` are 4/1, 12/3, 16/4 and 32/8; `-cold` cycles 24 distinct K and V pairs through the chain, 8/2; `-gen` takes `SDPA_QH`, `SDPA_KVH` and `SDPA_HD`, which the width-128 rows use).

Model level (`sdpa-hd256-probe-chunk.py`): decode of a real model with the two arms interleaved step by step in one process, 128-token prefill chunks, per-arm median ms per token and the bootstrap interval of the median, plus the largest logit difference (bf16, maximum over the timed steps) and whether the argmax matched.

Engagement at the model's own attention call (`sdpa-hd256-probe-engage.py`): the probe wraps `mx.fast.scaled_dot_product_attention`, captures the first width-256 single-query call of a decode step (the real query and the real cache views), and runs that exact call under both flag values: the largest output difference between the arms, each arm's error against a float64 reference of the same inputs, and the per-call time of each arm (ABBA blocks of 36 chained calls, 400 blocks, bootstrap interval). A row where the arms time the same did not run the variant.

Float64 gate at op level (`sdpa-f64-check.py`): the fused kernel, the split route with the flag off, and the split route with the flag on, each against a float64 reference, with the bounds of `check_shape` in `test_sdpa_split.cpp`.

Row names: `o<keys>` op level 8/2; `o<keys>h<q/kv>` op level at another head shape (h4 is 4 query and 1 key-value head, h8 is 8/2 again, h12 12/3, h16 16/4, h32 32/8); `oc<keys>` cache-cold op level; `d128h<q>c<keys>` width 128 at `<q>` query heads (h8 is 8/2, h32 is 32/8); `n<keys>` Qwen3.8-2B (8 query, 2 key-value heads) decode with other work running beside it; `k<keys>` the same with the neural engine idle; `m9` or `m9b`/`m9j` in the name is the Qwen3.8-9B (16 query, 4 key-value heads). Rows on the 13-inch G13G carry `r<n>` or `b<n>` repeat numbers. Rows named `q` ran with the neural engine idle. `e2b<keys>` and `e9b<keys>` are engagement rows (2B and 9B at context `<keys>`, so the call sees `<keys> + 1` keys), `f64h<q>c<keys>` float64 gate rows.

## Result

Count workgroups as key-value heads times ceil(keys / 128). Sorted that way, the op rows separate without overlap at 1, 2, 3, 4 and 8 key-value heads:

| Chip | Wins at (workgroups) | Loses at (workgroups) |
|---|---|---|
| G13G, width 256 | 6 to 16 | 17 to 64 |
| G13C, width 256 | 8 to 64 | 65 to 256 |
| G13G, width 128 | up to 32 | 34 and above |
| G13C, width 128 | up to 128 | 130 and above |

The boundary is one block wide: G13G at 1 key-value head wins +26.68 percent at 2048 keys (16 workgroups) and loses at 2049 (17); G13C at 1 key-value head wins +22.48 at 8192 keys (64) and loses at 8193 (65). The per-row lists are in the tables below.

A decode step at context N attends N + 1 keys, so a model row at a multiple of 128 sits one key past a block boundary. The model's own attention call (engagement table) wins at 3073 keys (50 workgroups) and loses at 4097 (66) on G13C, and wins at 769 keys (14) and loses at 1025 (18) on G13G; the random-data op rows give the same sign at the same key counts (G13C 3073 +28.39, 4097 -7.13).

The model-level gain is the per-call saving times the six full-attention layers of the 2B (24 layers, one in four): 47.8 us x 6 = 0.29 ms at 481 keys, against 0.28 ms measured in `k480`; 46.7 us x 6 = 0.28 ms at 3073 keys, against 0.26 ms in `k3072q`. The op-level percentage is a share of the attention call, the model-level percentage a share of the token.

All rows are single runs (one row name per run) unless the name carries a repeat number; `Per arm` is the number of timed repetitions of each arm.


## Tables

Op rows have no logit difference (the column reads 0.0 there); it is measured at model level only.

#### G13G

| Row | Level | Keys | Stock | Variant | Gain % | CI95 | Per arm | Logit diff |
|---|---|---:|---:|---:|---:|---|---:|---:|
| d128h32c512 | op | 512 | 93.67 us | 75.99 us | +18.88 | 18.78 to 18.98 | 3000 | 0.0 |
| d128h32c640 | op | 640 | 98.8 us | 118.11 us | -19.54 | -19.68 to -19.39 | 3000 | 0.0 |
| d128h8c2048 | op | 2048 | 104.09 us | 86.05 us | +17.33 | 17.2 to 17.48 | 3000 | 0.0 |
| d128h8c2049 | op | 2049 | 105.33 us | 118.26 us | -12.27 | -12.43 to -12.12 | 3000 | 3.725290298461914e-09 |
| d128h8c2176 | op | 2176 | 111.61 us | 133.27 us | -19.40 | -19.49 to -19.35 | 3000 | 0.0 |
| d128h8c2560 | op | 2560 | 110.33 us | 129.37 us | -17.26 | -17.59 to -16.91 | 2500 | 0.0 |
| n1024r1 | model | 1024 | 60.44 ms | 60.516 ms | -0.13 | -0.59 to 0.29 | 128 | 0.0 |
| n1024r2 | model | 1024 | 61.131 ms | 61.57 ms | -0.72 | -0.9 to -0.37 | 64 | 0.0 |
| n1024r3 | model | 1024 | 60.146 ms | 60.26 ms | -0.19 | -0.59 to 0.09 | 64 | 0.0 |
| n1280r1 | model | 1280 | 63.86 ms | 64.457 ms | -0.93 | -3.75 to 2.01 | 64 | 0.125 |
| n1400r1 | model | 1400 | 60.18 ms | 60.331 ms | -0.25 | -0.38 to -0.06 | 64 | 0.0 |
| n1400r2 | model | 1400 | 63.455 ms | 63.497 ms | -0.07 | -1.26 to 1.25 | 64 | 0.0 |
| n1536r1 | model | 1536 | 62.425 ms | 62.589 ms | -0.26 | -4.18 to 2.86 | 64 | 0.0 |
| n1792r1 | model | 1792 | 60.23 ms | 60.281 ms | -0.09 | -0.22 to 0.01 | 64 | 0.0 |
| n2048r1 | model | 2048 | 60.169 ms | 60.438 ms | -0.45 | -0.72 to -0.22 | 64 | 0.0 |
| n2048r2 | model | 2048 | 64.91 ms | 65.738 ms | -1.28 | -1.78 to -0.96 | 64 | 0.0 |
| n256r1 | model | 256 | 61.688 ms | 61.456 ms | +0.38 | -1.48 to 2.01 | 128 | 0.0 |
| n384m9b | model | 384 | 210.716 ms | 210.398 ms | +0.15 | -1.42 to 1.76 | 64 | 0.125 |
| n480r1 | model | 480 | 62.002 ms | 61.798 ms | +0.33 | 0.13 to 0.61 | 128 | 0.0 |
| n480r2 | model | 480 | 60.142 ms | 59.814 ms | +0.55 | 0.14 to 1.13 | 128 | 0.0 |
| n480r3 | model | 480 | 60.527 ms | 60.786 ms | -0.43 | -1.91 to 2.21 | 64 | 0.0 |
| n512m9b | model | 512 | 197.126 ms | 197.599 ms | -0.24 | -1.96 to 0.09 | 64 | 0.064453125 |
| n640r1 | model | 640 | 62.393 ms | 61.162 ms | +1.97 | -2.96 to 3.86 | 64 | 0.0 |
| n768m9b2 | model | 768 | 197.046 ms | 197.23 ms | -0.09 | -0.26 to 0.08 | 64 | 0.0 |
| n768q1 | model | 768 | 37.317 ms | 37.082 ms | +0.63 | 0.54 to 0.76 | 64 | 0.0 |
| n768r1 | model | 768 | 60.173 ms | 59.904 ms | +0.45 | 0.12 to 0.76 | 64 | 0.0 |
| n768r2 | model | 768 | 60.222 ms | 59.892 ms | +0.55 | 0.06 to 1.08 | 64 | 0.0 |
| n768r3 | model | 768 | 61.428 ms | 61.088 ms | +0.55 | -3.29 to 4.45 | 64 | 0.0 |
| n832r2 | model | 832 | 61.152 ms | 60.835 ms | +0.52 | 0.15 to 1.13 | 64 | 0.0 |
| n900r1 | model | 900 | 61.154 ms | 61.015 ms | +0.23 | 0.04 to 0.6 | 64 | 0.0625 |
| n900r2 | model | 900 | 61.214 ms | 60.787 ms | +0.70 | 0.36 to 0.97 | 64 | 0.0625 |
| n900r3 | model | 900 | 61.343 ms | 60.978 ms | +0.59 | 0.07 to 0.9 | 64 | 0.0625 |
| n960r1 | model | 960 | 61.541 ms | 62.396 ms | -1.39 | -3.84 to 2.48 | 64 | 0.0 |
| o1024 | op | 1024 | 156.79 us | 110.47 us | +29.55 | 29.32 to 29.87 | 200 | 0.0001220703125 |
| o1024h12 | op | 1024 | 174.2 us | 193.64 us | -11.16 | -11.21 to -11.12 | 3000 | 1.9073486328125e-06 |
| o1025 | op | 1025 | 164.06 us | 177.25 us | -8.05 | -8.08 to -8.0 | 3000 | 1.1920928955078125e-07 |
| o1025h4 | op | 1025 | 152.69 us | 104.69 us | +31.43 | 31.37 to 31.5 | 3000 | 0.0 |
| o1280 | op | 1280 | 167.85 us | 192.18 us | -14.49 | -14.54 to -14.45 | 3000 | 0.0001220703125 |
| o1536 | op | 1536 | 174.76 us | 196.77 us | -12.60 | -12.65 to -12.55 | 3000 | 1.52587890625e-05 |
| o1536h4 | op | 1536 | 166.65 us | 119.27 us | +28.43 | 28.38 to 28.48 | 3000 | 9.5367431640625e-07 |
| o1792 | op | 1792 | 184.08 us | 207.32 us | -12.62 | -12.68 to -12.57 | 3000 | 0.0 |
| o1792h4 | op | 1792 | 170.89 us | 123.51 us | +27.73 | 27.68 to 27.78 | 3000 | 0.0 |
| o2048 | op | 2048 | 178.65 us | 204.75 us | -14.61 | -15.17 to -13.99 | 200 | 0.0 |
| o2048h4 | op | 2048 | 176.45 us | 129.36 us | +26.68 | 26.63 to 26.72 | 3000 | 0.0 |
| o2049h4 | op | 2049 | 177.53 us | 189.8 us | -6.91 | -6.95 to -6.86 | 3000 | 1.1920928955078125e-07 |
| o2112h4 | op | 2112 | 178.56 us | 200.78 us | -12.45 | -12.48 to -12.4 | 3000 | 0.0 |
| o2176h4 | op | 2176 | 178.0 us | 202.71 us | -13.88 | -13.94 to -13.82 | 3000 | 0.0 |
| o2560h4 | op | 2560 | 173.6 us | 198.56 us | -14.37 | -14.5 to -14.25 | 3000 | 0.0 |
| o320h32 | op | 320 | 158.5 us | 183.54 us | -15.80 | -15.85 to -15.75 | 3000 | 0.0 |
| o4096 | op | 4096 | 351.47 us | 519.22 us | -47.73 | -48.75 to -46.98 | 150 | 7.62939453125e-06 |
| o480 | op | 480 | 150.05 us | 101.99 us | +32.03 | 31.93 to 32.09 | 200 | 7.62939453125e-06 |
| o480h16 | op | 480 | 152.54 us | 105.31 us | +30.96 | 30.89 to 31.04 | 3000 | 7.450580596923828e-09 |
| o512h16 | op | 512 | 155.17 us | 108.98 us | +29.77 | 29.74 to 29.79 | 3000 | 0.0 |
| o513h16 | op | 513 | 154.18 us | 167.48 us | -8.63 | -8.72 to -8.54 | 3000 | 0.0 |
| o640h12 | op | 640 | 156.65 us | 109.87 us | +29.86 | 29.82 to 29.89 | 3000 | 1.52587890625e-05 |
| o641h12 | op | 641 | 154.61 us | 167.06 us | -8.05 | -8.15 to -7.96 | 3000 | 0.0 |
| o768 | op | 768 | 156.49 us | 109.13 us | +30.27 | 30.15 to 30.41 | 200 | 0.0 |
| o768h12 | op | 768 | 155.42 us | 180.54 us | -16.16 | -16.25 to -16.08 | 3000 | 1.9073486328125e-06 |
| o768h16 | op | 768 | 168.8 us | 190.57 us | -12.90 | -12.96 to -12.85 | 3000 | 5.960464477539063e-08 |
| o768h32 | op | 768 | 212.75 us | 305.08 us | -43.40 | -43.51 to -43.33 | 3000 | 0.0001220703125 |
| o768h4 | op | 768 | 147.96 us | 100.09 us | +32.35 | 32.25 to 32.44 | 3000 | 0.0 |
| o768h8 | op | 768 | 156.89 us | 109.58 us | +30.15 | 30.12 to 30.19 | 3000 | 0.0 |
| o900 | op | 900 | 161.23 us | 114.28 us | +29.12 | 28.99 to 29.23 | 200 | 7.450580596923828e-09 |
| oc1024 | op | 1024 | 185.79 us | 146.49 us | +21.16 | 21.1 to 21.23 | 3000 | 0.0001220703125 |
| oc768 | op | 768 | 178.9 us | 132.45 us | +25.97 | 25.91 to 26.01 | 3000 | 0.0 |

#### G13C

| Row | Level | Keys | Stock | Variant | Gain % | CI95 | Per arm | Logit diff |
|---|---|---:|---:|---:|---:|---|---:|---:|
| d128h32c2048 | op | 2048 | 93.57 us | 74.18 us | +20.72 | 20.55 to 20.89 | 2500 | 0.0 |
| d128h32c2049 | op | 2049 | 93.57 us | 105.23 us | -12.46 | -12.5 to -12.43 | 2000 | 2.9802322387695312e-08 |
| d128h32c3072 | op | 3072 | 105.74 us | 123.2 us | -16.52 | -16.65 to -16.39 | 2000 | 3.725290298461914e-09 |
| d128h8c12288 | op | 12288 | 165.85 us | 181.74 us | -9.58 | -9.67 to -9.51 | 1000 | 0.0 |
| d128h8c8192 | op | 8192 | 132.22 us | 112.46 us | +14.95 | 14.82 to 15.09 | 1250 | 0.0 |
| d128h8c8193 | op | 8193 | 132.3 us | 143.5 us | -8.46 | -8.52 to -8.38 | 2000 | 4.76837158203125e-07 |
| d128h8c9216 | op | 9216 | 143.23 us | 162.68 us | -13.58 | -14.09 to -13.16 | 2000 | 4.76837158203125e-07 |
| k1024 | model | 1024 | 20.218 ms | 19.88 ms | +1.68 | 1.47 to 1.89 | 128 | 0.0 |
| k2048 | model | 2048 | 20.312 ms | 20.004 ms | +1.52 | 0.96 to 1.86 | 64 | 0.0 |
| k2048q9j | model | 2048 | 44.976 ms | 45.251 ms | -0.61 | -0.71 to -0.42 | 64 | 0.125 |
| k256 | model | 256 | 25.516 ms | 25.348 ms | +0.66 | 0.25 to 1.29 | 128 | 0.0 |
| k3072q | model | 3072 | 20.264 ms | 20.007 ms | +1.27 | 1.01 to 1.54 | 64 | 0.125 |
| k4096q | model | 4096 | 20.315 ms | 20.561 ms | -1.21 | -1.6 to -0.87 | 64 | 0.0 |
| k480 | model | 480 | 19.885 ms | 19.602 ms | +1.43 | 1.29 to 1.72 | 128 | 0.0 |
| k640 | model | 640 | 20.287 ms | 19.942 ms | +1.70 | 0.51 to 3.03 | 128 | 0.0 |
| n1024m9j | model | 1024 | 44.878 ms | 44.405 ms | +1.05 | 0.9 to 1.22 | 64 | 0.0 |
| n1536m9j | model | 1536 | 45.056 ms | 44.67 ms | +0.86 | 0.62 to 1.03 | 64 | 0.078125 |
| n2048m9j | model | 2048 | 44.911 ms | 45.143 ms | -0.52 | -0.76 to -0.34 | 64 | 0.125 |
| n3584r | model | 3584 | 17.951 ms | 17.686 ms | +1.48 | 1.25 to 1.69 | 64 | 0.0 |
| n4096m9j | model | 4096 | 45.314 ms | 46.207 ms | -1.97 | -2.13 to -1.82 | 64 | 0.0 |
| n768m9j | model | 768 | 44.809 ms | 44.353 ms | +1.02 | 0.9 to 1.1 | 64 | 0.0 |
| o1024 | op | 1024 | 143.83 us | 96.1 us | +33.19 | 32.97 to 33.51 | 200 | 0.0001220703125 |
| o1024h16 | op | 1024 | 155.9 us | 105.65 us | +32.23 | 32.21 to 32.26 | 3000 | 3.0517578125e-05 |
| o1024h32 | op | 1024 | 147.93 us | 101.06 us | +31.68 | 31.35 to 31.92 | 2500 | 7.450580596923828e-09 |
| o1025h32 | op | 1025 | 147.54 us | 159.67 us | -8.22 | -8.25 to -8.18 | 2000 | 5.960464477539063e-08 |
| o12288h4 | op | 12288 | 303.71 us | 332.46 us | -9.47 | -9.51 to -9.43 | 1000 | 0.0 |
| o1536h32 | op | 1536 | 169.29 us | 195.9 us | -15.72 | -15.81 to -15.65 | 2000 | 1.1920928955078125e-07 |
| o2048 | op | 2048 | 149.04 us | 101.03 us | +32.21 | 32.06 to 32.3 | 200 | 0.0 |
| o2048h12 | op | 2048 | 155.58 us | 108.69 us | +30.14 | 30.0 to 30.29 | 2500 | 0.0 |
| o2048h16 | op | 2048 | 170.65 us | 124.73 us | +26.91 | 26.89 to 26.93 | 3000 | 3.0517578125e-05 |
| o2049h12 | op | 2049 | 156.99 us | 109.79 us | +30.06 | 29.92 to 30.2 | 2000 | 0.0 |
| o2049h16 | op | 2049 | 157.44 us | 169.32 us | -7.55 | -7.9 to -7.24 | 2000 | 1.9073486328125e-06 |
| o3072h12 | op | 3072 | 167.96 us | 191.74 us | -14.15 | -14.32 to -13.99 | 3000 | 1.9073486328125e-06 |
| o3072h16 | op | 3072 | 182.9 us | 211.73 us | -15.77 | -15.8 to -15.73 | 2000 | 1.52587890625e-05 |
| o3073 | op | 3073 | 165.26 us | 118.34 us | +28.39 | 28.34 to 28.45 | 2000 | 0.0 |
| o4096 | op | 4096 | 177.2 us | 130.68 us | +26.26 | 26.15 to 26.49 | 150 | 7.62939453125e-06 |
| o4096h12 | op | 4096 | 196.72 us | 224.02 us | -13.88 | -13.92 to -13.85 | 1500 | 0.0 |
| o4096h16 | op | 4096 | 203.81 us | 230.36 us | -13.03 | -13.06 to -13.0 | 1500 | 4.76837158203125e-07 |
| o4096h32 | op | 4096 | 318.45 us | 466.49 us | -46.49 | -46.52 to -46.46 | 1500 | 1.52587890625e-05 |
| o4097 | op | 4097 | 176.46 us | 189.04 us | -7.13 | -7.31 to -6.98 | 2000 | 1.52587890625e-05 |
| o4097h4 | op | 4097 | 166.28 us | 117.93 us | +29.08 | 28.99 to 29.16 | 2000 | 7.62939453125e-06 |
| o480 | op | 480 | 152.44 us | 99.73 us | +34.58 | 34.54 to 34.64 | 200 | 7.62939453125e-06 |
| o640 | op | 640 | 136.29 us | 88.32 us | +35.20 | 35.15 to 35.25 | 200 | 3.814697265625e-06 |
| o768 | op | 768 | 156.46 us | 103.05 us | +34.14 | 34.06 to 34.2 | 200 | 0.0 |
| o768h12 | op | 768 | 154.65 us | 103.04 us | +33.37 | 33.28 to 33.46 | 3000 | 1.9073486328125e-06 |
| o768h16 | op | 768 | 153.61 us | 102.43 us | +33.32 | 33.27 to 33.36 | 3000 | 5.960464477539063e-08 |
| o768h32 | op | 768 | 155.53 us | 108.78 us | +30.06 | 30.01 to 30.11 | 3000 | 0.0001220703125 |
| o8192h4 | op | 8192 | 208.12 us | 161.32 us | +22.48 | 22.42 to 22.54 | 1250 | 0.0 |
| o8193h4 | op | 8193 | 214.51 us | 226.57 us | -5.62 | -5.89 to -5.44 | 2000 | 0.0 |
| o900 | op | 900 | 158.22 us | 106.01 us | +33.00 | 32.94 to 33.06 | 200 | 7.450580596923828e-09 |
| o9216h4 | op | 9216 | 222.69 us | 246.37 us | -10.63 | -10.72 to -10.56 | 2000 | 0.0 |
| oc4096 | op | 4096 | 207.6 us | 161.8 us | +22.06 | 22.03 to 22.09 | 1500 | 7.62939453125e-06 |
| oc768 | op | 768 | 160.6 us | 107.49 us | +33.07 | 33.05 to 33.09 | 3000 | 0.0 |

### Engagement at the model's own attention call

| Chip | Row | Model | Q/KV heads | Keys | Stock | Variant | Gain % | CI95 | Arms max abs diff | Float64 max err stock / variant | Float64 gate |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|---|
| G13G | e2b1024 | 2B | 8/2 | 1025 | 159.75 us | 172.82 us | -8.19 | -8.61 to -7.71 | 0 | 0.01424 / 0.01424 | pass |
| G13G | e2b2048 | 2B | 8/2 | 2049 | 190.48 us | 273.28 us | -43.47 | -43.82 to -43.14 | 0 | 0.01456 / 0.01456 | pass |
| G13G | e2b640 | 2B | 8/2 | 641 | 153.2 us | 105.33 us | +31.25 | 31.0 to 31.45 | 0 | 0.01387 / 0.01387 | pass |
| G13G | e2b768 | 2B | 8/2 | 769 | 159.85 us | 112.59 us | +29.57 | 29.46 to 29.73 | 0 | 0.0155 / 0.0155 | pass |
| G13G | e2b896 | 2B | 8/2 | 897 | 161.77 us | 114.67 us | +29.12 | 29.0 to 29.22 | 0 | 0.01515 / 0.01515 | pass |
| G13G | e9b384 | 9B | 16/4 | 385 | 154.83 us | 107.42 us | +30.62 | 30.56 to 30.75 | 0 | 0.007698 / 0.007698 | pass |
| G13G | e9b512 | 9B | 16/4 | 513 | 157.74 us | 170.61 us | -8.16 | -8.33 to -7.93 | 0 | 0.006956 / 0.006956 | pass |
| G13G | e9b768 | 9B | 16/4 | 769 | 166.25 us | 189.44 us | -13.95 | -14.56 to -13.19 | 0 | 0.007295 / 0.007295 | pass |
| G13C | e2b1024 | 2B | 8/2 | 1025 | 142.47 us | 94.6 us | +33.60 | 33.48 to 33.7 | 0 | 0.01424 / 0.01424 | pass |
| G13C | e2b256 | 2B | 8/2 | 257 | 100.54 us | 100.47 us | +0.07 | -0.23 to 0.31 | 0 | 0.01169 / 0.01169 | pass |
| G13C | e2b3072 | 2B | 8/2 | 3073 | 164.37 us | 117.66 us | +28.42 | 28.19 to 28.72 | 0.000244 | 0.009219 / 0.009219 | pass |
| G13C | e2b3968 | 2B | 8/2 | 3969 | 178.71 us | 132.03 us | +26.12 | 25.84 to 26.42 | 0 | 0.01488 / 0.01488 | pass |
| G13C | e2b4096 | 2B | 8/2 | 4097 | 174.92 us | 187.27 us | -7.06 | -7.15 to -6.98 | 0 | 0.0111 / 0.0111 | pass |
| G13C | e2b480 | 2B | 8/2 | 481 | 138.02 us | 90.18 us | +34.66 | 34.55 to 34.83 | 0 | 0.01375 / 0.01375 | pass |
| G13C | e9b1536 | 9B | 16/4 | 1537 | 151.26 us | 104.39 us | +30.99 | 30.92 to 31.04 | 0.00391 | 0.007405 / 0.007405 | pass |
| G13C | e9b2048 | 9B | 16/4 | 2049 | 157.54 us | 169.8 us | -7.78 | -7.93 to -7.6 | 0 | 0.007524 / 0.007524 | pass |

### Float64 gate, op level, random bf16

| Chip | Row | Q/KV heads | Keys | Fused max err | Stock max err | Variant max err | Variant mean err | Arms max abs diff | Gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| G13G | f64h16c384 | 16/4 | 384 | 0.0007905 | 0.0007905 | 0.0007905 | 9.375e-05 | 0 | pass |
| G13G | f64h16c512 | 16/4 | 512 | 0.000809 | 0.000809 | 0.000809 | 7.927e-05 | 0 | pass |
| G13G | f64h8c1024 | 8/2 | 1024 | 0.000477 | 0.000477 | 0.000477 | 6.025e-05 | 6.1e-05 | pass |
| G13G | f64h8c768 | 8/2 | 768 | 0.0004875 | 0.0004875 | 0.0004875 | 6.362e-05 | 0 | pass |
| G13C | f64h16c1536 | 16/4 | 1536 | 0.0004216 | 0.0004216 | 0.0004216 | 4.531e-05 | 9.54e-07 | pass |
| G13C | f64h32c768 | 32/8 | 768 | 0.0004883 | 0.0004883 | 0.0004883 | 6.753e-05 | 1.19e-07 | pass |
| G13C | f64h8c1536 | 8/2 | 1536 | 0.0004762 | 0.0004762 | 0.0004762 | 4.632e-05 | 0 | pass |
| G13C | f64h8c3072 | 8/2 | 3072 | 0.0002415 | 0.0002415 | 0.0002415 | 3.31e-05 | 0 | pass |

## Rows dropped from the claims

- `k256` (G13C, 2B, +0.66 percent): the model's call there attends 257 keys, below the 320-key minimum of the split route. The engagement row `e2b256` shows the arms equal (+0.07, CI -0.23 to 0.31) and bit-identical outputs, so the variant did not run in `k256`.
- The earlier model rows below 320 keys on either chip are not claims for the same reason.

## Comparison with omlx PR 5095

jundot/omlx #5095 (merged 2026-11-12) speeds up the Metal decode attention of omlx by splitting the QK scores across the simdgroups of a threadgroup and running the online softmax with 16 lanes per row; it targets a pass 1 where QK ran on one simdgroup and the softmax on eight threads, at 64-token tiles, head width 128, 32 query and 4 key-value heads, 249K tokens, on an M5 Pro. The pass 1 here already spreads QK over all 128 threads (one thread per key) and reduces the block softmax with subgroup operations; the variant in this receipt changes only the value phase (which lanes walk which keys, and how the partial sums combine). Both changes alter the reduction order, so outputs are not bit-equal to the old kernel: #5095 reports the first divergence at a median of 177 tokens (short prompts) with a mean absolute log-probability change of 0.0039 before it; this receipt reports the largest bf16 logit difference over the timed steps (column `Logit diff`) and the float64 error of both arms in the engagement and gate tables. The two changes do not overlap in code.

## Environment

| | G13G (Apple M1, 13-inch) | G13C (Apple M1 Max, 16-inch) |
|---|---|---|
| Kernel | 7.3.0-2-20261110.1-lab-ARCH | 7.1.12-2-20261103.2-lab-ARCH |
| Mesa | 439dc032d7 | 439dc032d7 |
| mlx-omarchy wheel | 0.32.4.dev202611120448+8937c93e | 0.32.4.dev202611120448+8937c93e |
| mlx-lm | 0.32.2 | 0.32.2 |
| Gate source | `overlay/mlx/backend/omarchy/device.h`, this branch (not in the wheel above) | same |

The wheel is built from the tree of main at 358f926182 (identical tree, commit 8937c93ece), so every timing row runs the old pass 1 and the variant selected by the flag, not the gate in this branch. Wheel sha256 `7fde49d1ba4c7c72532ac69b60a42db0f4918b8552b644308cbbfb1850daf316`.

Models: Qwen3.8-2B 4-bit (8 query and 2 key-value heads, 24 layers, 6 full-attention) and Qwen3.8-9B-Distill 4-bit (16 query and 4 key-value heads, 32 layers). On the 16-inch the stage copies hash to `c202656db3c08107` (2B) and `c05a1b77c4e175eb` (9B): the first 16 hex digits of the sha256 of `sha256sum *.safetensors config.json` run in the model directory. The 2B hash on the 13-inch was not recomputed.

## Commands

Op row: `SDPA_CTX=<keys> SDPA_NTOK=<reps> timeout 280 python scripts/<probe>.py` (reps are total, half per arm); width 128 rows: `SDPA_QH=<q> SDPA_KVH=<kv> SDPA_HD=128 python scripts/sdpa-op-probe-gen.py`. Model row: `SDPA_CTX=<keys> SDPA_NTOK=128 SDPA_MODEL_DIR=<dir> python scripts/sdpa-hd256-probe-chunk.py`; the quiet rows ran with no neural engine work. Engagement row: `SDPA_CTX=<keys> SDPA_NTOK=16 SDPA_ENGAGE_ONLY=1 SDPA_MODEL_DIR=<dir> python scripts/sdpa-hd256-probe-engage.py`. Float64 row: `SDPA_CTX=<keys> SDPA_QH=<q> SDPA_KVH=<kv> SDPA_HD=256 python scripts/sdpa-f64-check.py`. Every run went through the lab's single GPU queue with a kernel, boot id and file-hash preflight.

Gate unit test: `scripts/gate-harness.sh` compiles the real `device.h` against the gate test body of `overlay/tests/omarchy/test_sdpa_split.cpp` (g++ and the Vulkan headers, no GPU); `scripts/gate-mutations.sh` runs it on the old header, the new header and eight mutations, and its output is `logs/gate-harness.txt`. Result: against the old header the body does not compile (the signature gained `kv_heads`); against the new header 45 checks, 0 failures; each of the eight mutations (limits 17, 66 and 63, block count rounded down, 64-key blocks, the zero guard removed, a 32-bit workgroup count, key-value heads ignored) fails at least two checks. The float64 `check_shape` cases (8/2 at 500, 1536 and 3072 keys, 16/4 at 1536 and 777, 32/8 at 768, and the strided, offset and one-hot cases) are in `test_sdpa_split.cpp`; they need the built library and a GPU and were not run for this receipt. The same shapes ran as the op-level float64 rows above.

## Not in this receipt

- The gate as built into a library and run on the two chips: the timing rows force the flag per arm (`0` and `1`), so they measure the two arms the gate chooses between, not the gate itself.
- A backend dispatch trace. The engagement rows show which arm ran by timing and by the output of the model's own call; they do not read the dispatch.
- Thermal logs, Vulkan device reopen after a run, and Vulkan firmware identity.
- Decode rows at 3585 to 4096 keys for the 2B on G13C and at 1025 to 2048 keys for the 9B on G13C: the limit there rests on the op and real-call rows.
- Rows from earlier sessions in the same scratch directories (names a*, d1, m*) are left out: their protocol is not recorded.
- G14C: the width-256 variant has no default there; the same probes on that chip have not produced an engaged run yet.
