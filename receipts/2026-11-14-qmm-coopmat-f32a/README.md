# qmm coopmat: f32 A matrix for the f16 x build, M4 Pro (G16S)

Base: #429 head `de0492c4` (exact-shape gate; the f16 x route asks for f16 x f32, which AGX3B does not advertise, so it falls back to the tile kernel).
Head: `15ea35ef` (the same gate; the f16 x build loads x as f16, converts it to an f32 A matrix and asks for f32 x f32).

Both wheels were built from the exact trees with `DEV_RELEASE=1 scripts/build-wheel.sh` (`wheels.sha256`) and installed into two copies of one venv
(mlx-lm 0.31.3, Python 3.14.9). Arms interleaved, one GPU job at a time, nothing set except the ICD.

| | |
|---|---|
| Machine | MacBook Pro 16-inch, M4 Pro (J616s, T6040), `Apple M4 Pro (G16S B1)` |
| Driver | aurora-silicon/mesa aurora-wip 24162c16 with aurora-silicon/mesa#319 (55c06da6) merged, `git-efbb097055` (one further local commit, an opt-in that is off unless its variable is set); `VK_KHR_cooperative_matrix` exposed by default on the M4 Pro |

## fp16 4-bit quantized matmul (`tools/qmm_f16.py`, `qmm-f16.txt`)

x f16 (M x K), w 4-bit group 64, f16 scales; median of 10 timed calls after warm-up, three runs per arm; max abs error against an f32 reference.

| M x K x N | base ms | head ms | base max err | head max err |
|---|---:|---:|---:|---:|
| 512 x 4096 x 4096 | 7.83-7.86 | 3.29-3.31 | 0.0043 | 0.0026 |
| 2048 x 4096 x 4096 | 36.18-36.49 | 12.47-12.54 | 0.0045 | 0.0026 |
| 2048 x 4096 x 12288 | 106.5-107.4 | 36.99-37.71 | 0.0046 | 0.0027 |
| 2048 x 12288 x 4096 | 110.5-111.4 | 37.67-38.84 | 0.0086 | 0.0047 |

## Models (`models.txt`, `RESULT` lines, three rounds per arm)

Greedy, up to 128 tokens; pf300 and pf2k are warm prefill of a ~300 and ~2k token prompt.

| Model | pf2k tok/s base | pf2k tok/s head | decode base | decode head | text |
|---|---:|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct-4bit (f16) | 670.1 / 673.8 / 672.9 | 1572.3 / 1588.1 / 1548.0 | 135.5 / 138.0 / 140.8 | 136.9 / 136.9 / 147.1 | `225e436fc354` in every run |
| Qwen3-8B-4bit (bf16, control) | 327.9 / 326.7 / 328.0 | 327.2 / 327.1 / 328.9 | 41.5 / 41.8 / 39.8 | 40.4 / 41.1 / 44.7 | `9ee73549c37f` in every run |

The bf16 route already used the f32 A matrix and does not change; its decode spread is run to run.
