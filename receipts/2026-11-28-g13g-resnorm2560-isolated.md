# G13G (Apple M1) ResNorm(2560) fold — isolated arm measured negative (DEC-31)

Date: 2026-11-28
Device: 13-inch Apple M1 (G13G, Vulkan / Honeykrisp ICD `/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json`)
Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`rev 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`)
Runner contract: one wheel, `MLX_OMARCHY_QMM_RESIDUAL_NORM` env selects the arm; `MLX_OMARCHY_DEFER_TOKENS=8`, `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`, greedy sampler, NumPy-seed-292 prompts (512 / 2048 ids), 64 greedy output ids, interior yields 17–49.

## Why

DEC-31 asked for `ResNorm(2560)` on G13G with the `#947` findings fixed.
`#947` (M1D-1c) measured the fold on a confounded arm (`QMM_GEOM=2x16` +
`RESIDUAL_NORM=0` together) and closed with the isolated ABBA still
demanded by review. This is that isolated ABBA on current defaults:
arm A = `MLX_OMARCHY_QMM_RESIDUAL_NORM=0` (the current G13G default, fold
excluded), arm B = `=1` (the route the enable would default to), on the
shipped `0.32.4.dev202611272152+d75a3d0a` wheel (v0.7.55 tag
`d75a3d0a9b`, includes the #1067 G13G defaults bundle and
`norm_apple_128_default_part` for G13G, `test_fast_ops.cpp` pin).

## Pins

- Host boot `ebea5b2b-2ae4-44ea-9fda-c0bc03a11d94`, kernel `7.3.0-2-20261110.1-lab-ARCH`, hostname `jwm1-linux`.
- Wheel `mlx_omarchy-0.32.4.dev202611272152+d75a3d0a-cp314-cp314-linux_aarch64.whl` (v0.7.55 release asset), unpacked to a private vendor dir; every leg asserts `/proc/self/maps` resolves `libmlx.so` to that copy (`DEC31_LIB`), `importlib.metadata.version("mlx-omarchy") == 0.32.4.dev202611272152+d75a3d0a`, and `mlx_lm` 0.31.3 with the `.omarchy-patches-stamp`.
- GPU turn held through `gpu-turn -m 25` for each context; legs ran in `ABBAABBAAB` order; identical `prompt_sha256` across all legs.
- Harness: `~/scratch/w70-dec31/kit/run-abba-env.sh` + `profile_decode.py` (env-arm variant of w7G's `m1d3` runner).

## Measured (interior tok/s, yields 17–49)

### ctx=512 — `abba-4b-512-ebea5b2b.S8zCYo`

| Arm | Legs (tok/s) | Median |
|---|---|---:|
| A `RESIDUAL_NORM=0` | 24.243, 24.244, 24.197, 23.996, 24.227 | **24.227** |
| B `RESIDUAL_NORM=1` | 24.155, 24.117, 23.449, 23.472, 23.916 | **23.916** |

Paired ms saved (A->B): `-0.149, -0.217, -1.318, -0.931, -0.537`;
median pair `-0.537 ms/token`, **0/5 wins**. The fold is `-1.3%`.

### ctx=2048 — `abba-4b-2048-ebea5b2b.tntTEm`

| Arm | Legs (tok/s) | Median |
|---|---|---:|
| A `RESIDUAL_NORM=0` | 22.215, 22.282, 22.283, 21.790, 22.141 | **22.215** |
| B `RESIDUAL_NORM=1` | 21.951, 22.229, 21.445, 21.334, 22.214 | **21.951** |

Paired ms saved (A->B): `-0.542, -0.108, -1.753, -0.981, +0.147`;
median pair `-0.542 ms/token`, **1/5 wins**. The fold is `-1.2%`.

Identical 64-token greedy ids across all 20 legs:
`731c110d6d63fdbb24496cb7e2ea4dfbb64d80552c03d678dc8ef0865e18a6f0`
(same digest w7G's `m1d3` table records for this model on G13G).

## Verdict

`ResNorm(2560)` stays excluded on G13G. The isolated arm loses at both
contexts on the default `2x8w2` geometry; the M1D-1c combined-arm loss
(`-2.2%`) was not an artifact of the bundled geometry arm — the fold
itself is the regression on this part. `#947`'s requested isolated ABBA
is hereby run and negative.

## Parity note (DEC-31 target)

The `23.29 tok/s` figure in the row predates the #1067/#1104 G13G
defaults bundles. On this wheel the current default arm decodes the 4B
at **24.23 tok/s ctx512** — the `23.51 tok/s` parity line is already
crossed without the fold. The `norm_apple_128_default_part` rope+norm
D=128 subgroup dispatch the row names is live on G13G since #1067
(test-pinned at `test_fast_ops.cpp:4313`).
