# RTX 5060 Ti (NVIDIA, discrete Vulkan) rope+norm D=128 Apple-tree arm — isolated arm measured neutral (DEC-18)

Date: 2026-11-28
Device: NVIDIA GeForce RTX 5060 Ti (discrete, `VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`; vendor 4318, device 11524), NVIDIA proprietary driver 615.78.08, Vulkan ICD `/usr/share/vulkan/icd.d/nvidia_icd.json` selected via `VK_DRIVER_FILES`/`VK_ICD_FILENAMES`/`MLX_OMARCHY_FOREIGN_ICD` + `MLX_OMARCHY_ALLOW_NON_APPLE=1`. Native `subgroup_size == 32`, so no `requiredSubgroupSize` pin is involved in this run.
Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`rev 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`)
Runner contract: one wheel, `MLX_OMARCHY_NORM_APPLE_128` env selects the arm; `MLX_OMARCHY_DEFER_TOKENS=0`, `MLX_DISABLE_COMPILE=1`, `HF_HUB_OFFLINE=1`, greedy decode-128, decode-prompt ids to ctx 512 / 2048, 5 timed reps per leg after 1 discarded warmup rep.

## Why

DEC-18 replaces the Apple-only norm/QMV part gates with capability-driven
dispatch on discrete GPUs. On a native subgroup-32 part the only
behavioral delta of the change is `norm_apple_128_default_part` admitting
`DISCRETE_GPU` — the fused rope+norm D=128 call then takes
`FastRopeNormAppleBF16W128` (the 128-wide subgroup-tree build) instead of
the shared tree. `MLX_OMARCHY_NORM_APPLE_128=1/0` forces that exact part
rule on/off on any chip, so one wheel isolates the arm — same
methodology as `receipts/2026-11-28-g13g-resnorm2560-isolated.md`.
Every other norm shape (256/2560/4096) and the ResNorm folds were
already capability-gated and ran on discrete parts before this change.

## Pins

- Host boot `4b0c898e-0d61-4e8b-bc51-91a035f36492` (jw16-linux), kernel `7.1.12-2-20261103.2-lab-ARCH`, aarch64.
- Wheel `mlx_omarchy 0.32.4.dev202611261500+e1f6a24` in `/usr/lib/omarchy-mlx/venv`; `mlx_provenance.py` reports RECORD `match` for `mlx.core` and `libmlx.so`. The wheel predates this change; `=1` reproduces the exact dispatch the change defaults to.
- Harness: `~/scratch/w70-dec18-jw16/p45_cell_4b.py` (w8j cell: Qwen3-4B-4bit decode-128, boot-nonce checked) driven by `run_dec18_abba*.py`, `ABBAABBAAB` order; ctx2048 ran one warm leg (excluded) first.
- Concurrent load: another lane's `llama-server` + evals runner held the B60 and one host core throughout; legs 3–5 of each batch show the contention. Medians per arm use all 5 legs; steady-state legs are also quoted.

## Measured (leg median tok/s, decode-128)

### ctx=512

| Arm | Legs (tok/s) | Median |
|---|---|---:|
| A `NORM_APPLE_128=0` | 86.63, 19.97, 72.12, 98.65, 99.99 | **86.63** |
| B `NORM_APPLE_128=1` | 93.44, 75.55, 54.35, 97.82, 100.39 | **93.44** |

Nearest-pair wins: B 3/5. Steady-state legs 7–10: A 98.65/99.99,
B 97.82/100.39 — arms equal to ~0.5%. All 10 legs produced the
identical 128-token greedy digest `efb521ae04afaad6f2309bc14d21462c72f0a5c836832c1a18d4ddb439e7c7f9` (1 unique digest per
leg, same across arms).

### ctx=2048

| Arm | Legs (tok/s) | Median |
|---|---|---:|
| A `NORM_APPLE_128=0` | 73.01, 53.60, 76.70, 72.83, 72.94 | **72.94** |
| B `NORM_APPLE_128=1` | 67.73, 42.66, 76.14, 76.87, 73.56 | **73.56** |

Nearest-pair wins: B 2/5. Steady-state legs 6–10: A 76.70/72.83/72.94,
B 76.14/76.87/73.56 — arms equal to ~1%. All 11 legs produced the
identical greedy digest `542524d1f692f392f03221d79b8f8cc4345b3247072bf49e7572084bca2118e8`.

## Verdict

Neutral. The 128-wide subgroup-tree rope+norm neither wins nor loses on
the 5060 Ti inside this noise floor (±10–15% leg spread from the
concurrent B60 serving load; the best legs of both arms agree to ~1%).
The discrete admission in `norm_apple_128_default_part` lands: the gate
becomes capability-driven and the measured arm shows no regression —
the same basis the 2560/4096 norms went ungated on 2026-11-25.

The remaining DEC-18 surface — `requiredSubgroupSize = 32` pins on
subgroup-64 parts (RADV wave64) — is fail-closed and untestable here:
no host in the fleet reports native subgroup 64. `create_pipeline`
throws rather than compiling a wrong-lane kernel when the pin is not
admitted.
