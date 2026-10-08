# Batched GDN prefill: row-by-row through the B = 1 route (2026-10-08)

Claim: before `patches/mlx-gated-delta-prefill-rows.patch`, a batched GDN prefill (`B > 1`, `T > 1`) ran the per-token
composed fallback in every GDN layer and a `[4, T]` prefill took 5.6-7.6x the time of four `[1, T]` prefills. With the
patch the batched prefill is at parity with four sequential ones (0.86-1.06x) and every row is bit-equal to its B = 1 call.

## Source and builds

| item | value |
|---|---|
| base commit | `e6cd97d44` (origin/main) |
| change | `patches/mlx-gated-delta-prefill-rows.patch` (PR #40) |
| baseline wheels | `mlx_omarchy 0.32.4.dev202610081002+145886c` (M1 Max pfbatch/pfbg); `...dev202610080126+qbase.02e0277` (M1 pfmods) |
| fixed wheel | `mlx_omarchy 0.32.4.dev202610081925+qgpr.390f8a4` (base `e6cd97d44` plus the patch; the stamp is the build host's rebased commit) |
| mlx-lm | the ledger series (`omarchy-mlx` patches over the mlx-lm 0.32 line) |
| driver | Honeykrisp ICD from the omacom/mesa stack at commit `6543eeb7df`, selected with `VK_DRIVER_FILES` |

## Hardware and software identity

| item | M1 (G13G) | M1 Max (G13C) |
|---|---|---|
| device tree | `apple,j293` / `apple,t8103` | `apple,j316c` / `apple,t6001` |
| kernel | `7.1.12-2-12.6-sep-ARCH` (aurora 12.6 prerelease) | same |
| Vulkan device and firmware identity | not captured in these runs (no `vulkaninfo` or firmware version was recorded) | not captured |
| ANE | not used | not used |

## Model

`qwen3_5-4bit` (Qwen3.8-2B-class, 4-bit affine, group size 64): `model.safetensors` sha256
`b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863`, `config.json` sha256
`6834d47a6e13a1ac135f27be825abfa0fa54b9e33980599a5575e2f298503a34`. Prompt: the 442-token coreglass bench prompt, repeated to length.

## Commands

All from the scripts in this directory (`*-ticket.sh` call the matching `*.py`); each ran as one `gpu-turn` ticket on an otherwise idle host.

```
PYTHONPATH=<mlx-lm series> VK_DRIVER_FILES=<icd> <venv>/bin/python pfbatch.py <model> <prompt> 128 454 1024   # twice
PYTHONPATH=<mlx-lm series> VK_DRIVER_FILES=<icd> <venv>/bin/python pfmods.py  <model> <prompt> 128            # per-module profile
PYTHONPATH=<mlx-lm series> VK_DRIVER_FILES=<icd> <venv>/bin/python pfbg.py    <model> <prompt> 454 417 363 304 # padded real-server prefill
```

`pfbatch.py`: four identical prompts, plain caches, median of 3 timed repeats after a warm-up of each form: four sequential
`[1, T]` passes against one `[4, T]` pass. `pfmods.py`: one prefill pass per row count with every sublayer timed between an
eval and a stream sync. `pfbg.py`: four prompts of different lengths (left padding, a real mask) through mlx-lm's
`BatchGenerator` (`max_tokens=1`) against the same prompts one `BatchGenerator` each.

## Results

`pfbatch` (M1 Max, G13C), `batched4_s / seq4_s`, two mirrored processes each. Rows were identical in every run (max abs diff 0.0).

| T | baseline wheel | fixed wheel | row 0 equals the sequential result |
|---|---|---|---|
| 128 | 5.55 / 5.62 (2.58 s vs 0.47 s) | 0.88 / 0.86 | baseline no, fixed yes |
| 454 | 6.69 / 6.68 (9.64 s vs 1.44 s) | 0.98 / 0.99 | baseline no, fixed yes |
| 1024 | 7.58 / 7.56 (22.0 s vs 2.91 s) | 1.05 / 1.06 | baseline no, fixed yes |

`pfmods` (M1, G13G, T = 128): all of the loss is one op.

| | baseline wheel | fixed wheel |
|---|---|---|
| `[4, T]` pass | 8.757 s (4 x B=1: 1.540 s, ratio 5.69) | 1.327 s (4 x B=1: 1.560 s, ratio 0.85) |
| `mx.fast.gated_delta_update` at B=4 | 7489 ms (four B=1 calls: 102 ms) | 0.87x of four B=1 calls |
| `linear_attn` at B=4 | 7968 ms | not above 4 x B=1 |
| MLP, quantized matmuls, conv, attention | equal to or faster than 4 x B=1 | same |

`pfbg` (M1 Max, padded batch of four different lengths, first tokens equal in every run): baseline 10.74-10.77 s batched vs
1.48-1.49 s sequential (7.19 / 7.27x); fixed 2.39-2.44 s vs 1.48-1.49 s (1.62 / 1.64x). The padded rows take the masked scan
route, which is slower than the unmasked fused kernel, so a padded batched prefill is still 1.6x of sequential. That is the
next audit step, not part of this change.

## Backend dispatch trace

Independent trace (M1, G13G, T = 128, trace on), taken by a second reviewer on the fixed wheel and the baseline wheel:

| | dispatches | GDN prefill kernel | composed-chain kernels |
|---|---|---|---|
| B = 1 | 934 | `GatedDeltaPrefillRecur32BF16` x 18 | none |
| B = 4, baseline wheel | 51,598 | none | `ElementwiseLiteF32` 16,128; `CastBF16F32` 11,742; `ReduceF32` 4,608; `CopyGeneralBF16` 13,957 |
| B = 4, fixed wheel | 1,054 | `GatedDeltaPrefillRecur32BF16` x 72 (18 layers x 4 rows) | absent; `CopyGeneralBF16` 142 -> 205 (row slices and concatenation) |

Trace-on time at B = 4 against B = 1: 7.88 s vs 0.300 s on the baseline wheel; 1.182 s vs 0.299 s (3.95x) on the fixed wheel.
The device test `batched prefill stays on the fused dispatches` (`overlay/tests/omarchy/test_gdn_decode_batch.cpp`) asserts
the B = 4, T = 128 dispatch count stays within four rows plus 16 and prints the provenance line beside the count.

## Timing and thermal procedure

One process per timed form, a warm-up pass first, median of 3 repeats, forms alternating inside a process, two mirrored
processes per cell with 3 s of idle between them. The hosts were idle (no other GPU ticket), on mains power. The M1 and M1 Max
did not throttle in these runs: the run-to-run spread of the baseline cells was under 1.5%. No thermal sensor reading was recorded.

## Device reopen and recovery

Not exercised: no device reset, reopen or reboot happened during these runs, and the change adds no device lifecycle code.
