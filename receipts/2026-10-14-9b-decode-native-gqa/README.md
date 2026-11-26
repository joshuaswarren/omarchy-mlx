# Qwen3.5-9B decode, native GQA in the gated-delta decode kernel, 14-inch M2 Max, 2026-10-14

This receipt backs the change in this commit. Qwen3.5-9B has 16 key heads and 32 value heads. Before this change each of the 24 linear-attention layers expanded q and k from 16 to 32 heads with two copy dispatches per decoded token. The composed-order tiled decode kernel now reads key head j / (Hv / Hk) itself, and the Python repeat in the mlx-lm patch is opt-in.

## Result

One matched block on one machine, one boot and one driver. Arm O is current main. Arm P is this change. The two arms alternate O P O P O P O P over 4 rounds, with the ANE filler running.

- Decode, O: 40.58, 40.64, 40.67 and 40.61 tokens per second, median 40.62.
- Decode, P: 41.13, 41.16, 41.13 and 41.17 tokens per second, median 41.14.
- Decode P over O: 1.0128. The four pair ratios are 1.0136, 1.0128, 1.0113 and 1.0138.
- Prefill, O: 454.14, 454.50, 455.31 and 455.01 tokens per second, median 454.75.
- Prefill, P: 454.74, 455.06, 455.43 and 455.34 tokens per second, median 455.20.
- Prefill P over O: 1.0010. The prefill path is the same code in both arms.
- Noise band: a block with both arms on the same tree measured 0.32 percent on decode and 0.34 percent on prefill. Every decode pair ratio is above 1 plus that band.
- Output: all 8 rows print the token digest `80274aa790426468`. That is also the digest of the control block and of the macOS run of this model.
- Per-token time falls from 24.62 ms to 24.31 ms. That is 0.31 ms for 48 removed dispatches, about 6.5 microseconds each if the whole difference comes from them. This is derived, not measured per dispatch.

The gain is 1.3 percent. It is one change of several planned for the decode path and does not reach the 10 percent goal.

## Source and arms

- Base: `ad2809a3561e9165d17a6e6a68112a4f6e3c37f8`. This change is one commit on top of it.
- Arm O wheel: `0.32.4.dev202610140109+ad2809a3`, sha256 `39e59f18ef5791c71f681d84ff005e9f220fb1e7114ebc3a1970406ed59b940d`.
- Arm P wheel: `0.32.4.dev202610140230+849a36c5`, sha256 `353195d4fd810c2e71eb692dba5db6034aae7d08693fca4d1c84c65f691f6854`.
- Arm O uses the shared patched mlx-lm. Arm P uses a copy whose `models/gated_delta.py` differs in one line, the default of `MLX_OMARCHY_GDN_RAW_REPEAT` (sha256 `8d1f9d6d99b42884e7d679259407737caf4118d91bfac0b6eb65e5b96d69d3ac`).
- Both wheels match their RECORD files.

## Machine and software

- 14-inch M2 Max, Vulkan device `Apple M2 Max (G14C B1)`, kernel `7.1.12-2-20261013.18-sep-ARCH`, CPU governor performance.
- Honeykrisp, Mesa `26.3.0-devel` at git `69bde15340`, API 1.4.362. The driver library sha256 starts `40282811c6e1`.
- GPU kernel driver 0.2.0 and RTKit protocol 12. No separate firmware build string is exposed on this stack.
- Model `mlx-community/Qwen3.5-9B-4bit`, snapshot `8b2b98c00a6b4d291155e4890773ca8f769aee53`, 4-bit affine with group size 64.
- `model-00001-of-00002.safetensors` sha256 `3acb9f38a41e3f05344c3255f717897c11421fbb6e5f97124ee1da869f15ad95`.
- `model-00002-of-00002.safetensors` sha256 `e6f7e2e79e5580f4610b07cccd10b7da2284d09f0d25afd20d5c795d9366d69f`.

## Command and procedure

- The ticket ran one script, `runprov.sh on models.tsv <python> qwen35-9b mlx-community/Qwen3.5-9B-4bit 4 <arm O tree> <arm P tree>`, with the arms set to `O P` and no trace.
- The harness scripts are not in this repository.
- Each row is one process that loads the model from the local cache and prints its decode and prefill rates and an output digest. The prompt corpus and lengths are those of the scoreboard harness.
- The ANE filler was running as in production. ANE busy time was 97.6 to 100 percent on every row.
- The load average read before each row was 1.03 to 1.13.
- GPU timeouts, recovery lines, fault-info lines and out-of-memory events were all 0 before and after. A fresh process opened the device after the block and returned 0.
- The dispatch trace is not part of this block. It was taken in a separate ticket, described below.

## Dispatch trace

A separate untimed ticket traced both arms on the same machine and kernel. It ran later, on boot `35eb9d8f`, after a module-swap reboot. Each arm ran 4 and 12 greedy tokens with `MLX_OMARCHY_TRACE_DISPATCH=1`. One decoded token is the difference divided by 8.

- Arm O dispatches 385.0 kernels per decoded token. Arm P dispatches 337.0.
- Only `CopyGeneralBF16` changes, from 56.0 to 8.0 per token. These are the 48 q/k expansion copies.
- Every other kernel class has the same count in both arms.
- The 4-token digest is `4eca14f7f3a94967` in both arms. The 12-token digest is `c57d3ee2dfeb29e2` in both arms.
- Both arms ran inside one ticket, so the boot change does not touch the comparison.

## Device test

The new doctest is `gdn raw decode f32 A_log: native GQA equals the repeated route bit for bit`. CI configures with the C++ tests off, so it does not build this test. I built `omarchy_gdn_fast_route_repeat_tests` from candidate commit `849a36c5` and ran it on the M2. The C++ files are the same at this head.

- Binary sha256 `6276fc6fba473eaeea18c18dd87fe3a76fded67fb6794ed302bd362ed0dd6bc6`.
- The run was on boot `35eb9d8f`, kernel `7.1.12-2-20261013.18-sep-ARCH`, governor performance, with the ANE filler running.
- The device is `Apple M2 Max (G14C B1)` on Honeykrisp, Mesa `26.3.0-devel` at git `69bde15340`, driver library sha256 starting `40282811c6e1`. These come from the provenance record of a ticket on the same boot.
- No driver override variable was set, so the installed driver ran.
- Default run: 3 test cases passed and 3872 assertions passed, with 0 failures.
- The new case printed 1 dispatch for the native route and 1 for the repeated route at B=1, at B=1 with a buffer offset, and at B=3. Output and final state are bit-identical.
- With `MLX_OMARCHY_GDN_GQA_DECODE=0` the same 3 cases pass. The un-repeated route then takes 3 dispatches, the kernel and the two expansion copies.
- The earlier precomputed-gates cases take 3 dispatches for the un-repeated route in both runs. This change does not touch them.

## Integrity

The raw row logs, provenance files and checksums are kept with the lab notebook. The two wheels are identified above by sha256.
