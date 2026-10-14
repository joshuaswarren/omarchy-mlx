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
- The dispatch trace is not part of this block. I have not traced the candidate wheel.

## Integrity

The raw row logs, provenance files and checksums are kept with the lab notebook. The two wheels are identified above by sha256.
