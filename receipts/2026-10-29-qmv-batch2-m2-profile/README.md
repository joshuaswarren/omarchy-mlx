# Batch 2 on a 14-inch M2 Max with the hoisted qmv kernel (issue 304): where the time goes

Result: batch 2 in every #303 table is an unpadded batch, and an unpadded batch decodes through the composed gated-delta fallback (1225 dispatches and about twice the step time of a padded batch) when mlx-lm passes an all-true mask at every decode step, as the copy in the 0.7.28 system package does; the current omarchy-mlx patch series passes none (CPU probes below). The 2 to 5 percent batch-2 loss of #303 is **not explained**: the batch kernel is not slower in the profile in any row set (the medians put the hoisted kernel 0.02 to 0.07 ms faster per step, and the three rounds overlap in three of the four sets), the unprofiled profile-driver ratio at the unpadded 2-row set is 0.995, and the decode A/B on the same prompts is 0.978 in all three pairs. The GPU run that would show whether the current series removes the slowness and the loss has not run yet (`PREREG.md`, Amendment 3, predictions written first).

Everything below ran on a 14-inch M2 Max, stock kernel 7.1.12-2-12.6-sep-ARCH, Qwen3.8-2B 4-bit, `mlx_lm` 0.31.3 from the 0.7.28 system package, two diagnostics wheels that differ only by #303 (shipped `a736744a`, hoisted `42b30a2d`), greedy decode, `MLX_DISABLE_COMPILE=1`.

## The four prompts

The four prompts are 12, 12, 13 and 13 tokens long. Batch 2 uses the first two: equal length, so `BatchGenerator` pads nothing. Batches 3 and 4 include a 13-token prompt and are left padded. Every batch-2 number in the #303 receipt is therefore an unpadded batch and every batch-3 and batch-4 number is a padded one. Row sets used here: E2 (prompts 0,1: unpadded), P2 (0,2: 12 and 13 tokens: padded), E3 (0,1,4: three prompts of 12 tokens: unpadded), P3 (0,1,2: 12, 12 and 13: padded).

## Stock-kernel profile (`stock-kernel-profile/`, `t7b-b2prof2.sh`)

Three rounds, each cell once with GPU timestamps (a barrier per dispatch, so the profiled wall time is inflated) and once unprofiled for 64 steps. Medians of the three rounds:

| set | wheel | unprofiled step, ms (3 rounds) | median | dispatches per step | GPU busy, ms (profiled) | batch kernel, ms (profiled) | `GatedDeltaDecodeBF16PfWide` per step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2 | shipped | 29.22, 29.20, 28.78 | 29.20 | 1225 | 17.69 | 2.332 | 0 |
| E2 | hoisted | 28.86, 29.65, 29.36 | 29.36 | 1225 | 17.47 | 2.308 | 0 |
| P2 | shipped | 14.33, 14.40, 14.35 | 14.35 | 628 | 9.11 | 2.438 | 18 |
| P2 | hoisted | 14.05, 14.07, 14.20 | 14.07 | 628 | 8.56 | 2.414 | 18 |
| E3 | shipped | 31.75, 31.38, 31.77 | 31.75 | 1225 | 16.73 | 2.321 | 0 |
| E3 | hoisted | 30.29, 30.39, 29.94 | 30.29 | 1225 | 17.19 | 2.297 | 0 |
| P3 | shipped | 17.23, 17.28, 17.38 | 17.28 | 628 | 9.05 | 2.641 | 18 |
| P3 | hoisted | 15.80, 15.86, 15.81 | 15.81 | 628 | 9.10 | 2.569 | 18 |

Speed ratio, shipped over hoisted, unprofiled: E2 0.995, P2 1.020, E3 1.048, P3 1.093 (`tabulate_stock_profile.py`). The three E2 rounds of the two wheels overlap (shipped 28.78 to 29.22, hoisted 28.86 to 29.65): E2 shows no loss that three rounds can separate.

- The unpadded sets run 1225 dispatches per step and take about twice the step time of the padded sets (E2 29.2 ms against P2 14.3 ms). The GPU is busy for about a quarter of the profiled window in every set; the rest is host time between dispatches.
- The 600 extra dispatches are the composed gated-delta update: in the unpadded sets `GatedDeltaDecodeBF16PfWide` (18 layers, 18 dispatches) is absent, and `ElementwiseLiteF32` (+198 per step), `CastBF16F32` (+126), `ClearU32` and `LogicalOrBool` (+54 each), `ReduceF32` and `SelectF32` (+36 each) and smaller families appear instead.
- The batch kernel (`QmmVecQ4WordSubgroupBatch4BF16`) takes 2.3 to 2.6 ms per step in every set, and the hoisted wheel is 0.02 to 0.07 ms faster per step than the shipped one in the medians of all four sets (0.024, 0.024, 0.024 and 0.072 ms; the three rounds overlap in E2, P2 and E3). No set shows the batch kernel slower in the medians.

## Same decode harness as #303, on the stock kernel (`stock-kernel-decode-ab.log`, `t7b-bgdiag.sh`)

`BatchGenerator`, rows 0 to B-1, greedy, 64 tokens, three interleaved pairs per batch size, each run its own process, ids hash identical in all runs per batch size. Step time, shipped and hoisted, ms:

| Batch | shipped, 3 runs | hoisted, 3 runs | speed ratio (medians) |
| --- | --- | --- | --- |
| 2 (unpadded) | 28.78, 28.61, 28.45 | 29.10, 29.25, 29.37 | 0.978 |
| 3 (padded) | 17.29, 17.27, 17.31 | 15.86, 15.75, 16.21 | 1.091 |
| 4 (padded) | 20.10, 19.98, 20.13 | 17.39, 17.51, 17.45 | 1.152 |

Batch 2 is 2.2 percent slower here (hoisted above shipped in all three pairs). The #303 receipt measured 5.3 to 5.6 percent on two lab kernels (kernel `7.2.11-lab`).

## What mlx-lm passes as the mask (`mask-probe/`, `mask_probe.py`, CPU)

For two prompts, `mask_probe.py` records the `mask` argument of every `gated_delta_update` call in `BatchGenerator` prefill and decode (Qwen3.8-2B, three steps), in three states of mlx-lm 0.31.3:

| mlx-lm state | equal lengths (12, 12): prefill / decode | 12 and 13 tokens: prefill / decode |
| --- | --- | --- |
| pristine | all-true (2, 11) / all-true (2, 1) | mask with false entries (2, 12) / none |
| packaged copy in the 0.7.28 system package (older `ssm-maskless`, test `not any(left_padding_list)`) | none / all-true (2, 1) | all-true (2, 12) / none |
| current patch series (`scripts/patch-mlx-lm-ssm-maskless.py`, test `all(p <= 0 ...)`) | none / none | all-true (2, 12) / none |

The probe does not explain why the padded prefill mask has false entries in the pristine tree and none in the two patched ones (the first chunk is 12 tokens and both prompts are at least 12 long); that difference was not examined. An unpadded batch never calls `finalize()` on its caches, so its recurrent caches keep a left padding that `advance()` lowers below zero at every step; a test for zeros only returns the mask after the first token. The backend takes the fused decode kernel at batch above 1 only without a mask (`GatedDeltaUpdate::eval_gpu`, `fused_ready`: `B == 1 || (T == 1 && !has_mask)`), so every gated-delta layer of an unpadded batch falls back.

## Not run, not recorded

- The GPU run with the current patch series (`t7b-nopad2.sh`). It needs the M2 on the stock kernel again. Until it runs, "the current series removes the slowness" rests on the CPU probe and the fused-kernel gate in the code.
- The raw GPU profile streams (about 7 MB each) were deleted after summarizing; the summaries, window medians and logs are kept.
- The first profile run (kernel names resolved with a parser that skipped one enum entry, so the batch kernel showed under another name) is not used anywhere.
- The 13-inch M1 and 16-inch M1 Max were not profiled. The 13-inch M1 venvs also load mlx-lm from the system package (its revision was not compared with the M2's). Its step time does not show the jump (batch 2, 3, 4: 69, 89, 116 ms) and its dispatch count is about 2300 per step at every batch size (`receipts/2026-10-29-qmv-batch-hoist/dispatch-trace-13-inch-M1.log`); I did not test why.
- Model: `Qwen3.8-2B-mlx-4Bit`, `model.safetensors` sha256 `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863`.

## Files

- `PREREG.md`: the plan and its three amendments, each written before the run it precedes.
- `stock-kernel-profile/`: per-cell profiled summaries (`*-sum.json`), unprofiled step medians (`*-u-wall.json`) and the run log.
- `stock-kernel-decode-ab.log`, `mask-probe/`: raw output.
- `mask-probe/make-mask-excerpts.txt`: `ArraysCache.make_mask` in the packaged copy and in the current patch series (the two test lines differ).
- `t7b-b2prof2.sh`, `t7b-bgdiag.sh`, `t7b-nopad2.sh`, `profile_batch_driver2.py`, `b2prof_summary.py`, `mask_probe.py`, `tabulate_stock_profile.py`: the scripts (paths come from `$HOME`; host names (hostB, hostC) and private scratch-directory names are replaced by generic ones in these copies, the logic is unchanged).
