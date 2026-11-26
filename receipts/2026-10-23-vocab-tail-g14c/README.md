# Chunked vocabulary-tail kernels as the default on G14C (2026-10-23)

Qwen3.5-9B 4-bit decodes 3.16 to 3.85 percent faster on the Apple M2 Max (G14C) with the chunked ArgMax and LogSumExp kernels on: 43.2 against 41.6 tokens per second on one boot and 42.4 against 41.1 on another, median of 5 interleaved rounds each, token digest `9a142db605d5939d` in every arm. The kernels exist since PR 223 behind `MLX_OMARCHY_ARGREDUCE_CHUNKED` and `MLX_OMARCHY_LSE_CHUNKED`. This change makes them the default on parts whose device name contains G14C. Every other part keeps the single-workgroup kernels, and a set flag still wins everywhere.

The rule is the pure function `vocab_tail_chunked_default(device_name)` in `device.h`. `chunked_flag_enabled(name, default_on)` in `primitives.cpp` returns the default when the flag is unset, true for `1` and false for any other value.

## Machines

14-inch M2 Max, Vulkan device Apple M2 Max (G14C), Mesa 26.3.0-devel (git-f144c7949b), performance governor. Firmware identity read from the device tree (vendor prefix of the property names dropped): model j414c, chip t6021, iboot1 mBoot-20457.1.29, iboot2 iBoot-8422.141.2, os-firmware 13.5, system-firmware 27.0.

16-inch M1 Max (G13C, T6001) and 13-inch M1 (G13G, T8103) with the same Mesa build.

Kernel and settle value differ between the runs and are named next to every number. "Stock" is 7.1.12-2-20261013.18-sep-ARCH, "egpu" is 7.1.12-2-20261013.18-egpu-ARCH. The speed runs on the M2 and the census used `call_settle_us` 100 (the census ran on the boot of the egpu 9B speed run, whose header prints 100). The numerics capture and the doctest ran at 20. A number is only compared with numbers from the same run.

## Stack

The ledger wheel: mlx_omarchy 0.32.4.dev202610221002+94f5ac8 (main 94f5ac8), libmlx prefix `60c698ba4827a2a4`, driver `vulkan-f144c7949b`, mlx-lm series main. The speed arms are environment switches on this one wheel, so "on" is the code path this change makes the default on G14C. The change itself adds no other kernel. The doctest ran on a separate build of the PR commit 3acfb9f2 (below).

Models, mlx-community 4-bit snapshots: Qwen3.5-9B (index sha256 prefix `dd023913fb87cfda`), Qwen3-4B (`f7825defe5865d17`), Qwen3.8-2B (`0f3e25e8b0d783a6`). The Qwen3.5-9B and Qwen3.8-2B logits rows have 248320 columns, so both routes engage with 61 chunks.

## Speed (speed/*/arms.tee, summarised by speed/summarize-arms.py)

The step is the ledger LLM step (`scripts/llmstep.py`: 477 prompt tokens, 128 generated, greedy, one process per sample). Five rounds. Every arm runs once per round and the arm order rotates by one place each round. The change is the median of the per-round paired changes against `base`. Decode prints at 0.1 tokens per second, so one step is 0.2 percent at 41 tokens per second.

Qwen3.5-9B, Apple M2 Max, stock kernel (`speed/m2-stock-9b/`): base 41.6 tok/s. `argc` +1.68%, rounds +1.68 +1.68 +1.69 +1.20 +1.45. `lsec` +2.16%, rounds +2.16 +1.92 +2.17 +1.68 +2.17. Both: 43.2 tok/s, +3.85%, rounds +3.85 +3.85 +3.86 +3.37 +4.10. The effects add. All 20 digests equal `9a142db605d5939d`.

Qwen3.5-9B, Apple M2 Max, egpu kernel, a second boot (`speed/m2-egpu-9b/`): base 41.1 tok/s. `argc` +1.46%, `lsec` +1.95%. Both: 42.4 tok/s, +3.16%, rounds +3.16 +3.41 +3.16 +4.16 +3.16. All digests equal.

Qwen3-4B (dense attention), Apple M2 Max, stock kernel (`speed/m2-stock-4b/`): base 54.7 tok/s. Both +2.74%, rounds +1.64 +2.93 +2.74 +2.94 +2.55. This is below the 3 percent bar. The rule still enables it, because no G14C run showed a loss beyond the print resolution.

Qwen3.8-2B, Apple M2 Max, stock kernel (`speed/m2-stock-2b/`): base 105.7 tok/s. `argc` +0.09%, `lsec` -0.19%, both 0.0%, rounds +0.09 0.0 0.0 -0.19 +0.09. Neutral. The prefill column of every sample stays between 1675 and 1701 tokens per second, so no sample was disturbed.

An earlier 2B run on the egpu kernel (`speed/m2-egpu-2b-disturbed/`) was noisy: single arms dipped by up to 17 percent in rounds whose prefill column also fell, and the both arm read +0.19%. It is kept as the record of the disturbed run and carries no claim.

Parts where the default stays off:

- Apple M1 Max (G13C), kernel 7.1.12-2-12.6-sep-ARCH (an older stock build), Qwen3-4B base 51.7 tok/s: both +0.58%, rounds +0.78 +0.78 +0.58 0.0 +0.39 (`speed/m1max-4b/`). Qwen3.8-2B base 102.2: both +0.10% (`speed/m1max-2b/`).
- Apple M1 (G13G), stock kernel, Qwen3-4B base 16.8 tok/s: both +0.6%, rounds +0.6 +0.6 0.0 0.0 +0.6, one step of the print (`speed/m1-4b/`).

## Launches (census/)

`scripts/census-tail.sh` counts the traced dispatches of the ledger greedy step (mlx-lm `stream_generate`, no sampler) for 8 and 12 generated tokens in separate processes and divides the difference by 4. Qwen3.5-9B on the M2: base 305 launches per decode token. `argc` 306: kernel 66 (`ArgReduceBF16`) -1, kernel 627 (`ArgReduceChunkBF16`) +1, kernel 628 (`ArgReduceCombine`) +1. `lsec` 306: kernel 38 (`LogSumExpBF16`) -1, 631 (`LogSumExpChunkBF16`) +1, 634 (`LogSumExpCombineBF16`) +1. Both 307. The kernel names come from the `ComputeKernel` enum order in `compute.h` at 94f5ac8. Both routes run in the decode step, on bf16 rows.

## Numerics (numerics/)

`scripts/tfref-tail.py` runs the model one token at a time on four text segments of 160 tokens (636 scored steps) with teacher forcing. At every step it also reduces the model-dtype logits row (bf16) and takes the ArgMax index, the ArgMin index and the LogSumExp, and the same row in float64 on the host with numpy (the independent reference of check 1 in `docs/numerics-gate.md`, per op, on real captured operands). Five arms, each in its own process: `base`, `argc`, `lsec`, `both`, `base2` (a repeat of base). `scripts/tailchk.py` and `scripts/tfcmp.py` compare them.

Qwen3.5-9B on the Apple M2 Max (G14C, egpu kernel, `numerics/m2-9b/`): the ArgMax and ArgMin index of every arm equals the base index and the numpy float64 index at 636 of 636 steps, for the float32 log-probability row and for the bf16 row. Top-1 agreement 636 of 636 in every arm. The bf16 LogSumExp is bit-equal to base at 636 of 636 steps. The float32 LogSumExp error against float64: base mean 1.053e-06, max 8.030e-06. With the chunked route: mean 5.645e-07, max 2.073e-06. The next-token log-probability differs from base by at most 8e-06 and equals base bit for bit at 481 of 636 steps. Mean negative log-likelihood +0.0001 percent. `argc` equals base bit for bit at all 636 steps.

Qwen3.8-2B on the Apple M1 Max (G13C, egpu kernel, `numerics/m1max-2b/`) and the Apple M1 (G13G, stock kernel, `numerics/m1-2b/`): the same checks pass with the same figures on both: indices equal 636 of 636 in every arm, bf16 LogSumExp bit-equal 636 of 636, float32 LogSumExp error base mean 2.268e-06 max 6.786e-06 against 5.097e-07 and 1.775e-06 with the chunked route, bit-equal to base at 206 of 636 steps. The two chips print the same figures on these captures, and the capture files of the Apple M1 Max (G13C) and the Apple M1 (G13G) are byte-identical for every arm and segment (`sha256sum numerics/m1max-2b/captures/*/* numerics/m1-2b/captures/*/*` pairs them). The run headers do not print the device. The M1 Max ran on the 16-inch machine and the M1 on the 13-inch machine, each with the Apple GPU through the honeykrisp driver.

The bf16 LogSumExp error against float64 (`scripts/tailhist.py`, `numerics/*/tailhist.out`) is at most half a bf16 step of the result plus 1e-5 at 636 of 636 steps in every arm on all three captures (the step is 0.0625 for a value from 8 to 16, 0.125 from 16 to 32 and 0.25 above 32). The mean error is 0.19 half-steps on the 9B and 0.42 on the 2B, equal in every arm. So the bf16 outputs differ from the correctly rounded value only by the output rounding, and the arms are indistinguishable there. A full float32 CPU forward of the model is out of scope here. The change touches only the final reduction, and the per-op check above bounds its error against float64.

## Doctest (run-logs/)

`doctest-vocab-tail-run.out`: `omarchy_vocab_tail_tests` built at 3acfb9f2 passes 6 of 6 test cases and 6952 of 6952 assertions on the Apple M2 Max, no case skipped, exit code 0. The cases check the chunked ArgMax and ArgMin index against the single-workgroup kernel and a host reference for float32, bf16 and float16 at widths 8192 to 1048576 with ties, NaN, infinities and signed zeros, the LogSumExp error against float64, the route gates (rows of 8191 and 1048577 columns and integer rows keep one dispatch), and the two new cases: the part rule, and the launch count with flags unset (2 on G14C, 1 elsewhere), `0` (1) and `1` (2). `doctest-vocab-tail-build.out` is the build. Each run log ends with a repeat of its last lines, which the queue wrapper appends after the job. `scripts/m2-vtail-doctest.sh` in this directory differs from the script that ran only in the work-directory name and the `git fetch` line. That line names the pull request head here (`pull/269/head`) in place of the branch name of the run. The run is a GPU run on the M2; the default test reads the device name it runs on.

## Limits

One text and one prompt of 477 tokens, one chip per part, one boot per run, three models. The default also applies to every unmeasured model on G14C with a vocabulary row from 8192 to 1048576 columns. The Qwen3-4B gain of 2.74 percent is below 3 percent. The 2B is neutral. The M1 Max and M1 rows show no gain, and the rule leaves them off. LogSumExp bits change on G14C for every caller, within the error shown above.

## Timing and thermal procedure

Every timed run is one queue job with no other GPU or ANE job running, and the load average is read at the start. The script aborts above a load of 1.3 at its start and prints the load at every sample. No temperature reading was taken. The governor is performance. Samples run as one process each, the arms interleave, and the order rotates. The numerics and census runs are not timed and ran as normal queue jobs.

## Commands

Speed, on each machine and model: `bash scripts/swscreen2.sh OUT 5 MODELDIR tail`, with the ledger wheel (venv-main), the mlx-lm series main and driver vulkan-f144c7949b selected inside the script. Arms `base`, `argc`, `lsec`, `both` are the switch settings of the `tail` set. Census: `bash scripts/census-tail.sh OUT MODELDIR`. Numerics: `bash scripts/tail-num.sh OUT MODELDIR`, then `python3 scripts/tailhist.py ARM=DIR ...`. Doctest: `bash scripts/m2-vtail-doctest.sh build` and `run OUT`.

## Reopen

Reopen this change if a G14C run of an unmeasured model loses speed with both switches on, or if the doctest fails on a new Mesa build. The switches `MLX_OMARCHY_ARGREDUCE_CHUNKED=0` and `MLX_OMARCHY_LSE_CHUNKED=0` restore the old kernels at run time.
