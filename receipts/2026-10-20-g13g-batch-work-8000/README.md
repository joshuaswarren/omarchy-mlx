# 13-inch M1 (G13G) submission budget: 5000 to 8000

Date: 2026-10-20. Machine: 13-inch M1 (T8103), Linux kernel 7.1.12-2-20261013.18-sep-ARCH, performance governor, one boot, CPU frequency policy unchanged. This directory holds the raw run logs, the scripts that made them, and the checksums.

## Question

Release v0.7.33 changed the default per-submission work budget of the 13-inch M1 from 40000 to 5000 (commit 3b24ad416, "Calibrate submission group defaults by board (#159)"). The ledger then showed the Qwen3.8-2B decode at 41.0 tok/s against 43.9 on v0.7.32. Is there a budget that keeps submissions inside the 2 to 6 ms band of issue 19 and costs less decode speed than 5000?

## Verdict

8000. On this chip the board row `G13G` becomes 8000. The unmeasured-board fallback stays 5000, as a constant, so recalibrating this row no longer moves boards nobody measured.

| Model | Submission p50 / p95 at 5000 | at 8000 | Decode, 8000 against 5000 | Pairs |
|---|---|---|---|---|
| Qwen3.8-2B 4-bit | 1.96 / 3.01 ms | 2.97 / 5.48 ms | 42.0 against 41.0 tok/s (+2.4%) | 8000 faster in 4 of 5 pairs (pair 5 hit a load spike) |
| Qwen3-4B 4-bit | 3.24 / 5.79 ms | 4.61 / 6.72 ms | 16.8 against 16.3 tok/s (+3.1%) | 5 of 5 |

Both models stay inside the 2 to 6 ms p50 band at 8000. Output tokens are bit-identical in every arm of both models (one tokens digest per model, 5 of 5 pairs).

## Origin of the loss

`runs/s1-bw`, `s2-r32-r33`, `s3-r32bw5000`, `s4-bisect`: ledger LLM step, 5 pairs, medians.

| Comparison | Decode tok/s | Delta |
|---|---|---|
| v0.7.32 stack against v0.7.33 stack | 43.9 / 41.0 | -6.6% |
| parent of #159 against #159, same series | 43.9 / 41.2 | -6.2% |
| v0.7.33 default against `MLX_OMARCHY_BATCH_WORK=40000` | 41.1 / 43.8 | override restores +6.6% |
| v0.7.32 against v0.7.32 with `MLX_OMARCHY_BATCH_WORK=5000` | 43.9 / 40.4 | -8.0% |

The change is a deliberate trade that the 2026-10-12 calibration measured on the 4B model: 40000 gives submission p50 23.6 ms on this chip, about four times the band. The ledger had no row for this chip, so the cost showed up only after the release.

## Budget sweep

`runs/s6-*`: two runs 6 minutes apart (the first at 12:30Z, the second at 12:36Z), v0.7.33 stack, medians of 5, Qwen3.8-2B. The table shows the second run. The first run agrees within 0.1 tok/s on decode and within 1.1 tok/s on prefill, but it ran under load (load1 up to 1.27, because `tools/sweep-run.sh` compares only the integer part of load1). The second run started at load1 0.02. Only the second run feeds the table. The 40000 row is from `runs/s1-bw`.

| MLX_OMARCHY_BATCH_WORK | Decode tok/s | Prefill tok/s | TTFT ms | Decode against 40000 |
|---:|---:|---:|---:|---|
| 5000 | 41.1 | 404.3 | 1096 | -6.2% |
| 10000 | 42.7 | 414.6 | 1069 | -2.5% |
| 15000 | 43.3 | 420.0 | 1055 | -1.1% |
| 20000 | 43.6 | 421.5 | 1051 | -0.5% |
| 40000 | 43.8 | 424.2 | 1045 | reference |

Submission size per budget (`subcap/`, diagnostics wheel of 3b24ad4, `scripts/subcap_calibrate.py`, 64 decode tokens after a 512-token prefill, one process per budget, the tool's own p50 and p95 of the per-submission GPU time):

| BATCH_WORK | Qwen3.8-2B p50 / p95 ms | Qwen3-4B p50 / p95 ms |
|---:|---|---|
| 5000 | 1.96 / 3.01 | 3.24 / 5.79 |
| 8000 | 2.97 / 5.48 | 4.61 / 6.72 |
| 10000 | 3.67 / 9.66 | 6.28 / 8.36 |
| 12000 | 4.29 / 9.07 | 6.34 / 9.21 |
| 15000 | 5.30 / 12.19 | 7.51 / 10.56 |
| 20000 | 6.86 / 11.86 | 10.75 / 19.55 |
| 40000 | 12.68 / 15.44 | 20.96 / 27.14 |

The 4B model leaves the band between 8000 and 10000. That fixes 8000 as the largest tested budget that keeps both models inside it. Each row is one run.

## The 5000 against 8000 pairs

- `runs/s7-2b-1020T174654`: Qwen3.8-2B, 5 pairs, medians. Decode 41.0 against 42.0 tok/s (+2.4%), prefill 404.8 against 409.9 (+1.3%), TTFT 1094.6 against 1081.2 ms (-1.2%). One 8000 sample (pair 5) reads 26.5 tok/s at load1 1.75; the median ignores it.
- `runs/s7-4b-1020T174654`: Qwen3-4B, void. Its script `tools/bw8000-run.sh` compared only the integer part of load1, so it started at load 1.75; `tools/bw8000-4b.sh` compares the decimal value and produced the rerun below. Another job held load1 at 1.4 to 2.7 and the first three pairs read 10.2 to 11.2 tok/s. The log stays here and no figure of it is used.
- `runs/s8-4b-1020T182107`: Qwen3-4B rerun with an abort-if-loaded guard (load1 0.45 to 0.66). Decode 16.3 against 16.8 tok/s (+3.1%), prefill 169.0 against 170.7 (+1.0%), TTFT 2621.3 against 2595.6 ms (-1.0%). All 5 pairs agree.

## The PR wheel against the release wheel

`runs/s9-2b-1020T193935` and `runs/s9-4b-1020T193935`: the wheel built from this branch (default environment, no override, so the board row decides) against the v0.7.33 release wheel (default 5000). Ledger step, 5 pairs per model, medians, load1 0.25 to 0.76, one boot. Output tokens are bit-identical in every arm.

- Qwen3.8-2B: decode 42.3 against 41.1 tok/s (+2.9%), prefill 410.5 against 405.6 (+1.2%), TTFT 1079.5 against 1092.5 ms (-1.2%). The PR wheel is faster in 5 of 5 pairs.
- Qwen3-4B: decode 16.8 against 16.4 tok/s (+2.4%), prefill 170.8 against 168.9 (+1.1%), TTFT 2594.8 against 2622.7 ms (-1.1%). The PR wheel is faster in 5 of 5 pairs.

The two wheels are 45 commits apart, so this run is not a single-variable A/B and the gain cannot be assigned to the row alone. The budget-only pairs above isolate the change. This run shows that the default path of the PR wheel produces the same tokens and decodes faster by about the size those pairs predict (+2.4% and +3.1%).

## Hardware and software record

- Firmware: iBoot1 mBoot-20457.1.29, iBoot2 iBoot-8422.141.2, OS firmware 13.5, system firmware 27.0.
- Platform strings `apple,j293`, `apple,t8103`, `apple,arm-platform`.
- Kernel 7.1.12-2-20261013.18-sep-ARCH, governor `performance` before the runs, boot id replaced by `<id>` in the logs.
- Vulkan driver of the runs: the release driver directory `vulkan-e7631595df` (ICD library sha256 `03037bf663548c31`). The run logs also print `libvulkan 40282811c6e14ef9`, the sha256 prefix of the system Mesa library (package 26.2.4-1), which the runs did not load.
- Models: Qwen3.8-2B 4-bit, `model.safetensors` sha256 prefix `b0d5de688567bf4a`; Qwen3-4B 4-bit, `e240c0bdc0ebb068`.
- Stacks: release v0.7.32 (wheel 0.32.4.dev202610101057+ae5d0950), release v0.7.33 (0.32.4.dev202610160529+33630f9c), parent and commit builds at ff4d488 and 3b24ad4.
- Harness: `tools/ledgerab.sh` drives the coreglass ledger LLM step (`llmstep.py`, 477 prompt tokens, 128 generated, fresh process per sample, arms alternated). `tools/summ.py` prints the medians.

Device identity, read after the runs with the PR wheel venv (`tests/device-identity.log`): the Vulkan device name is `Apple M1 (G13G B1)`, the driver is Honeykrisp on Mesa 26.3.0-devel (git e7631595df), 8 GPU cores. That name contains `G13G`, the key of the changed table row. Two fresh processes read the same identity, so the device reopened. The identity was read on the same boot as the runs, not inside them.

Not recorded: temperature and fan readings, and a dispatch trace of the runs.

## Tests of the change

Built on the 16-inch M1 Max from branch commit 6b3597ab0 (wheel 0.32.4.dev202610201840+6b3597a, sha256 prefix `ca15b107606dbcd8`): the `omarchy_runtime_tests` binary passes all 59 cases and 27070 assertions (`tests/pr250-tests.log`, binary sha256 recorded in its first line). The two budget cases pass alone: "submission-cap default resolves per calibrated board" 9 of 9 and "work budget resolves per device name" 6 of 6. They now require 8000 for the 13-inch M1 row and 5000 for the fallback.

## Limits

One chip, one boot, 5 pairs per comparison, two models. The submission sizes come from one run per budget. Desktop smoothness was not measured on a logged-in compositor. No independent review of these numbers yet.
