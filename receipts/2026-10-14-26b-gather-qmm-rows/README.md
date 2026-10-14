# gemma-4-26B-A4B decode with several `gather_qmm` outputs per workgroup, 14-inch M2 Max

This is the source for the 26B decode figure in [docs/compatibility.md](../../docs/compatibility.md). Decode went from 19.738 to 23.600 tokens per second, a ratio of 1.1957 (plus 19.57 percent), with the same output digest in all eight rows. Prefill did not move (ratio 0.9985).

P is main plus one commit that runs 8 outputs per workgroup in the direct `gather_qmm` decode kernel. O is main. Both ran in one block on the same boot, interleaved O P O P O P O P.

## Source and wheels

| Arm | Source | Wheel | `mlx/lib/libmlx.so` sha256 |
|---|---|---|---|
| O | main `a2853d561722fab517164d50623724a8b24c6823` | `mlx 0.32.4.dev202610141540+a2853d56`, sha256 `1e76d34fdd7095b005827c55d152679a4c670ed09bd05b3400da2183abeb73e2` | `f5302719078a43051abd199bdd7c41de7203435469adec3135f3cf356b18321c` |
| P | `171aeb063291982b9137bb1c59dace3556d7000c`, one commit on `a2853d56`, 5 files, +139 -4 | `mlx 0.32.4.dev202610141546+171aeb06`, sha256 `931fb4ec91a592b6f6c08a5731cedd2a061b2bd371822b955f08c5dd44b3c15f` | `b7b0186e2f3449ed0f0e0bc7b9e771ecff2ad5407e31433aa61cd6d2a434b5f9` |

The core extension `mlx/core.cpython-314-aarch64-linux-gnu.so` is the same file in both wheels, sha256 `7542a0bcff502c00f2c48a2395367e9d49e3405b13f59b17f54abb78b9a0444d`. Both library files match their wheel RECORD (`provrec-control.json`, `provrec-candidate.json`). The two wheels have different stamps. Both ran with `mlx-lm` 0.32.0, the same directory for both arms and no `mlx_lm` change. The row JSON files name the wheel stamp of each arm.

The default is on. `MLX_OMARCHY_GATHER_QMM_ROWS` unset runs 8 outputs per workgroup. The block set no variable on either arm.

## Kernel, driver and device

| Item | Value |
|---|---|
| Chip | 14-inch M2 Max (G14C), Vulkan device `Apple M2 Max (G14C B1)`, integrated GPU |
| Kernel | `7.1.12-2-20261013.18-sep-ARCH`, stock, boot `5bdfa1cd`, the first block after a graceful reboot |
| Driver | Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362 |
| Driver library | sha256 prefix `40282811c6e1` (the package record is in `provrec-candidate.json`) |
| GPU kernel driver | version 0.2.0, RTKit protocol 12 |
| Governor | `performance` |
| ANE | filler on, busy 97.0 to 100.0 percent over each row. No ANE number is claimed. |

## Model and quantization

Model `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`. Quantization is affine, group size 64, 4 bits, with 8-bit router projections on layers 0 to 29.

| File | Hash |
|---|---|
| `config.json` | git blob `9e6233a7e7174ab214c49dca84f2828a8fb504ef` |
| `model-00001-of-00003.safetensors` | sha256 `801b67693463d31f36ea860494922b09adc9db920884a59b97d0d6e12db1ad25` |
| `model-00002-of-00003.safetensors` | sha256 `5ae0d2a6bb02c4d99accbe4fdc5880280dcac9d7b0ec841bc80c40294b0ce717` |
| `model-00003-of-00003.safetensors` | sha256 `8936e717c2cc88bfc13e83eb556a4c60d5939a3b2fdc3c9b258fb9572e1daa58` |
| `model.safetensors.index.json` | git blob `f6f9e61c33077cc7e2ef701eff7056927a4a65e6` |

## Command

One ticket, run on 2026-10-14 from 17:32:00Z to 17:51:19Z. Paths below show `$HOME` for the home directory.

```
$HOME/scratch/w71/score/venv/bin/python $HOME/scratch/w71/score/provrec.py $HOME/scratch/w71/score/gpro mlx-community/gemma-4-26b-a4b-it-4bit > pair26b-m2-gpro-for-gqc-main-b5bdfa1cd.prov.json && DL_GB=6 TRACE_ARMS='' BLK_ARMS='O P' bash $HOME/scratch/w71/score/runprov.sh on models.tsv $HOME/scratch/w71/score/venv/bin/python gemma4-26b-a4b mlx-community/gemma-4-26b-a4b-it-4bit 4 gpro gqc
```

Arm O runs the tree `gpro` (the O wheel unpacked) and arm P runs the tree `gqc` (the P wheel unpacked). The harness scripts are not in this repository. Their hashes: `runprov.sh` `552b326909635e3f32e95ffe311a0740050815336a4d63a0c0ed2db4f69c33e4`, `runwhl.sh` `6bed2895fbfae97e04b4af04446dc959687621ff95ab636c86a677a3e4a0a447`, `runsb.sh` `8f5e16983824c2686ac84f0472709e949b184395e0aa2e6844607a09fbfcad9d`, `provrec.py` `cbf00a706687bf6877099476ce1329e0ff1c342398d7af6f46e95fa59dae6e94`.

## Result

Rows in the order they ran. Each row reports the median of 10 timed decode samples and of 5 prefill samples, from the row's JSON file. [rows.tsv](rows.tsv) and the JSON files in [rows-26b](rows-26b) hold the same values. Every row has the same decode digest `276e9fba48abdee348ea4d20ed9f75dca4fcaf81359669d8775e66d2f3e93680` and the same corpus sha256 `9299a3b2fc136a4c0c355f9ad3e211be81823a5fb9eeadbc2e6608fb3e5718a0`.

| Row | Decode tok/s | Prefill tok/s |
|---|---|---|
| O1 | 19.751 | 228.302 |
| P1 | 23.606 | 228.108 |
| O2 | 19.734 | 227.662 |
| P2 | 23.587 | 227.799 |
| O3 | 19.729 | 228.405 |
| P3 | 23.594 | 227.986 |
| O4 | 19.742 | 228.157 |
| P4 | 23.631 | 227.547 |

The aggregate is the median of the four row medians. Decode: O 19.738, P 23.600, P over O 1.19566. The four pair ratios are 1.19518, 1.19525, 1.19590 and 1.19699. The lowest P row, 23.587, is 8.6 percent above 1.10 times the O median (21.7118). Prefill: O 228.2295, P 227.8925, a ratio of 0.99852 (minus 0.148 percent). The decode step went from 50.66 to 42.37 ms. Each arm has 40 decode samples and 20 prefill samples.

The decode digest is the same in all eight rows, so P and O produced the same tokens.

## Control run on Qwen3.5-9B

The same two trees ran Qwen3.5-9B in the next block on the same boot (guard job 17:51:07Z to 18:07:41Z), four rounds interleaved O P O P O P O P. Qwen3.5-9B decode does not run `gather_qmm`, so this block can only show that nothing else moved.

| Row | Decode tok/s | Prefill tok/s |
|---|---|---|
| O1 | 40.560 | 455.312 |
| P1 | 40.538 | 455.297 |
| O2 | 40.483 | 454.065 |
| P2 | 40.544 | 455.302 |
| O3 | 40.568 | 454.089 |
| P3 | 40.602 | 454.978 |
| O4 | 40.558 | 454.164 |
| P4 | 40.510 | 454.992 |

The decode digest is `80274aa79042646846b361a205d5f41b553f8343a4e54019e7effc2f70452f8b` in all eight rows. Decode: O 40.559, P 40.541, ratio 0.99956, pair ratios 0.99946, 1.00151, 1.00084 and 0.99882. Prefill: O 454.1265, P 455.1445, ratio 1.0022. The 9B model has no 8-bit projection and no routed experts, so decode and prefill did not change.

The JSON files of the 9B control rows are in [rows-9b](rows-9b).

## Numerical contract

Each output keeps the one-output kernel's arithmetic: the lane partials over k = lane + 256 j in ascending order, the same product `x * (scale * q + bias)`, the same subgroup add and the same sequential sum of the eight subgroup totals. A device check on the same wheel compared `MLX_OMARCHY_GATHER_QMM_ROWS` 0, 4 and 8 over the gemma-4-26B-A4B and Qwen3-30B-A3B expert shapes, bf16 and f16: 320 output comparisons, 0 differing bits. Provenance verified the wheel first. The block above is the model-level check: the same digest in all eight rows.

## Dispatch trace

The device check traced the dispatches of that synthetic run. Per dtype and setting, the one-output kernel ran 24 times, the 4-output kernel 20 times and the 8-output kernel 16 times, the counts the shapes require (N = 100 falls back to the one-output kernel at 8 outputs). The per-layer chain of two gate and up calls and one down call took 982 microseconds at 0, 636 at 4 and 660 at 8, with weights from random numbers and the filler on. Those numbers are not model timings.

## Timing and thermal procedure

Each row is a fresh process. It waits 15 seconds, then up to 60 seconds for the one-minute load average to drop below 1, then runs the timed bench. Seven rows waited the full period and O3 started after 21 seconds because its load was 0.95. The ANE busy share over each row, read from the accelerator busy-time counter, was 98.1, 97.9, 99.0, 99.0, 97.0, 97.9, 100.0 and 98.0 percent in the order O1, P1, O2, P2, O3, P3, O4, P4. The guard restarted one filler job between O2 and P2. The governor read `performance` before the block.

## Device reopen

After the block `vulkaninfo` returned 0 and a fresh MLX process opened `Device(gpu, 0)` (`reopen.out`).

## What this does not cover

One boot, one prompt set, four rounds, and no run of the exact P wheel with the variable set to 0. The 35B model with 256 experts was not run. The 4-output setting was measured only in the synthetic check above. Other models and other chips were not run.
