# Several `gather_qmm` outputs per workgroup in decode, 14-inch M2 Max

This is the source for the decode figures in [docs/compatibility.md](../../docs/compatibility.md). The change runs 8 outputs per workgroup in the direct `gather_qmm` decode kernel. On the same boot, back to back, decode went from 20.3935 to 24.581 tokens per second on gemma-4-26B-A4B (plus 20.53 percent). It went from 25.066 to 28.663 on Qwen3.6-35B-A3B (plus 14.35 percent). Qwen3.5-9B did not change. Every row of all three blocks has the same output digest as its same-block main rows.

P is the head of PR 180, `e625c4e6`. PR 180 merged as main `b0799ab895bf6748666d83939da72a05910947d8` with the same tree, `8378903106d127ef745a76e5515bf1dd169444a9`. O is main with PR 179 merged. Each block ran four rounds interleaved O P O P O P O P.

## Source and wheels

| Arm | Source | Wheel | `mlx/lib/libmlx.so` sha256 |
|---|---|---|---|
| O | `7604d92c9f6459d974e7c2108c1e0412dbbab0a8`, the merge of PR 179 into the previous main; its tree equals main `9ab220f7e93483e98629e00791fdee3edf5e9066` | `mlx 0.32.4.dev202610141543+7604d92c`, sha256 `3b9ecf001de4dc49a63233d027c8b091f98c6ccef592bab0f28e21b5626eb18e` | `7c9c9fff09bbc988832797b75522156272837e0bd6df46ade96fafa58d93dc7a` |
| P | `e625c4e6bfacf9ee4e1abc15cd582fbf2721e323`, six commits on main `9ab220f7` | `mlx 0.32.4.dev202610142002+e625c4e6`, sha256 `715ce99076264c811dd4592c3688f058d2cc41679c77cd5b4138fe5bd347a8dd` | `be731051e39075a804d136d056b583f67df897d37a9863d72e1fac0b659ac77e` |

The core extension `mlx/core.cpython-314-aarch64-linux-gnu.so` is the same file in both wheels, sha256 `7542a0bcff502c00f2c48a2395367e9d49e3405b13f59b17f54abb78b9a0444d`. Both library files match their wheel RECORD (`provrec-control.json`, `provrec-candidate.json`). The wheel stamps differ. Both arms used the same `mlx_lm` directory with `mlx-lm` 0.32.0. There is no `mlx_lm` change.

The default is on. `MLX_OMARCHY_GATHER_QMM_ROWS` unset runs 8 outputs per workgroup. No block set a variable on either arm.

## Kernel, driver and device

| Item | Value |
|---|---|
| Chip | 14-inch M2 Max (G14C), Vulkan device `Apple M2 Max (G14C B1)`, integrated GPU |
| Kernel | `7.1.12-2-20261013.18-sep-ARCH`, stock, boot `a5445704` |
| Driver | Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362 |
| Driver library | package `vulkan-asahi-omarchy 1:26.2.4.git69bde15340-2`, sha256 prefix `40282811c6e1` |
| GPU kernel driver | version 0.2.0, RTKit protocol 12 |
| Governor | `performance` |
| ANE | filler on, busy 97.6 to 100.0 percent over each row. No ANE number is claimed. |

## Models and quantization

gemma-4-26B-A4B is `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`. Affine, group size 64, 4 bits, with 8-bit router projections on layers 0 to 29.

| File | Hash |
|---|---|
| `config.json` | git blob `9e6233a7e7174ab214c49dca84f2828a8fb504ef` |
| `model-00001-of-00003.safetensors` | sha256 `801b67693463d31f36ea860494922b09adc9db920884a59b97d0d6e12db1ad25` |
| `model-00002-of-00003.safetensors` | sha256 `5ae0d2a6bb02c4d99accbe4fdc5880280dcac9d7b0ec841bc80c40294b0ce717` |
| `model-00003-of-00003.safetensors` | sha256 `8936e717c2cc88bfc13e83eb556a4c60d5939a3b2fdc3c9b258fb9572e1daa58` |
| `model.safetensors.index.json` | git blob `f6f9e61c33077cc7e2ef701eff7056927a4a65e6` |

Qwen3.6-35B-A3B is `mlx-community/Qwen3.6-35B-A3B-4bit`, snapshot `38740b847e4cb78f352aba30aa41c76e08e6eb46`. Affine, group size 64, 4 bits, with 80 modules at 8-bit group 64.

| File | Hash |
|---|---|
| `config.json` | git blob `e3a2334ebf2df216742ef3ed4b784417bfffe6fc` |
| `model-00001-of-00004.safetensors` | sha256 `cc12d0240a91cb7e28543f40b09a9e79b3c5e1cfda1cb2fb197196415a0916a6` |
| `model-00002-of-00004.safetensors` | sha256 `2eb11b405baa19e2da85d06df3dc4651ed5fcadff4ae023ed4330fbc3acc92df` |
| `model-00003-of-00004.safetensors` | sha256 `236124d9dfc007c1cbd46b97ed8e90941599e784136a5cf3b4884d10aae6b849` |
| `model-00004-of-00004.safetensors` | sha256 `40e6a54bd9c762a10449605ba173e80f6661dd969df27eedd67231e64e91ea73` |
| `model.safetensors.index.json` | git blob `f714d295484e01790bad8b40c2ce49323d7d2598` |

Qwen3.5-9B is `mlx-community/Qwen3.5-9B-4bit`, snapshot `8b2b98c00a6b4d291155e4890773ca8f769aee53`. Affine, group size 64, 4 bits.

| File | Hash |
|---|---|
| `config.json` | git blob `0435ae3f4b3c8b907f7359a2f8f7150d316f3435` |
| `model-00001-of-00002.safetensors` | sha256 `3acb9f38a41e3f05344c3255f717897c11421fbb6e5f97124ee1da869f15ad95` |
| `model-00002-of-00002.safetensors` | sha256 `e6f7e2e79e5580f4610b07cccd10b7da2284d09f0d25afd20d5c795d9366d69f` |
| `model.safetensors.index.json` | git blob `d366fa524ccd8f4ed93a67c7857597dc56172b92` |

## Commands

Three tickets ran on 2026-10-14 on boot `a5445704`. The 26B block ran from 20:25:28Z to 20:45:26Z. The 9B block ran from 20:45:27Z to 21:02:06Z. The 35B block ran from 21:02:08Z to 21:21:37Z. Paths show `$HOME` for the home directory. Each ticket ran the provenance record of the O tree and then the block.

```
$HOME/scratch/w71/score/venv/bin/python $HOME/scratch/w71/score/provrec.py $HOME/scratch/w71/score/gprp mlx-community/gemma-4-26b-a4b-it-4bit > pair26b-m2-gprp-for-gqe-main-ba5445704.prov.json && DL_GB=6 TRACE_ARMS='' BLK_ARMS='O P' bash $HOME/scratch/w71/score/runprov.sh on models.tsv $HOME/scratch/w71/score/venv/bin/python gemma4-26b-a4b mlx-community/gemma-4-26b-a4b-it-4bit 4 gprp gqe
```

The 9B and 35B tickets have the same form with their own model key and repository. Arm O ran the tree `gprp`, which is the O wheel unpacked. Arm P ran the tree `gqe`, which is the P wheel unpacked. The harness scripts are not in this repository. Their hashes: `runprov.sh` `552b326909635e3f32e95ffe311a0740050815336a4d63a0c0ed2db4f69c33e4`, `runwhl.sh` `6bed2895fbfae97e04b4af04446dc959687621ff95ab636c86a677a3e4a0a447`, `provrec.py` `cbf00a706687bf6877099476ce1329e0ff1c342398d7af6f46e95fa59dae6e94`.

The ticket line ran these checks before the command. A mismatch aborts the ticket. For the 26B ticket they were:

```
uname -r | grep -q '20261013.18'
test $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) = performance
test -d $HOME/scratch/w71/score/gprp/mlx
test -d $HOME/scratch/w71/score/gqe/mlx
test -d $HOME/.cache/huggingface/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit
test $(sha256sum $HOME/scratch/w71/score/whl-gprp/mlx/lib/libmlx.so | cut -c1-16) = 7c9c9fff09bbc988
test $(sha256sum $HOME/scratch/w71/score/whl-gprp/mlx/core.cpython-314-aarch64-linux-gnu.so | cut -c1-16) = 7542a0bcff502c00
test $(sha256sum $HOME/scratch/w71/score/whl-gqe/mlx/lib/libmlx.so | cut -c1-16) = be731051e39075a8
test $(sha256sum $HOME/scratch/w71/score/whl-gqe/mlx/core.cpython-314-aarch64-linux-gnu.so | cut -c1-16) = 7542a0bcff502c00
test $(cut -c1-8 /proc/sys/kernel/random/boot_id) = a5445704
test $(sha256sum $HOME/scratch/w71/score/runprov.sh | cut -c1-16) = 552b326909635e3f
test $(sha256sum $HOME/scratch/w71/score/runwhl.sh | cut -c1-16) = 6bed2895fbfae97e
test $(sha256sum $HOME/scratch/w71/score/provrec.py | cut -c1-16) = cbf00a706687bf68
bash -n $HOME/scratch/w71/score/runprov.sh
test $(lslocks -n -o PATH | grep -cxF /var/tmp/ane-run.lock) -ge 1
test $(pgrep -fc '[a]ne-run') -ge 1
```

The 35B ticket carries the same clauses with its own model cache directory. The 9B ticket has no model cache check and is otherwise the same. The harness command by itself does not assert the wheel stamps. The checks above did. Each row JSON also records the stamp of its arm: `0.32.4.dev202610141543+7604d92c` for O and `0.32.4.dev202610142002+e625c4e6` for P, in all 24 rows of the three blocks.

## Results

Each row reports the median of 10 timed decode samples and of 5 prefill samples. The values come from the row JSON files in [rows-26b](rows-26b), [rows-9b](rows-9b) and [rows-35b](rows-35b). Each aggregate is the median of the four row medians. Each arm has 40 decode samples and 20 prefill samples.

### Results on the 26B model

Decode digest `276e9fba48abdee348ea4d20ed9f75dca4fcaf81359669d8775e66d2f3e93680` and corpus sha256 `9299a3b2fc136a4c0c355f9ad3e211be81823a5fb9eeadbc2e6608fb3e5718a0` in all eight rows.

| Row | Decode tok/s | Prefill tok/s |
|---|---|---|
| O1 | 20.424 | 229.798 |
| P1 | 24.572 | 230.063 |
| O2 | 20.400 | 229.699 |
| P2 | 24.565 | 229.900 |
| O3 | 20.362 | 229.510 |
| P3 | 24.626 | 229.261 |
| O4 | 20.387 | 229.780 |
| P4 | 24.590 | 229.688 |

Decode medians: O 20.3935, P 24.5810. P over O is 1.205335 (+20.53 percent). Pair ratios: 1.20309, 1.20417, 1.20941, 1.20616. Prefill medians: O 229.7395, P 229.7940. The ratio is 1.000237. The lowest P row, 24.565, is 9.5 percent above 1.10 times the O median (22.4329). The decode step went from 49.04 to 40.68 ms.

### Results on the 35B model

Decode digest `6666c20f80815c027a1e2857ccd5adecb025156aa1f9fb1846f76d9ad1b21540` and corpus sha256 `9299a3b2fc136a4c0c355f9ad3e211be81823a5fb9eeadbc2e6608fb3e5718a0` in all eight rows.

| Row | Decode tok/s | Prefill tok/s |
|---|---|---|
| O1 | 25.069 | 230.969 |
| P1 | 28.654 | 230.979 |
| O2 | 25.121 | 230.849 |
| P2 | 28.673 | 231.141 |
| O3 | 25.061 | 231.028 |
| P3 | 28.672 | 231.139 |
| O4 | 25.063 | 230.791 |
| P4 | 28.637 | 231.372 |

Decode medians: O 25.0660, P 28.6630. P over O is 1.143501 (+14.35 percent). Pair ratios: 1.14301, 1.14140, 1.14409, 1.14260. Prefill medians: O 230.9090, P 231.1400. The ratio is 1.001000. The 35B has 256 experts per layer and a gated-delta mixer around them. It also has a one-row shared expert gate.

### Results on the 9B control

Decode digest `80274aa79042646846b361a205d5f41b553f8343a4e54019e7effc2f70452f8b` and corpus sha256 `9299a3b2fc136a4c0c355f9ad3e211be81823a5fb9eeadbc2e6608fb3e5718a0` in all eight rows.

| Row | Decode tok/s | Prefill tok/s |
|---|---|---|
| O1 | 40.526 | 455.275 |
| P1 | 40.546 | 455.251 |
| O2 | 40.443 | 454.671 |
| P2 | 40.551 | 453.920 |
| O3 | 40.588 | 454.221 |
| P3 | 40.565 | 454.927 |
| O4 | 40.556 | 455.362 |
| P4 | 40.612 | 454.905 |

Decode medians: O 40.5410, P 40.5580. P over O is 1.000419 (+0.04 percent). Pair ratios: 1.00049, 1.00267, 0.99943, 1.00138. Prefill medians: O 454.9730, P 454.9160. The ratio is 0.999875. Qwen3.5-9B decode does not run `gather_qmm`, so this block can only show that nothing else moved.

## Numerical contract

Each output keeps the arithmetic of the one-output kernel. The lane partials run over k = lane + 256 j in ascending order. The product is the same `x * (scale * q + bias)`. The subgroup add and the sequential sum of the eight subgroup totals are the same. An earlier device check on a wheel with the same kernel compared `MLX_OMARCHY_GATHER_QMM_ROWS` 0, 4 and 8. It covered the 26B and Qwen3-30B expert shapes in bf16 and f16. It made 320 output comparisons and found 0 differing bits. The blocks above are the model-level check, with the same digest in every row of every block.

## Timing and thermal procedure

Each row is a fresh process. It waits 15 seconds. Then it waits up to 60 seconds for the one-minute load average to drop below 1. Then it runs the timed bench. In these blocks every row started its bench after the full wait of 75 or 76 seconds. The load average before each row was 1.01 to 1.57. The governor read `performance` before each block. After each block `vulkaninfo` returned 0 and a fresh MLX process opened `Device(gpu, 0)`.

## What this does not cover

These blocks took no dispatch trace. The ticket ran with `TRACE_ARMS` empty, so the blocks do not show which kernel ran. The route evidence that exists for this head is separate: a 4-token trace on the 13-inch M1 with DeepSeek-Coder-V2-Lite-Instruct-4bit showed kernel id 612 (the 8-output bf16 kernel) 465 times in 8201 dispatches, with ids 580, 581, 609, 610 and 611 at zero; the P wheel contains the four new `GatherQmmSubDirectR` kernel names and the O wheel contains none; and the earlier device check, on a wheel with the same shader file, asserted the dispatch counts of the one-output, 4-output and 8-output kernels. This receipt makes no CPU-count claim. These blocks cover one boot, one prompt set per model and four rounds. They include no run of the exact P wheel with the variable set to 0. The 4-output setting was measured only in the synthetic device check. Other models and the other chips were not run in these blocks.
