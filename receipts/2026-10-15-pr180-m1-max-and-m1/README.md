# Several `gather_qmm` outputs per workgroup in decode, 16-inch M1 Max and 13-inch M1

This receipt adds two pairs to the 14-inch M2 Max pair in [receipts/2026-10-14-gather-qmm-rows-pr180](../2026-10-14-gather-qmm-rows-pr180/README.md). The change is the same. The direct `gather_qmm` decode kernel runs 8 outputs per workgroup.

On the 16-inch M1 Max, decode went from 17.6435 to 22.7190 tokens per second on gemma-4-26B-A4B (plus 28.77 percent). On the 13-inch M1, decode went from 5.7485 to 8.6400 on DeepSeek-Coder-V2-Lite-Instruct 4-bit (plus 50.30 percent). Prefill did not change on either chip. Every row of each pair has the same output digest.

The two chips and the M2 Max are separate results. The gains differ by chip and by model, so they are not averaged.

## Arms and builds

P is the head of PR 180, `e625c4e6`. PR 180 merged as main `b0799ab895bf6748666d83939da72a05910947d8` with the same tree. O is main with PR 179 merged, built from `7604d92c`. The O tree is the one used in the M2 Max pair.

| Item | Value |
|---|---|
| P wheel | `mlx 0.32.4.dev202610142002+e625c4e6` |
| P `mlx/lib/libmlx.so` sha256 | `be731051e39075a804d136d056b583f67df897d37a9863d72e1fac0b659ac77e` |
| P `mlx/core.cpython-314-aarch64-linux-gnu.so` sha256 | `7542a0bcff502c00f2c48a2395367e9d49e3405b13f59b17f54abb78b9a0444d` |
| mlx-lm | 0.32.0 in every row |

The P library files match their wheel RECORD on both chips (`prov-26b-m1-max.json`, `prov-deepseek-m1-part1.json`, `prov-deepseek-m1-part2.json`). The provenance files record the P wheel only. Every O row records the wheel stamp `mlx-omarchy 0.32.4.dev202610141543+7604d92c`. It is the stamp of the O wheel in the M2 Max pair. The O library hash is not recorded here.

Both arms set `MLX_OMARCHY_QUEUE_PRIORITY=low`. This receipt lists no other variable and does not claim that the variable has no effect.

Rows alternate O P O P within a pair. Each row is 10 timed decode samples and 5 prefill samples of 512 tokens. The prompt corpus has sha256 `9299a3b2fc136a4c0c355f9ad3e211be81823a5fb9eeadbc2e6608fb3e5718a0`. The ANE filler was on. The load average at the end of each row was 1.14 to 1.56 on the 16-inch M1 Max. It was 1.20 to 1.43 on the 13-inch M1.

## Device, kernel and driver

| Item | 16-inch M1 Max | 13-inch M1 |
|---|---|---|
| Chip | M1 Max (G13C), Vulkan device `Apple M1 Max (G13C C0)` | M1 (G13G) |
| Kernel | `7.1.12-2-20261013.18-sep-ARCH`, boot `b02ba9f6` | the same kernel, boot `7c94b356` |
| Driver | Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362 | package build below; the Vulkan record is empty because the host has no `vulkaninfo` |
| Driver package | `vulkan-asahi-omarchy 1:26.2.4.git69bde15340-2` | the same package |
| Driver library | sha256 prefix `40282811c6e1` | the same prefix |
| GPU kernel driver | version 0.2.0, RTKit protocol 12 | the same |
| Governor | `performance` | `performance` |
| ANE | filler on, busy 99.3 to 100 percent in each row | filler on; the busy percentage is not in the row files |

## Models

| Chip | Model | Snapshot | Quantization |
|---|---|---|---|
| 16-inch M1 Max | `mlx-community/gemma-4-26b-a4b-it-4bit` | `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` | affine, group size 64, 4 bits, with 8-bit router projections |
| 13-inch M1 | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit` | `156620553457c261a1fdee60b46da888b95b8b4c` | affine, group size 64, 4 bits, model type `deepseek_v2` |

The 26B file hashes are in `prov-26b-m1-max.json` and equal the ones in the M2 Max receipt. The DeepSeek provenance lists file names only, so its file hashes are not recorded.

## Results

Each figure is the median of the four row medians. The row files are in [rows-26b-m1-max](rows-26b-m1-max) and [rows-deepseek-m1](rows-deepseek-m1). `python3 recompute.py` prints every figure below from those files.

### gemma-4-26B-A4B on the 16-inch M1 Max

Decode digest `276e9fba48abdee348ea4d20ed9f75dca4fcaf81359669d8775e66d2f3e93680` in all eight rows. It is the same digest as in the M2 Max pair.

| Row | Decode tok/s | Prefill tok/s |
|---|---|---|
| O1 | 17.602 | 177.206 |
| P1 | 22.772 | 177.036 |
| O2 | 17.662 | 177.062 |
| P2 | 22.665 | 177.006 |
| O3 | 17.625 | 176.778 |
| P3 | 22.711 | 177.027 |
| O4 | 17.665 | 176.985 |
| P4 | 22.727 | 177.093 |

Decode medians: O 17.6435, P 22.7190. P over O is 1.287670. Pair ratios: 1.29372, 1.28326, 1.28857, 1.28656. The row spread is 0.357 percent for O and 0.471 percent for P. Prefill medians: O 177.0235, P 177.0315. The ratio is 1.000045.

### DeepSeek-Coder-V2-Lite-Instruct on the 13-inch M1

Decode digest `b4341e720ba98e884635708cf78f8776749c64e21e1800c8f31e0310cb5ca8bc` in all eight rows.

The pair ran as two jobs of two rounds each on one boot. The second job started about 43 minutes after the first, and other tickets ran on the host in between.

| Row | Decode tok/s | Prefill tok/s |
|---|---|---|
| part 1 O1 | 5.766 | 59.508 |
| part 1 P1 | 8.620 | 59.428 |
| part 1 O2 | 5.728 | 59.506 |
| part 1 P2 | 8.628 | 59.415 |
| part 2 O1 | 5.741 | 59.394 |
| part 2 P1 | 8.671 | 59.548 |
| part 2 O2 | 5.756 | 59.414 |
| part 2 P2 | 8.652 | 59.508 |

Decode medians: O 5.7485, P 8.6400. P over O is 1.503001. Pair ratios: 1.49497, 1.50628, 1.51036, 1.50313. The two jobs give 1.500609 and 1.506741 on their own. The row spread is 0.661 percent for O and 0.590 percent for P. Prefill medians: O 59.460, P 59.468. The ratio is 1.000135.

## Counters

The 16-inch M1 Max boot already held 11 GPU timeouts, 11 recovery lines and 11 fault-info lines before the job started. No out-of-memory line. The count did not change during the job, so the job added none. The boot is not clean.

On the 13-inch M1 the GPU timeout, recovery, fault-info and out-of-memory counters were 0 before and after both jobs.

No row showed a queue-priority refusal line.

## Limits

- Each pair is one chip, one boot, one model and four rounds.
- The credit applies to merged main only. PR 180 is merged as `b0799ab8`, and the P tree equals it.
- The 26B and 35B do not fit the 13-inch M1, so DeepSeek is the model that ran there. The 35B has no pair on the 16-inch M1 Max.
- The harness scripts are not in this repository. The row files are copied unchanged from the lab record, and the lab receipt log is not copied because it names hosts.
- Same output digests mean the O and P arms produced the same tokens on this corpus. They are not a numerical comparison against an independent reference.
- The DeepSeek gain is not comparable with the gemma-4-26B-A4B gains on other chips.

## Files

| File | Content |
|---|---|
| `rows-26b-m1-max/` | the eight row files of the 26B pair |
| `rows-deepseek-m1/` | the eight row files of the DeepSeek pair |
| `prov-*.json` | wheel, driver and model provenance of each job |
| `analysis-*.txt` | the lab analyzer output for each pair |
| `recompute.py` | recomputes the figures above from the row files |
| `SHA256SUMS` | hashes of every file in this directory |
