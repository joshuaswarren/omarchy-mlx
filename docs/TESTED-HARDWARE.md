# Tested hardware

Back to the [README](../README.md). How to produce a row:
[ANY-GPU.md](ANY-GPU.md).

One row per real output. Empty cell = not recorded, never guessed.
Source kinds: **author run** (the testing section of a merged PR),
**collector row** (the community dataset), **benchmark** (a published
benchmark table). Author-run and benchmark rows carry test results.
Collector rows do not: the collector captures machine facts and runs
no test suite, so the collector sections are a capture inventory, not
a compatibility list.

## Non-Apple GPUs

| GPU | Driver and version | OS and kernel | MLX build | Tests passed of total | What failed | Source kind |
|---|---|---|---|---|---|---|
| Intel UHD 770 | Mesa ANV 26.2.4, Vulkan loader 1.4.363 | CachyOS, Linux 7.2.9 | source build on main `e95e86b6` + PR #111 (MLX 0.32.3, upstream `9c3d355`) | 55/55 `omarchy_runtime_tests`; op check 0 bad, 0 failed | none | author run, [PR #111](https://github.com/joshuaswarren/omarchy-mlx/pull/111) |
| NVIDIA GeForce RTX 5060 Ti | NVIDIA proprietary 615.71.09, Vulkan loader 1.4.363 | CachyOS, Linux 7.2.9 | source build on main `e95e86b6` + PR #111 (MLX 0.32.3, upstream `9c3d355`) | 54/55 `omarchy_runtime_tests`; all op checks within bf16 tolerance | independent-streams timing ratio 0.78-0.88 against the 0.95 floor (5 runs); one `SUBMIT-RECOVER ... resubmitted 1 stalled batch` line per run | author run, [PR #111](https://github.com/joshuaswarren/omarchy-mlx/pull/111) |

AMD: no result yet.

## Apple silicon capture inventory (community collector rows)

Machine captures from the community dataset snapshot of 2026-10-11,
199 records:
[SUMMARY.md](https://github.com/joshuaswarren/omarchy-mlx/blob/community-data/snapshot/SUMMARY.md)
and
[latest.jsonl](https://github.com/joshuaswarren/omarchy-mlx/blob/community-data/snapshot/latest.jsonl)
(counts below; the snapshot holds the per-row detail). Linux rows name
the SoC by its devicetree compatible string; macOS rows are native
reference captures. Driver for Linux rows: Mesa Honeykrisp where
recorded, not recorded otherwise. Tests passed of total: not collected.

| GPU / SoC | Rows | Driver and version | OS and kernel | MLX build | Tests passed of total | What failed | Source kind |
|---|---|---|---|---|---|---|---|
| apple,t6020 | 45 | Honeykrisp where recorded | Linux 7.1.x-ARCH | 0.32.2.dev to 0.32.3.dev where recorded | not collected | not collected | collector row |
| Apple M2 Pro | 14 | (native macOS) | macOS where recorded (15.7 to 27.0.1) | 0.32.1 where recorded | not collected | not collected | collector row |
| apple,t6000 | 13 | Honeykrisp where recorded | Linux 7.1.x-ARCH | 0.32.2.dev to 0.32.4.dev where recorded | not collected | not collected | collector row |
| apple,t8103 | 12 | Honeykrisp where recorded | Linux 7.1.x-ARCH | 0.32.2.dev where recorded | not collected | not collected | collector row |
| Apple M1 Ultra | 11 | (native macOS) | macOS 26.2 to 26.6.2 (Darwin 25.2 to 25.6) | 0.29.3 to 0.32.1 where recorded | not collected | not collected | collector row |
| Apple M5 Max | 10 | (native macOS) | macOS 27.0 to 27.0.1 (Darwin 27.0) | 0.32.3 where recorded | not collected | not collected | collector row |
| Apple M3 Pro | 8 | (native macOS) | macOS where recorded (15.5 to 27.0) | 0.32.1 to 0.32.3 where recorded | not collected | not collected | collector row |
| apple,t6001 | 7 | Honeykrisp where recorded | Linux 7.1.x-ARCH | 0.32.2.dev to 0.32.3.dev where recorded | not collected | not collected | collector row |
| apple,t8112 | 7 | not recorded | Linux 7.1.x-ARCH | 0.32.2.dev to 0.32.4.dev where recorded | not collected | not collected | collector row |
| Apple M4 Max | 7 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| apple,t6021 | 6 | Honeykrisp where recorded | Linux 7.1.x-ARCH, and one record with a 7.1.12 USB-C build | 0.32.2.dev to 0.32.3.dev where recorded | not collected | not collected | collector row |
| Apple M2 Max | 6 | (native macOS) | macOS where recorded (26.3 to 27.0) | where recorded | not collected | not collected | collector row |
| Apple M5 | 6 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| Apple M2 Ultra | 5 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| Apple M3 | 4 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| Apple M4 | 4 | (native macOS) | macOS where recorded | not recorded | not collected | not collected | collector row |
| Apple M5 Pro | 4 | (native macOS) | macOS 26.6.2 / 27.2 beta | 0.32.3 where recorded | not collected | not collected | collector row |
| linux,dummy-virt | 4 | no GPU captured | Linux 7.2.x aarch64-ARCH (VM) | 0.32.4.dev where recorded | not collected | not collected | collector row |
| apple,t8122 | 3 | Honeykrisp where recorded | Linux 7.1.x where recorded | not recorded | not collected | not collected | collector row |
| Apple M1 Max | 3 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| Apple M1 Pro | 2 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| Apple M1 | 2 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| Apple M3 Max | 2 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| Apple M2 | 1 | (native macOS) | macOS 26.6.2 (Darwin 25.6.0) | 0.32.1 | not collected | not collected | collector row |
| Apple M4 Pro | 1 | (native macOS) | macOS where recorded | where recorded | not collected | not collected | collector row |
| apple,t8140 | 1 | not recorded | Linux 7.1.x-ARCH | not recorded | not collected | not collected | collector row |
| apple,t6030 | 1 | not recorded | Linux 7.1.x-ARCH | not recorded | not collected | not collected | collector row |
| chip not captured | 10 | no GPU captured | Linux 6.17.2-1-pve (x86_64 hosts without Vulkan) | none | not collected | not collected | collector row |

Apple-silicon benchmark numbers live in
[BENCHMARKS.md](BENCHMARKS.md) and are outside this matrix's tests
column.

## Add a row

Ran the kit on hardware that is not here? Open an issue with the
outputs listed in [ANY-GPU.md](ANY-GPU.md), or submit with the
collector. `./scripts/run-on-your-hardware.sh` produces both in one
command. Rows are added from real outputs only.
