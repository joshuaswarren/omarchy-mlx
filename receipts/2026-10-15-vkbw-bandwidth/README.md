# vkbw: Vulkan memory read and copy bandwidth on three Apple Silicon laptops (2026-10-15)

First results of [`tools/vkbw`](../../tools/vkbw), a standalone Vulkan program that streams a buffer through a compute shader and reports GB/s per load width, lane stride, buffer size, workgroup size and subgroup size. It exists so the same test can run on any Vulkan GPU (for example an M3 under macOS and under Linux) and the tables can be compared per chip. This directory holds the numbers from the three laptops that run Omarchy Linux. It makes no claim about why any number differs.

## Machines and software

| Item | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| Vulkan device | `Apple M1 (G13G B1)` | `Apple M1 Max (G13C C0)` | `Apple M2 Max (G14C B1)` |
| Run started (UTC) | 2026-10-15 23:50 | 2026-10-15 20:57 | 2026-10-15 23:42 |
| 1-minute load average at start | 1.79 | 2.12 | 3.07 |

| Item | Value (all three) |
|---|---|
| Kernel | `7.1.12-2-20261013.18-sep-ARCH` |
| Driver | Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362, system driver (no ICD override) |
| Package line printed by the run | `mesa 1:26.2.4-1` |
| Subgroup size | 32 (reported and controllable range 32 to 32) |
| `vkbw` binary sha256 | `b7819515e6e053683f12f956eb6b5b855de168f4b899edd9b87f4c3050cdb70d` (built on each laptop from `tools/vkbw/vkbw.c` as committed in this change, `-O2`, `-lvulkan`; the 13-inch M1 ran a copy of the 14-inch M2 Max binary, same hash) |

Not recorded by these runs: the CPU and GPU governor, the clock state, the ANE state and the driver library hash. They are not quiet-machine numbers: other lab jobs ran on the same laptops in the same hour, and the 1-minute load average above is the only load figure kept. Read them as "what this program measures on this machine tonight", with the spread between the three runs as the only error bar.

## Method

Each case is one compute dispatch that reads every element of a buffer exactly once, in a batch of about 60 ms, repeated 5 times. The table cell is the median of the 5 batch medians, then the median of 3 full runs, with the lowest and highest run in brackets. GB/s is 1e9 bytes of useful traffic (read, plus written for a copy) over the wall time of the batch. Every pass sums the data and the host compares the sum with the CPU sum of the same pattern; all 3 x 3 runs have 0 failed checks (`check` column). The program timed with wall clock because GPU timestamp queries on this driver returned a time that did not cover the work (every row showed 0.000 ms per pass and millions of GB/s while the checksums were correct).

Sources: `data/*.csv` (the raw output of each run, 174 rows each) and `tools/vkbw/summarize.py`, which makes the full tables.

## Results: coalesced lanes, GB/s

Read, 512 MiB buffer:

| Load width | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| 2 B (bf16) | 9 (9-9) | 39 (38-39) | 336 (333-355) |
| 4 B (f32) | 28 (22-28) | 103 (103-104) | 341 (341-345) |
| 8 B | 32 (30-33) | 323 (322-323) | 343 (343-348) |
| 16 B | 33 (33-33) | 319 (319-319) | 341 (340-345) |

Read, 64 MiB buffer:

| Load width | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| 2 B (bf16) | 33 (32-33) | 323 (322-323) | 351 (348-356) |
| 4 B (f32) | 34 (34-34) | 325 (324-326) | 339 (337-340) |
| 8 B | 34 (34-34) | 338 (337-338) | 344 (343-347) |
| 16 B | 34 (34-34) | 330 (330-331) | 346 (346-351) |

Read, 4 MiB buffer:

| Load width | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| 2 B (bf16) | 48 (48-48) | 163 (163-164) | 179 (177-215) |
| 4 B (f32) | 60 (59-61) | 213 (212-214) | 222 (220-223) |
| 8 B | 61 (60-62) | 254 (253-254) | 242 (241-242) |
| 16 B | 61 (59-62) | 260 (260-262) | 250 (249-250) |

Copy (read plus write, counted in both directions), 512 MiB buffer:

| Load width | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| 2 B | 13 (13-13) | 32 (32-32) | 343 (342-356) |
| 4 B | 18 (18-18) | 49 (49-49) | 350 (349-354) |
| 8 B | 23 (23-23) | 98 (97-99) | 348 (347-353) |
| 16 B | 37 (36-37) | 341 (341-341) | 343 (342-347) |

## Results: lane stride and workgroup size

Read, 512 MiB, 4-byte loads, distance between neighbouring lanes (every element is still read once):

| Stride | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| 4 B (coalesced) | 28 (22-28) | 103 (103-104) | 341 (341-345) |
| 32 B | 9 (9-9) | 44 (43-44) | 44 (44-45) |
| 64 B | 5 (5-5) | 22 (22-22) | 23 (23-23) |
| 128 B | 5 (5-5) | 23 (23-23) | 28 (28-28) |
| 256 B | 10 (10-10) | 26 (25-26) | 106 (106-106) |
| 1024 B | 16 (15-16) | 72 (72-72) | 124 (106-128) |

Read, 512 MiB, coalesced 2-byte loads, workgroup size (subgroup size 32 forced and the default give the same numbers, since 32 is the only size the driver offers):

| Workgroup | 13-inch M1 | 16-inch M1 Max | 14-inch M2 Max |
|---|---|---|---|
| 32 | 14 (14-14) | 150 (138-150) | 238 (120-248) |
| 64 | 18 (18-18) | 41 (41-50) | 350 (213-355) |
| 128 | 14 (14-14) | 38 (38-38) | 335 (286-352) |
| 256 | 14 (14-14) | 38 (38-39) | 331 (288-348) |
| 512 | 14 (14-14) | 39 (39-39) | 324 (284-341) |

## What the tables show, and what they do not

- The same binary and the same driver give very different numbers for narrow loads on a large buffer: 2-byte coalesced reads of 512 MiB reach 336 GB/s on the 14-inch M2 Max, 39 GB/s on the 16-inch M1 Max and 9 GB/s on the 13-inch M1.
- On the 16-inch M1 Max the same 2-byte read reaches 323 GB/s on a 64 MiB buffer and 39 GB/s on a 512 MiB buffer. The 14-inch M2 Max does not show that change (351 and 336 GB/s). The 13-inch M1 reads 33 GB/s on 64 MiB and 9 GB/s on 512 MiB.
- Strided lanes read 5 to 124 GB/s on every machine.
- Nothing here says what causes any difference: driver, page size, caches, the compiler's code for the load or something else. The 4-byte stride-4 row shows the same split between chips as the 2-byte row does.
- Peak memory bandwidth of the chips was not measured here, and no table cell is compared with a datasheet figure.
- The 13-inch M1 has 16 GB of memory; the 512 MiB cases allocate two 512 MiB buffers.

## Receipt fields

| Field | State |
|---|---|
| Source commit | the commit that adds this receipt, `tools/vkbw` as committed (binary sha256 above) |
| Kernel and Mesa versions | in the table above |
| Vulkan device | in the table above (the CSV `#` header lines repeat device, driver, limits) |
| Firmware identity | not recorded |
| Exact command | per laptop, as one job on that laptop: `make -s vkbw`, then a pilot `./vkbw --quick --sizes-mib 4`, then `./run.sh <label>` with `REPS=3 RUN_TIMEOUT=600`, which runs `./vkbw --csv results/<label>-runN.csv` three times |
| Numerical result | the tables above and `data/*.csv`; every pass is checked against a CPU checksum, 0 failures |
| Backend dispatch trace | not applicable: the tool dispatches its own two compute shaders through the Vulkan API and does not use the MLX backend |
| Timing and thermal procedure | wall clock of 60 ms batches, 5 repeats, 3 runs; no thermal procedure and no clock or governor read-back; each laptop ran the job as the only GPU job on that host (the lab scheduler holds a per-host GPU lock), not in a quiet window, with other non-GPU lab jobs running |
| Device reopen result | each run is a new process that opens the device; the pilot and 3 runs per laptop all opened it (12 opens, 0 failures) |

A rerun in a quiet window with the clock state read back would replace the tables. Until then the tables are what they say: one binary, tonight, three runs.

## Reproduce

```
cd tools/vkbw && ./run.sh <label>      # about 1 minute per run on these laptops; needs glslangValidator, a C compiler, the Vulkan headers and loader
python3 summarize.py results/*.csv     # one column per label, ratio column when there are two
```
