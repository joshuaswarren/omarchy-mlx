# uclamp placement: which threads must carry the clamp for decode

The serve raises `uclamp_min` on its startup thread. This receipt measures decode with the clamp applied in four ways. It covers two machines and two models. It backs the numbers in `docs/install-omarchy.md`.

Result: on gemma-4-E4B on the 13-inch M1, any of the three clamped placements decodes 44 percent faster than no clamp. Qwen3.5-9B is flat on the 13-inch M1 and on the M2 Max.

## Arms

All arms run `scripts/bench_decode.py` through `run_clamped.py`, which sets the clamp and prints the `util_min` of every thread before and after the run.

- N: no clamp.
- T: the calling thread, after mlx and mlx_lm are imported. This is the order the serve uses.
- B: the calling thread, before anything starts, so every later thread inherits it. This equals `uclampset -m 1024 CMD`.
- A: every thread of the process, after the imports.

The clamp value is 1024 in every clamped arm.

## Method

- The CPU governors were forced to `schedutil`. This is the stock governor, and the clamp only acts under it. They were `performance` at the start. The script read them back after the write. It restored them on exit.
- One run is one `bench_decode.py` process. It loads the model, runs 4 untimed tokens, then 128 timed tokens. EOS is suppressed, so every run generates exactly 128 tokens. The decode rate is 127 gaps over the time from the first to the last token.
- One fixed prompt: `Write a short story about a lighthouse keeper who finds a message in a bottle.` Greedy, temperature 0, seed 0.
- Four rounds. Each round runs the four arms once. The order is a Latin square. The rounds run `N T B A`, `T B A N`, `B A N T` and `A N T B`, so each arm takes every position once.
- 15 seconds of idle before each run, and the load average under 0.5 before the first. The script refuses to start when an ANE process is running.
- Each block ran as one `gpu-turn` ticket of the lab's idle guard. The guard queues GPU jobs per host and holds the host's own GPU lock while one runs: `/tmp/m1-gpu.lock` on the 13-inch M1 and `/tmp/m2-gpu.lock` on the M2 Max. No other GPU job ran during a block.
- The command of every run:

```
timeout 900 $PYTHON run_clamped.py MODE bench_decode.py --model MODEL --prompt PROMPT --tokens 128 --temp 0 --seed 0
```

- Environment: `HF_HUB_OFFLINE=1`, `MESA_SHADER_CACHE_DISABLE=1`, a fresh process per run, no other GPU work on the machine.
- A spread is the range of the four values of an arm divided by their median. A spread above 5 percent in arm N would have made a block noisy. None did.

## Environment

Both machines ran kernel 7.1.12-2-20261013.18-sep-ARCH, and the mlx-omarchy tree of each passed `mlx_provenance.py` with `verified=match`.

- 13-inch M1: tree 0.32.4.dev202610102357+e57cb301, libmlx sha256 prefix `9a225264dcf2dbcb`. Run window 2026-10-17 UTC: E4B 03:53 to 04:01, 9B 04:01 to 04:09.
- M2 Max: tree 0.32.4.dev202610161042+75cc013b, libmlx sha256 prefix `3f6d0a63405aa6f7`. Run window 2026-10-17 UTC: 04:26 to 04:34.
- Model gemma-4-E4B: `mlx-community/gemma-4-e4b-it-4bit`, snapshot `475b9088d29754a3379866cf5aeb6b41acd313c2`. Run on the 13-inch M1 only.
- Model Qwen3.5-9B: `mlx-community/Qwen3.5-9B-4bit`, snapshot `8b2b98c00a6b4d291155e4890773ca8f769aee53`. Run on both machines.

The mlx-lm version was not recorded for these runs. The `bench_decode.py` of these runs is the one in `scripts/` at commit `2e1c3ffd`, sha256 prefix `f5062d88f34b0845`. It did not clamp.

## Results

Decode rate in tok/s, the median of four runs, with the spread of the four in brackets. The four values of each arm are in the logs.

| Model, machine | N | T | B | A |
|---|---|---|---|---|
| E4B, 13-inch M1 | 10.61 [0.9%] | 15.28 [0.2%] | 15.27 [0.7%] | 15.27 [0.4%] |
| 9B, 13-inch M1 | 12.45 [1.9%] | 12.52 [0.5%] | 12.57 [0.6%] | 12.57 [0.6%] |
| 9B, M2 Max | 39.10 [0.3%] | 39.38 [0.8%] | 39.39 [1.0%] | 39.41 [0.3%] |

Ratios against N, from the medians:

| Model, machine | T/N | B/N | A/N |
|---|---|---|---|
| E4B, 13-inch M1 | 1.441 | 1.440 | 1.440 |
| 9B, 13-inch M1 | 1.006 | 1.010 | 1.010 |
| 9B, M2 Max | 1.007 | 1.007 | 1.008 |

The token digest is equal in every arm of a block: `6e98b674dcae2e22` for E4B, `958cb85f5f5a32ee` for the 9B on both machines.

Mean P-cluster clock and heatpipe power per arm, over the whole runs (load and warm-up included):

| Model, machine | N | T | B | A |
|---|---|---|---|---|
| E4B, 13-inch M1 | 2036 MHz, 6.1 W | 2529 MHz, 8.6 W | 2563 MHz, 8.4 W | 2576 MHz, 8.7 W |
| 9B, 13-inch M1 | 1914 MHz, 9.5 W | 1954 MHz, 9.9 W | 2114 MHz, 10.4 W | 2104 MHz, 10.5 W |
| 9B, M2 Max | 1950 MHz, 28.9 W | 2088 MHz, 28.5 W | 2449 MHz, 30.9 W | 2312 MHz, 30.3 W |

## Backend evidence

These were collected by a second ticket on each machine, hours after the timed blocks, with the same trees, models and kernel package. The tickets pinned the boot id of the 13-inch M1 and it was unchanged. The ids are removed from the files. The M2 Max had rebooted in between. They show that the Vulkan backend ran. They are in `backend/`.

- Mesa: `mesa 1:26.2.4-1`. Vulkan driver package: `omarchy-mlx-vulkan-omarchy 1:26.2.4.git69bde15340-2`. The M2 Max reports driver `Honeykrisp`, `Mesa 26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362.
- Device: `Apple M2 Max (G14C B1)` from `vulkaninfo` on the M2 Max. `vulkaninfo` is not installed on the 13-inch M1. Its device name comes from `bench_decode.py`: `Apple M1 (G13G B1)`, printed in every `runs/` output.
- Kernel package: `linux-aurora 7.1.12.aurora2-20261013.18`. No separate firmware version string was captured. The `packages:` line of each log has the firmware extractor package and its version.
- Provenance: the first line of every file under `runs/` is the provenance line of that run. All 48 say `verified=match`. The libmlx hash in them equals the one in the Environment section.
- Dispatch trace: one 16-token `bench_decode.py` run per model with `MLX_OMARCHY_TRACE_DISPATCH=1`. Dispatch lines counted:
  - E4B on the 13-inch M1: 35801 lines, 29 distinct kernels.
  - Qwen3.5-9B on the 13-inch M1: 11362 lines, 32 distinct kernels.
  - Qwen3.5-9B on the M2 Max: 11006 lines, 33 distinct kernels.
  The `omarchy-mlx 0.7.28-2` in the `packages:` lines is the installed package, not the tree the timed runs used. The logs list the ten most dispatched kernel ids with their line counts. The ids are labels of that wheel and change between wheels. The largest `count` field of a dispatch line is 524288 for the 9B and 333312 for E4B. It is a parameter of the dispatch, not a running total.
- Device reopen: a new process opened the device after the run and computed a small array. All three logs show `Device(gpu, 0)` and `[2.0, 2.0, 2.0, 2.0]`.
- Serving processes: `litellm=0` and `mlx_lm_server=0` on both machines, counted when the backend evidence ran. They were not counted during the timed blocks.

The trace shows which kernels the backend dispatched. It does not show that no tensor op ran on the CPU.

## What it shows

- On E4B the three clamped placements agree within 0.1 percent and sit 44 percent above N. Clamping the calling thread after the imports, the way the serve does, is enough.
- The threads that decode are created after the clamp and inherit it. In arm T on the M2 Max, 12 threads that existed after the imports stay at `util_min` 0. The 12 threads created later are at 1024.
- The 9B does not move with the clamp on either machine, although the P-cluster clock rose by up to 26 percent on the M2 Max. Its decode did not follow the CPU clock in these runs.

## What it does not show

- The serve process over HTTP. These runs use `bench_decode.py`.
- That no tensor op ran on the CPU. The dispatch trace does not show it.
- Any other model, any other machine, or the `performance` governor, where the clamp has no role.
- Whether the clock gain would appear on a different kernel.
- The mlx-lm version of the runs.

## Files

- `runs/` holds the stdout of all 48 runs, 16 per block. Each file starts with the provenance line and has the decode row, the token digest and the JSON result line.
- `backend/` holds the backend evidence logs described above.
- `ticket-e4b-13in.log`, `ticket-9b-13in.log` and `ticket-9b-m2.log` are the logs of the three blocks. Each has every run with its decode row and digest, the `util_min` table of the threads at start and exit, and the summary lines.
- `clampab.sh`, `run_clamped.py` and `clampab_analyze.py` are the scripts that ran. Two path strings are replaced: the score directory by `$WORK/score` and the sampler directory by `$WORK/sampler`. In the logs the tree path is replaced by `$TREE`, and the boot id that the script prints at the start of a block is removed.
- `SHA256SUMS` holds the sha256 of every file here. Check it with `sha256sum -c SHA256SUMS`.

The scripts as run had these sha256 prefixes: `clampab.sh` `6669a1c70f15c304`, `run_clamped.py` `9249ca2a2ec08145`, `clampab_analyze.py` `d8b7801b808237ec`. The copies here differ only by the replaced paths. The helper modules of `bench_decode.py`, `caps_sim_guard.py` and `mlx_provenance.py`, are the files in `scripts/`.
