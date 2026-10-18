# CPU governor: served decode and idle power on the 16-inch M1 Max

Two measurements on one machine. The first compares the `schedutil` and `performance` CPU governors while the repo's serve process serves gemma-4-26B-A4B. The second compares the same two governors with nothing running. It backs the paragraph in `docs/BENCHMARKS.md`, section "Set the CPU governor for local AI".

Result: served decode reads 13.62 tok/s under `schedutil` and 13.93 under `performance`, a ratio of 1.023. At idle the `performance` governor draws about 2 W more system power. The kernel default stays.

## 1. Served decode

### Method

- One server process per run: `python -m mlx_omarchy_serve` from `serve/` of omarchy-mlx at commit `00751337`, started on a fresh port. The serve raises `uclamp_min` to 1024 on its startup thread in every run, under both governors (`shim: uclamp placement applied (uclamp_min=1024)` in every `serve/runs/run-*.serve.log`). Arm S therefore measures `schedutil` with the clamp, and arm P adds the `performance` governor on top.
- Each run: one 8-token warm-up request (the model loads on the first request), then 3 request pairs. A pair is a 32-token request and a 128-token request, not streamed, with the prompt `Write a short story about a lighthouse keeper who finds a message in a bottle.`. The decode rate comes from the 128-token request.
- Four rounds of two runs. The order is S P, P S, S P, P S. 15 seconds between runs. Before the first run the script waits for a 1-minute load average under 0.5 (up to 300 s).
- The script writes the governor of every cpufreq policy with `sudo -n tee`, reads it back, and restores the starting value (`performance`) on exit. Read back after the block: `performance` on all three policies.
- The script refuses to start when an ANE process is running. None ran.
- One block ran as one `gpu-turn` ticket of the lab's idle guard, which holds the host's GPU lock for the block. No other GPU job ran during it. The ticket limit was 30 minutes; the block took 8 minutes 25 seconds.
- Exact command (the ticket line; the path replacements are listed in section 5):

```
unset VK_ICD_FILENAMES VK_DRIVER_FILES ; SERVE_CMD="env PYTHONPATH=$WORK/score/gprp:$WORK/score/mlxlm-latest-deps:$WORK/serve-pkg/serve MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1 PYTHONDONTWRITEBYTECODE=1 $WORK/score/venv/bin/python -m mlx_omarchy_serve" bash $WORK/run/serveab7.sh g26 mlx-community/gemma-4-26b-a4b-it-4bit 15.0
```

### Environment

- Machine: 16-inch M1 Max (T6001, GPU G13C), AC power, no other load. Run window 2026-10-17 UTC 22:18:55 to 22:27:20.
- Kernel: the running kernel is `7.1.12-2-12.6-sep-ARCH` (build 2026-10-08 10:26:57). The installed kernel package is `linux-aurora 7.1.12.aurora2-20261013.18`. The machine had not rebooted since the 10-08 build.
- Mesa: `mesa 1:26.2.4-1`. Vulkan driver package `omarchy-mlx-vulkan-omarchy 1:26.2.4.git69bde15340-2`. The driver library has sha256 prefix `40282811c6e14ef9` and a file date of 2026-10-11, before the run. The last package transaction on the machine before the run was on 2026-10-15.
- Tree: mlx-omarchy `0.32.4.dev202610141543+7604d92c`, wheel sha256 `3b9ecf001de4dc49a63233d027c8b091f98c6ccef592bab0f28e21b5626eb18e`, `libmlx.so` sha256 `7c9c9fff09bbc988832797b75522156272837e0bd6df46ade96fafa58d93dc7a`.
- mlx-lm 0.32.0. The pinned version is 0.31.3; `MLX_OMARCHY_ALLOW_UNPINNED_MLXLM=1` allows the newer one, and the server prints the warning in line 1 of each `serve/runs/run-*.serve.log`.
- Serve package: the `serve/` tree at commit `00751337`. Manifest (sha256 of the sorted per-file sha256 list, first 16 hex): `b2f4c8cff7dc027a`. The preflight of the ticket checked it.
- Model: `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`, 15.0 GiB of weights. Offline (`--offline`).

### Results

Decode rate of the 128-token request, tok/s. The three values of each run are in `serve/gov7-serve.out`.

| Round | S run median | P run median | P/S |
|---|---|---|---|
| 1 | 13.68 | 13.93 | 1.018 |
| 2 | 13.31 | 13.88 | 1.043 |
| 3 | 13.76 | 13.98 | 1.016 |
| 4 | 13.33 | 13.97 | 1.048 |

Over all 12 requests of each arm: S median 13.62 (13.08 to 13.77), P median 13.93 (13.85 to 14.04), P/S 1.023. The spread of the four run medians is 3.4 percent in S and 0.7 percent in P (the analyzer's definition). P is above S in 4 of 4 rounds. The rule fixed before the run: build a governor helper if P/S is at least 1.10, document if it is at most 1.05, call the run noisy if an arm spread is above 5 percent. P/S is 1.023, so the outcome is "document".

The text of all 24 requests has the same digest, `26437c7247eef04b`. Every request finished with `finish=length`.

Mean over the whole runs of each arm (the sampler runs at 1 Hz; load and warm-up are included), from `samples/rows-serve7-g26-20261017T221855Z.csv`:

| Arm | E cores, policy0 | P cores 2 to 5, policy2 | P cores 6 to 9, policy6 | Heatpipe power | System power | Battery hotspot | Fan 1 |
|---|---|---|---|---|---|---|---|
| S, schedutil | 723 MHz | 2418 MHz | 2379 MHz | 36.89 W | 45.25 W | 30.4 C | 1523 rpm |
| P, performance | 2064 MHz | 3036 MHz | 3036 MHz | 39.47 W | 48.35 W | 30.3 C | 1520 rpm |

P draws 2.6 W more heatpipe power and 3.1 W more system power over these runs for 2.3 percent more decode.

The `util_min` table of the server threads is in every run log (`THREADS` lines in `serve/gov7-serve.out`): 12 of 23 threads at 1024 in every run, in both arms. The threads that decode are among them.

## 2. Idle power

- One block: `idle/gov7-idle.out`, `samples/rows-idlepow-20261017T211019Z.csv`. Four windows of 180 seconds with nothing running, in the order P S S P. The first 40 seconds of each window are dropped. Run window 2026-10-17 UTC 21:10:19 to 21:22:19. The governors were `performance` before and after.
- Same machine, kernel, driver and AC power as section 1. No serving process was running (the preflight counts `litellm`, `llama-server`, `mlx_omarchy_serve`, `mlx_lm.server` and `vllm` and requires 0), and no ANE process (counters unchanged in all windows).

| Window | Arm | System power | Heatpipe power | Clock, policy2 (P cores 2 to 5) | Clock, policy6 (P cores 6 to 9) |
|---|---|---|---|---|---|
| 1 | P | 10.65 W | 5.91 W | 3036 MHz | 3036 MHz |
| 2 | S | 8.40 W | 4.47 W | 1288 MHz | 1399 MHz |
| 3 | S | 8.38 W | 4.52 W | 1319 MHz | 1396 MHz |
| 4 | P | 10.02 W | 5.80 W | 3036 MHz | 3036 MHz |

P minus S: system power +1.94 W (10.33 W against 8.39 W, +23.1 percent) and heatpipe power +1.36 W (+30.2 percent). The window gaps are 0.63 W and 0.02 W (system). The fan ran in window 1 only (1496 to 1558 rpm in every row, 0 in windows 2 to 4), so window 1 sits 0.63 W above window 4. The last P window alone gives +1.62 W (+19.4 percent) and the first alone +2.25 W (+26.9 percent). The battery hotspot temperature window means are 31.9, 31.4, 31.0 and 30.8 C in window order (every row between 30.7 and 32.1 C), so the two governors do not differ in temperature beyond the window gaps. AC input power equals the system figure within 0.01 W.

## 3. Backend evidence (read after the run)

The timed blocks did not capture these fields. A separate ticket read them on 2026-10-18 UTC 04:00:11 to 04:00:22, after the machine rebooted at 01:42 (a later boot than both timed blocks). The kernel string, the mlx tree, the model and the driver library are the same. The governors were `performance` and unchanged by the ticket. Nothing in this section is a timed measurement.

- Same machine and software as the timed blocks, checked by the ticket's preflight: running kernel `7.1.12-2-12.6-sep-ARCH`, `libmlx.so` sha256 prefix `7c9c9fff09bbc988`, driver library sha256 prefix `40282811c6e14ef9`, model snapshot as in section 1.
- Provenance line of the tree, from `scripts/mlx_provenance.py` (CPU only): `mlx-omarchy 0.32.4.dev202610141543+7604d92c mx=0.32.4.dev202610141543+7604d92c verified=match harness=unknown core.cpython-314-aarch64-linux-gnu.so=sha256:7542a0bcff502c00 libmlx.so=sha256:7c9c9fff09bbc988`. The traced run printed the same line. `harness=unknown` means the harness commit was not recorded.
- Vulkan device, from `vulkaninfo`: `Apple M1 Max (G13C C0)`, driver `Honeykrisp`, `Mesa 26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362, driver version 26.2.99. The same listing has a second device, the `llvmpipe` software renderer. The traced run reports `"device": "Apple M1 Max (G13C C0)"`.
- Packages: `mesa 1:26.2.4-1` and the Vulkan driver package `omarchy-mlx-vulkan-omarchy 1:26.2.4.git69bde15340-2`. The full list, with the kernel package `linux-aurora 7.1.12.aurora2-20261013.18`, is in the `packages:` line of `backend/backend-g26-16in.log`.
- Firmware: no separate firmware version string was captured. The `packages:` line has the firmware extractor package and its version.
- Dispatch trace: one 16-token `bench_decode.py` run of gemma-4-26B-A4B with `MLX_OMARCHY_TRACE_DISPATCH=1`. It is not one of the 24 served requests. It counted 42207 dispatch lines and 40 distinct kernels. The largest `count` field of a dispatch line is 698368, a parameter of the dispatch and not a running total. The ten most dispatched kernel ids with their line counts are in the `TRACE` line of the log. The ids are labels of this wheel and change between wheels. The run decoded at 16.12 tok/s over 15 tokens with the `performance` governor and tracing on; that figure is not comparable with the served rows. The trace shows which kernels the backend dispatched. It does not show that no tensor op ran on the CPU.
- Device reopen: a new process after the traced run opened the device and computed a small array: `Device(gpu, 0)` and `[2.0, 2.0, 2.0, 2.0]`. In the timed block each of the 8 runs started a new server process after the previous one was stopped. All 8 opened the device and served their requests with the same text digest.
- Thermal: no cooldown or temperature control. The script waited for a load average under 0.5 before the first run only. The load average read at the start of each run is in the log: 0.44, 0.51, 0.74, 0.66, 0.61, 0.65, 0.75 and 0.59, in run order. Seven of the eight are above 0.5 because each run starts 15 seconds after the previous server stopped. The battery hotspot sensor read 30.3 to 30.4 C on average in both arms.

## 4. What it shows and does not show

Shows:
- On this machine, for this model and this serve process, the `performance` governor adds 2.3 percent to decode (4 of 4 rounds above, 1.6 to 4.8 percent per round) and costs 2.6 W more heatpipe power.
- At idle the `performance` governor draws between 1.6 and 2.3 W more system power.

Does not show:
- Any other machine or model. The 13-inch M1 and the 14-inch M2 Max are not measured here. The direct `bench_decode.py` rows in `docs/BENCHMARKS.md` are a different measurement, and their logs are not in this receipt. Whether they carried the serve clamp is not established here.
- A run without the clamp under `schedutil`. Arm S carries the serve's clamp by design.
- Whether the mlx-lm version (0.32.0, unpinned) changes the ratio.
- Power at the wall beyond the AC input counter, or fan noise.
- Two windows per governor is a weak estimate of the idle gap.

## 5. Files

- `scripts/` holds the scripts that ran: `serveab7.sh` (sha256 prefix `7ab70a9a9ffbd050`), `serveab7_analyze.py` (`1af6c832cc1d2415`), `serve_client.py` (`ee2cb55198c8c12a`), `thread_table.py` (`2bef8caee53b2909`), `idlepow.sh` (`24b8dbba2abb58af`), `idlepow_analyze.py` (`d0207ac4e5e88ed9`) and the sampler `row-sampler.sh` (`384b88da4660a314`). The copies here differ from those prefixes only by the replaced paths.
- `serve/gov7-serve.out` is the ticket log of section 1. `serve/runs/` holds, for each of the 8 runs, the server log, the warm-up reply and the 3 request replies.
- `idle/gov7-idle.out` and `idle/block.log` are the logs of section 2.
- `backend/` holds the logs of section 3: `backend-g26-16in.log` (the ticket log), `traced.out` (the output of the traced run) and `reopen.out`. In the copies the script directory is replaced by `$WORK/run`.
- `samples/` holds the two 1 Hz sampler files. In the analyzer lines of the logs, `pcluster_MHz` and the `CLOCK` lines are policy6 only.
- Replaced in the copies: `$HOME`, the score directory by `$WORK/score`, the serve package directory by `$WORK/serve-pkg`, the script directory by `$WORK/run`, the sampler directory by `$WORK/sampler`; the boot id that the scripts print is replaced by `BOOT`; one long dash character in the server warning line is replaced by a hyphen.
- `SHA256SUMS` holds the sha256 of every file here. Check it with `sha256sum -c SHA256SUMS`.
