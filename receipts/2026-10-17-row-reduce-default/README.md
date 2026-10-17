# Row-reduce Sum route on by default: receipt

This receipt records the hardware runs behind the default change in the pull request "Reduce: the workgroup-per-row Sum route is the default, off with `MLX_OMARCHY_ROW_REDUCE=0`". The `SHA256SUMS` file lists the source record of every number below.

## Source

- The route merged in PR 215 at commit `566d0bb7cd05a13a20fdd1ed514bbc1d67a05887`. It routes Sum.
- Every run below used a build of an earlier head of that route, `50f72c21`. That head routes Sum, Max and Min. `libmlx.so` sha256 `3bb4504761874dc51146eb2dfdf46efa22ccef39a7159c1c18f1c40bc0d6bfdb`.
- In the traced graph only the mean sums took the route. The decode step shows 12 row-kernel dispatches. The 8 other reduce dispatches are the same count in the route-off arm and in the float64-reference arm, so no Max or Min of this graph moved to the row kernel.

## Machine and driver

- 14-inch M2 Max (G14C), Linux kernel 7.1.12-2-20261013.18-sep-ARCH, CPU governor performance.
- Vulkan driver library sha256 prefix `40282811c6e14ef9` (Mesa 26.3.0-devel, git 69bde15340). The M2 moved to a different library after these runs. The cluster run used the same library on its M2 ranks.
- Platform firmware of the 14-inch M2 Max, read from its device tree on 2026-10-17 after the runs, on a later boot of the same machine: OS firmware version 13.5, system firmware version 27.0, iBoot-8422.141.2, boot loader v1.6.1-omarchy.aurora14. The kernel log of the timing boot was not kept, so the firmware lines the GPU driver printed at that boot are not in this receipt.

## Model

Muse Spark Open 2-bit. The single-machine runs cover layers 14 to 17 of 62. Snapshot `model.safetensors.index.json` sha256 `bc53f5a5f9afb6960c828f2192eef64898a1de173a3a34e6c46b19625f8629eb`. Tensor list sha256 `c5b35cff78450ff3b29ee1ed073b4a9644ade4d7007b40045c2af41db6af9eda`.

## 1. Timing of the 4-layer decode step

Command, as the guard ran it (paths shortened): `bash layer-bench7-run.sh rrow <tree>`. The wrapper has sha256 `1b85a20c7e2eabc42d13dde7e701e4f485e2233a7ead5e96279040b0808dc05e`. It runs the pinned benchmark harness (`muse-layer-bench2.py`, sha256 prefix `641ddd40eabd8c13`) and the micro-benchmark (`muse-glue-micro.py`, `344c746792b51593`) through `benchpy.sh` (sha256 `cacb55f2391e846752c38be5da73f03e110974969cdd425b68f971e0cd6630cc`) in one boot. Graph compilation is off. Layers 14 to 17, 145 tokens, 12 decode steps per arm. The arm `sub-on` is the route off. The wrapper runs these five commands in order:

```
benchpy.sh <tree> muse-layer-bench2.py <tiny model> 2 5 --arms "sub-on,env:MLX_OMARCHY_ROW_REDUCE=1" --rounds 2 --steps 5 --no-ops
for v in 0 1 0 1: benchpy.sh <tree> muse-glue-micro.py --env=MLX_OMARCHY_ROW_REDUCE=$v
MLX_OMARCHY_TRACE_DISPATCH=1 benchpy.sh <tree> muse-layer-bench2.py <snapshot> 14 15 --trace --arms "sub-on,env:MLX_OMARCHY_ROW_REDUCE=1" --rounds 1 --steps 4 --no-ops
benchpy.sh <tree> muse-layer-bench2.py <snapshot> 14 17 --arms "sub-on,env:MLX_OMARCHY_ROW_REDUCE=1" --rounds 3 --steps 12 --no-ops
```

The first command is a smoke run on a tiny model. The micro-benchmark ran twice per setting, in the order 0, 1, 0, 1. The mean over (1, 1, 8192) read 908.3 and 906.8 us with the route off and 7.6 and 7.5 us with it on. The text below quotes 908.3 and 7.6.

Procedure. The arms ran in the order off, on in round 0, on, off in round 1 and off, on in round 2, inside one process. Before the job the guard's settle step read a one-minute load of 0.37 and passed. The CPU governor was performance. No temperature, fan or throttle reading was recorded for this run. The guard counted GPU timeouts before and after the job: 0 and 0. Every process opened the device (printed as `Device(gpu, 0)`) and returned 0. No separate device reopen test was run.

| Quantity | Route off | Route on |
| --- | --- | --- |
| Decode step median per round (ms) | 38.585, 38.090, 38.200 | 27.545, 27.555, 27.575 |
| Per layer (ms) | 9.550 | 6.889 |
| Spread over rounds (ms) | 0.49 | 0.03 |
| Prefill, 4 layers (ms) | 1668.0, 1668.0, 1667.5 | 1656.3, 1656.2, 1656.0 |

The 4-layer decode step is 27.9 percent shorter. The two arms give different output digests, as expected from the new association. The largest absolute difference of the final hidden state is 0.0625 (maximum of the base state 7.5, rms 0.656).

Micro-benchmark, microseconds per call, route off then on: mean over the last axis of a (1, 1, 8192) float32 tensor 908.3 then 7.6; the uncompiled float32 norm 944.8 then 40.9.

## 2. Hidden-state fidelity against a float64 sum

Both runs are preregistered. The reference replaces the mean inside the float32 norm with a float64 mean of the same squares, rounded once to float32. The metric is u = ln((d_R + delta) / (d_S + delta)) per window. Here d is the rms distance of the last hidden state from the reference, and delta is 2^-9 of its rms. The interval is the mean plus or minus 2 standard errors over 32 windows, with one look. Inputs are synthetic and heavy tailed, with one decode token. This is not a model-quality measurement.

| Run | Graph | Windows | Mean u | Interval | Route closer | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | uncompiled | 32 to 63 | -0.9166 | [-1.2201, -0.6131] | 31 of 32 | ACCEPT-BETTER |
| 3 | compiled | 64 to 95 | -0.4874 | [-0.7540, -0.2208] | 31 of 32 | ACCEPT-BETTER |

Commands, as the guard ran them (paths shortened). Run 2: `bash rr-run4.sh <out> 32 64` (sha256 `766e1ecc46d7f3a9c2ede24bc671b6ce0ce9c302c75b79a752212b32e9925dc6`). Run 3: `bash rr-run5.sh <out> 64 96` (sha256 `9d635594b37f280bcae1d9a3ff7a72cc96e54546b7edaaf3bda273b674f27800`). Each script first takes a one-window dispatch trace of the arms with `MLX_OMARCHY_TRACE_DISPATCH=1 benchpy.sh <tree> <harness> <snapshot> 14 17 --arm <A> --windows 0:1 --trace-window 0 --out <dir> --libmlx 3bb4504761874dc5`. Then it runs one process per arm with `benchpy.sh <tree> <harness> <snapshot> 14 17 --arm <A> --windows <first>:<last> --out <dir> --libmlx 3bb4504761874dc5`. Run 2 uses the arms S, R, S2, R2 and F and the harness `rowreduce_fidelity7.py` (`8459384397cd344ac5aea93fb3d9f66c5a55141c874a8265214c44fdc8c8cbea`). Run 3 uses S, R, F, Sc, Rc, Sc2 and Rc2 and `rowreduce_fidelity8.py` (`4d27bd44f2164b7ef7298d0bcfe5cc71722db51b7cf5d97a62f789f63a9c67cd`). S is the route off, R is the route on and F is the float64 reference. The arms with c run with graph compilation on, started with the environment of `benchpy.sh` minus `MLX_DISABLE_COMPILE`. Both runs used the 14-inch M2 Max with the governor performance. The guard counted GPU timeouts before and after each job: 0 and 0 in both. The settle step read a load of 0.05 and 0.51 and skipped the wait in both runs. No temperature, fan or throttle reading was recorded. Every arm opened the device and finished, and no separate device reopen test was run.

Guards of both runs passed: the route-off and route-on arms are bit-equal to their repeats in all 32 windows, call counts are equal across arms, and the reference differs from both arms.

Dispatch trace of the decode step in window 0 of run 3 (compiled arms Sc and Rc, uncompiled S and R, reference F): total dispatches S 528, Sc 484, R 520, Rc 476, F 496. `ReduceRowF32` dispatches S 0, Sc 0, R 12, Rc 12, F 0. Compilation lowers the totals in both arms, which shows it was active. Run 2 trace: `ReduceRowF32` S 0, R 12, F 0; `ReduceGeneralF32` S 28, R 8, F 8.

## 3. Whole model on a cluster

All 62 layers ran on five ranks across four Apple Silicon machines, on one loaded ring. The route was switched per request through each rank's live environment file. Off is the variable unset and on is `1`. The log of every rank shows the setting in the on requests. The text of every answer was read from the capture files. This is not a quality benchmark.

Commands, as the scripts and logs show them (paths shortened, machines named by model). The driver (`pL2-driver.sh`, sha256 prefix `9d3a3cdb68ba1b81`) is run as `pL2-driver.sh 24,12,12,12,2`. It checks free memory and the tree on each node, sets the pipeline-prune flag file on the four nodes and starts the Mac Studio script, which runs:

```
python -m omarchy_cluster.cli serve <model> --nodes <studio>,<m2>,<m2>,<16-inch>,<13-inch> --split 24,12,12,12,2 --no-fit-check \
  --ring-ip <host>=<ip> ... --python-mac <dir>/rankpy.sh --python-linux <dir>/rankpy.sh --port 18020 --engine-port 18031 --ctx 4096
```

The agent starts each rank as `<rankpy.sh> -m omarchy_cluster.rank --model <model> --layers A-B --rank N --hostfile <state>/ring-hostfile.json --engine-port 18031`. The wrapper `rankpy.sh` (sha256 prefix `d50164d8a4323cf8`) adds `MLX_OMARCHY_PIPELINE_PRUNE=1`, `HK_SYSMEM` (16-inch 59000000000, 13-inch 11000000000, none on the M2), `MLX_DISABLE_COMPILE=1` and `OMARCHY_LIVE_ENV=<dir>/live-env`, and puts the route tree (libmlx `3bb4504761874dc5`) first on `PYTHONPATH` on the three Linux nodes. The pipeline-prune flag file is the pruning of PR 199. The 120 s no-progress variable was not set in this run.

Route switch. Before each pass the driver copies one of two files to `<dir>/live-env` on the M2, the 16-inch and the 13-inch. Off is a one-line comment, so the variable is unset. On is the line `MLX_OMARCHY_ROW_REDUCE=1`. Each Vulkan rank reads the file at the start of every job and logs `live env now: {...}` on every change. The Mac Studio rank has no live-env file.

Capture. `MUSE_CAPTURE_META="row_reduce=<ARM>;pass=<n>" CAPTURE_LENGTHS=<64|128|256> bash capture-full62.sh http://<studio>:18020 <run>/captures/pass<n>-<ARM>-<len> <model> 5 4 "<placement text>"` (sha256 prefix `e62d59570307519b`). The request is a POST to `/v1/completions` with `{"model": "<model>", "prompt": "<|begin_of_text|>" + <prompt file text>, "raw": true, "max_tokens": <64|128|256>, "temperature": 0, "timing": true}`. A warm-up request with `max_tokens` 8 ran before the passes.

Score. `score_client.py` (sha256 prefix `e95601997d74e152`) posts `{"text": "<|begin_of_text|>" + <passage>, "limit": 256}` to `http://<studio>:18031/score` for three passages in turn, in the order off, on, on, off. Every rank runs one forward and rank 0 returns the NLL of tokens 2 to 256.

Procedure. The passes ran back to back. The only step between two passes was the copy of the live-env file, which takes a second or two. No settle wait, CPU governor value or temperature was read or set. The driver did not read the GPU timeout counter before or after the run, so no count is recorded. Reads by other jobs found no kernel GPU timeout on the 13-inch boot (read at 14:53Z) and on the M2 boot (read at 16:23Z) after the run. The 13-inch rank logged user-space Vulkan timeline timeouts in two earlier loads. No separate device reopen test was run.

Text: every on text equals its off text. The runs were 64 tokens in the order off, on, on, off, and 128 and 256 tokens in the order off, on. An off-against-off repeat is byte-identical.

Decode speed from the gateway, tokens per second:

| Prompt | Route off | Route on |
| --- | --- | --- |
| 17 x 23 (2 tokens) | 3.91, 3.99 (64); 3.84 (128); 3.96 (256) | 4.93, 4.95 (64); 4.77 (128); 4.82 (256) |
| capital of Australia (8 tokens) | 2.33, 2.22 (64); 2.28 (128); 2.22 (256) | 2.93, 2.92 (64); 2.86 (128); 2.83 (256) |
| is_prime function (64 tokens, cut) | 2.04, 2.00 | 2.54, 2.54 |
| is_prime function (113 tokens) | 1.98 (128), 1.96 (256) | 2.53 (128), 2.55 (256) |

The two short answers have 1 and 7 decode gaps. The `is_prime` rows are the speed record: +25.7 percent at 64 tokens, +27.8 percent at 128 and +30.1 percent at 256. Time to first token for the 161-token prompt does not move: 26.1 to 27.1 s off and 25.9 to 27.0 s on.

Negative log-likelihood, teacher forced, 255 scored tokens per passage, same load. Off against off and on against on are bit-identical per token. On differs from off on 98 to 100 percent of tokens.

| Passage | NLL off | NLL on | On minus off | Paired SE | 2 SE interval |
| --- | --- | --- | --- | --- | --- |
| GPL v3 | 0.135350 | 0.133092 | -0.002258 | 0.005459 | [-0.0132, +0.0087] |
| Apache 2 | 0.257471 | 0.245036 | -0.012435 | 0.010763 | [-0.0340, +0.0091] |
| README | 2.866704 | 2.874434 | +0.007730 | 0.013263 | [-0.0188, +0.0343] |
| all 765 tokens | | | -0.002321 | 0.005977 | [-0.0143, +0.0096] |

The registered rule was a mean difference within 0.01 nats per token on each passage. Apache 2 misses it with 0.0124, so the rule as written fails. On that passage the route gives the lower NLL, by 1.2 standard errors. The paired per-token standard deviation was 0.09, 0.17 and 0.21, three to four times the estimate that set the bound. Three tokens carry the mean of the Apache 2 difference; the largest changes by -2.6 nats. The follow-up study is registered with its rule fixed in advance. The margin of +0.02 was fixed before the study ran and is twice the 0.01 of the first rule. It does not replace the first rule, which failed on one passage and is not scored again. It uses eight passages written after the model's training data, with 255 scored tokens each. The unit is the passage and the interval is the mean plus or minus 2 standard errors. The route stays accepted if the upper end of the interval is at most +0.02 nats per token. It is rejected if the upper end is above.

## 4. Unit test on three chips

`omarchy_reduce_row_tests` was built once from the pull request head `c9717a7b6dab2d954af8cae2b69405cd584f7ce1` (CPU build, no GPU use) and the one binary (sha256 `afc4f949a065233dca3972e8720f553eb988f6fda309c52abccccc8d332d6ef1`) ran on each machine. The code of that head is the code of the pull request. The test needs an Apple GPU; without one it prints "Skipping" and reports no assertions. No run printed "Skipping".

| Machine | Driver library sha256 prefix | Mesa | Cases | Assertions |
| --- | --- | --- | --- | --- |
| 13-inch M1 (T8103) | `40282811c6e14ef9` | git 69bde15340 | 6 of 6 | 216 of 216 |
| 16-inch M1 Max (T6001) | `40282811c6e14ef9` | git 69bde15340 | 6 of 6 | 216 of 216 |
| 14-inch M2 Max (T6021) | `c499962c742f399a` | git a5b1797446 | 6 of 6 | 216 of 216 |

Build, once on the 14-inch M2 Max at `c9717a7b` (CPU only): `cmake -S .work/mlx -B .work/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF -DMLX_BUILD_TESTS=ON -DMLX_BUILD_PYTHON_BINDINGS=OFF -DBUILD_SHARED_LIBS=OFF`, then `cmake --build .work/build --target omarchy_reduce_row_tests`. The build has no machine-specific compiler flags. Run, on each machine (the guard ran `bash rr226-doctest.sh run <out> <binary> <sha256>`, which runs these three commands): `env -u MLX_OMARCHY_ROW_REDUCE <binary>`, `env MLX_OMARCHY_ROW_REDUCE=0 <binary>` and `env MLX_OMARCHY_ROW_REDUCE=1 <binary>`.

| Machine | Kernel | OS firmware | System firmware | iBoot |
| --- | --- | --- | --- | --- |
| 13-inch M1 | 7.1.12-2-20261013.18-sep-ARCH | 13.5 | 27.0 | iBoot-8422.141.2 |
| 16-inch M1 Max | 7.1.12-2-12.6-sep-ARCH | 13.5 | 26.6.2 | iBoot-8422.141.2 |
| 14-inch M2 Max | 7.1.12-2-20261013.18-sep-ARCH | 13.5 | 27.0 | iBoot-8422.141.2 |

The firmware values are read from each device tree on 2026-10-17 after the runs. The guard counted GPU timeouts before and after each of the three jobs: 0 and 0. Each run opened the device and returned 0. The test checks results and takes no timing, so a thermal procedure and a backend dispatch trace do not apply to it. No separate device reopen test was run.

Each machine ran the binary three times, with `MLX_OMARCHY_ROW_REDUCE` unset, `0` and `1`. The test sets that variable itself for every case, so the outer setting does not change a run. All nine outputs are byte-identical, `device-doctest.txt` in this directory is one of them (sha256 `88485fa676e78872e3138fe633e9c5f1f415dbc5d31239bdeef60a13bcc39cd1`). For float32, float16 and bfloat16 rows of 256 to 70001 elements it lists the worst row's error of each route against a long double sum, in units of 2^-24 times the sum of magnitudes. Each case also checks the row route against a bound of 32 of those units plus the storage step. The values on the three chips are equal.

## Source records

`SHA256SUMS` holds the sha256 of the source record of each section: the benchmark log, output and traces of the timing run, the two fidelity result files with their ticket logs, and the cluster result and comparison files. The source files are in the lab archive, not in this directory, so `sha256sum -c` run here cannot check those lines.
