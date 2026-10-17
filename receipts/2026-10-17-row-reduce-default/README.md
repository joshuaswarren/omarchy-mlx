# Row-reduce Sum route on by default: receipt

This receipt records the hardware runs behind the default change in the pull request "Reduce: the workgroup-per-row Sum route is the default, off with `MLX_OMARCHY_ROW_REDUCE=0`". The `SHA256SUMS` file lists the source record of every number below.

## Source

- The route merged in PR 215 at commit `566d0bb7cd05a13a20fdd1ed514bbc1d67a05887`. It routes Sum.
- Every run below used a build of an earlier head of that route, `50f72c21`. That head routes Sum, Max and Min. `libmlx.so` sha256 `3bb4504761874dc51146eb2dfdf46efa22ccef39a7159c1c18f1c40bc0d6bfdb`.
- In the traced graph only the mean sums took the route. The decode step shows 12 row-kernel dispatches. The 8 other reduce dispatches are the same count in the route-off arm and in the float64-reference arm, so no Max or Min of this graph moved to the row kernel.

## Machine and driver

- 14-inch M2 Max (G14C), Linux kernel 7.1.12-2-20261013.18-sep-ARCH, CPU governor performance.
- Vulkan driver library sha256 prefix `40282811c6e14ef9` (Mesa 26.3.0-devel, git 69bde15340). The M2 moved to a different library after these runs. The cluster run used the same library on its M2 ranks.

## Model

Muse Spark Open 2-bit. The single-machine runs cover layers 14 to 17 of 62. Snapshot `model.safetensors.index.json` sha256 `bc53f5a5f9afb6960c828f2192eef64898a1de173a3a34e6c46b19625f8629eb`. Tensor list sha256 `c5b35cff78450ff3b29ee1ed073b4a9644ade4d7007b40045c2af41db6af9eda`.

## 1. Timing of the 4-layer decode step

Command: a wrapper that runs the pinned benchmark harness (sha256 prefix `641ddd40eabd8c13`) and the micro-benchmark (`344c746792b51593`) in one boot with graph compilation off. It runs three rounds that alternate the route off and on, with the route set by the environment of each process. Layers 14 to 17, 145 tokens, 12 decode steps per arm.

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

Guards of both runs passed: the route-off and route-on arms are bit-equal to their repeats in all 32 windows, call counts are equal across arms, and the reference differs from both arms.

Dispatch trace of the decode step in window 0 of run 3 (compiled arms Sc and Rc, uncompiled S and R, reference F): total dispatches S 528, Sc 484, R 520, Rc 476, F 496. `ReduceRowF32` dispatches S 0, Sc 0, R 12, Rc 12, F 0. Compilation lowers the totals in both arms, which shows it was active. Run 2 trace: `ReduceRowF32` S 0, R 12, F 0; `ReduceGeneralF32` S 28, R 8, F 8.

## 3. Whole model on a cluster

All 62 layers ran on five ranks across four Apple Silicon machines, on one loaded ring. The route was switched per request through each rank's live environment file. Off is the variable unset and on is `1`. The log of every rank shows the setting in the on requests. The text of every answer was read from the capture files. This is not a quality benchmark.

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

Each machine ran the binary three times, with `MLX_OMARCHY_ROW_REDUCE` unset, `0` and `1`. The test sets that variable itself for every case, so the outer setting does not change a run. All nine outputs are byte-identical, `device-doctest.txt` in this directory is one of them (sha256 `88485fa676e78872e3138fe633e9c5f1f415dbc5d31239bdeef60a13bcc39cd1`). For float32, float16 and bfloat16 rows of 256 to 70001 elements it lists the worst row's error of each route against a long double sum, in units of 2^-24 times the sum of magnitudes. Each case also checks the row route against a bound of 32 of those units plus the storage step. The values on the three chips are equal.

## Source records

`SHA256SUMS` holds the sha256 of the source record of each section: the benchmark log, output and traces of the timing run, the two fidelity result files with their ticket logs, and the cluster result and comparison files. The source files are in the lab archive, not in this directory, so `sha256sum -c` run here cannot check those lines.
