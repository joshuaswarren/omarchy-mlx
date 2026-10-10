# oMLX serve on Linux: measured numbers (2026-10-09 and 2026-10-10)

Raw run logs behind the "What you get: measured numbers" section of [docs/omlx-linux.md](../../docs/omlx-linux.md). Host names and local paths are replaced by chip names in the copies; nothing else was edited.

## Method

- `oMLX serve`, one server process per run, greedy decoding, `OMLX_LINUX_CUSTOM_KERNELS=0`, the omarchy-mlx wheel `0.32.4.dev202610091126+a48b7cc`, Mesa package `1:26.2.4-1` (the run header records the package; it does not record whether a private driver-file override was set).
- `new` = oMLX v0.7.1.dev1 (`c0b1056b`) with the repo's packaging patches. `old` = the previous pin (`cc1fdc9a`) with the previous packaging.
- Each run sizes its prompts from a measured prefill rate so one forward stays short, then sends one request (`single_fixed`), a second request of a similar size (`single_a`), and 4 and 8 concurrent requests (`conc4`, `conc8`). Decode speed is the request's own decode tok/s.
- Machines: an M2 Max with 96 GB, and an M1 with 16 GB.

## Table cells to files

| doc row | file | prompt tokens | first token | decode |
|---|---|---|---|---|
| M2 Max, Qwen3.8-27B | `raw/m2max-new-qwen38-27b-1009T2311.run.log` | 267 | 5.104 s | 9.354 |
| M1, Qwen3.5-9B (first run) | `raw/m1-new-qwen35-9b-1009T1712.run.log` | 356 | 5.438 s | 8.288 |
| M1, Qwen3.5-9B (second run) | `raw/m1-new-qwen35-9b-1009T1741.run.log` | 476 | 6.905 s | 8.308 |
| M2 Max, Qwen3.6-35B-A3B, offload 0.25 | `raw/m2max-new-qwen36-35b-offload-1009T1959.run.log` | 90 | 10.015 s | 2.902 |
| M1, Qwen3.6-35B-A3B, offload 0.10 | `raw/m1-new-qwen36-35b-offload-1009T1807.run.log` | 43 | 19.751 s | 1.311 |

Old-pin partners for the version comparison (same machine and model, same method):

| pair | old-pin file | decode (single_fixed) |
|---|---|---|
| M2 Max 27B | `raw/m2max-old-qwen38-27b-1009T2318.run.log` | 9.387 (new 9.354, -0.4 %) |
| M1 9B, first pair | `raw/m1-old-qwen35-9b-1009T1718.run.log` | 8.351 (new 8.288, -0.8 %) |
| M1 9B, second pair | `raw/m1-old-qwen35-9b-1009T1746.run.log` | 8.283 (new 8.308, +0.3 %) |
| M2 Max 35B offload | `raw/m2max-old-qwen36-35b-offload-1009T2304.run.log` | 2.84 (new 2.902, +2.2 %) |
| M1 35B offload | `raw/m1-old-qwen36-35b-offload-1009T1813.run.log` | 1.434 (new 1.311, -8.6 %) |

Prefill rates quoted in the doc are the `pilot rate` line of each run (the second, warm pilot request): 35.27 tok/s (M2 Max 27B, 142-token pilot), 40.82 and 41.26 tok/s (M1 9B, 100-token pilot).

## Packed prefill (concurrent requests)

`raw/m1-new-qwen35-9b-packed-prefill-1010T0524.run.log` and `.packspy.log`: the M1, Qwen3.5-9B, same method, with a wrapper that logs every call of the scheduler's packed prefill forward (rows, tokens, wall time). Six calls: 435 tokens in 6.28 s (69 tok/s), 290 in 4.36 s (67), 176 in 2.87 s (61), 352 in 5.53 s (64), 352 in 5.50 s (64), 176 in 2.92 s (60). One 476-token request in the same run: first token at 6.965 s (about 68 tok/s). Four concurrent prompts: slowest first token 14.547 s; eight: 26.21 s.

## Offload memory estimate

`raw/m2max-estimate-qwen3-30b-res0.25.out`, `...res0.4.out`, `...res0.5.out` and `estimate-vs-peak.py`: an M2 Max, Qwen3-30B-A3B 4-bit (checkpoint 17,181,071,994 bytes), one process per residency, 128-token prefill. Estimate, load and peak memory (bytes): 4,950,481,530 / 4,978,323,456 / 5,137,678,336 (residency 0.25); 7,371,119,226 / 7,696,232,448 / 7,899,725,824 (0.4); 9,027,345,018 / 9,055,186,944 / 9,299,488,768 (0.5). The measured peak is 3.8, 7.2 and 3.0 percent above the estimate.

## Runs that failed (cited in the limits)

- `raw/failed/m1-custom-kernels-on-qwen35-9b-1009T1752.run.log` and `.server-error.txt`: an M1 run with custom kernels on and an older wheel (`0.32.4.dev202610071347+9b5c938`). No request returned tokens; the server log shows a shader compile error at the decode step.
- `raw/failed/m2max-new-qwen36-35b-no-offload-1009T1827.run.log` and `.server-error.txt`: an M2 Max run of Qwen3.6-35B-A3B without offload. No request returned tokens; the server log shows the prefill failing on an unsupported kernel feature (`device pointer arithmetic`).
