# Vulkan allocator cache fix — M2 verification

## Build

- Built a wheel from the fresh M2 checkout at allocator commit `f9500ec56` with `DEV_RELEASE=1`, CMake parallelism 4, and the repository's pinned MLX source bundle.
- Wheel: `mlx_omarchy-0.32.4.dev202610051848+f9500ec-cp314-cp314-linux_aarch64.whl`.
- SHA-256: `9acbd5fce64753f1c14d08e89e398538ad415e9eb1f8bc7d53d0a356dadbe2b4`.
- Hardware was Apple M2 Max with a 50,600,083,456-byte Honeykrisp heap. The default working-set/cache ceiling is 95% of that heap (48,070,079,283 bytes; 44.76 GiB).

## M2 test suites

Built the Omarchy test targets in a separate Release CMake build with CPU and Omarchy enabled, and ran each on the GPU through `gpu-turn`:

- `omarchy_runtime_tests`: 47/47 cases and 22,890/22,890 assertions passed. This includes cache growth across shape-changing allocations, the working-set release gate, memory/wired-limit behavior, and injected first-allocation OOM followed by a successful retry.
- `omarchy_primitive_tests`: 104/104 cases and 2,743,003/2,743,003 assertions passed.
- `omarchy_matmul_family_tests`: exit status 0.
- `omarchy_fast_regression_tests`: exit status 0.

The first runtime-test attempt exposed a pre-existing 32-bit test literal (`64u << 30`) that evaluated to zero. The test now uses a 64-bit literal and passes on the M2.

## A20 oMLX request

Replayed the existing one-model server harness with Qwen3-4B-Instruct-2507-4bit, TurboQuant KV 8-bit, memory guard off, one concurrent request, 651-token prompt, and `max_tokens=96`.

- Chat completion returned HTTP 200 with a non-empty response; `control.txt` was `request_success`.
- Elapsed time: 98.24 s; response body: 779 bytes.
- Profile: 88 memory events, no `async_eval_failure` events.
- Maximum observed cache memory: 4,815,446,016 bytes (4.48 GiB), below the 44.76 GiB working-set ceiling.
- Maximum active memory: 3,342,581,760 bytes (3.11 GiB); peak memory: 3,398,901,760 bytes (3.17 GiB).
- The cached-buffer OOM retry was not needed for this request; its injected failure and successful retry are covered by the runtime test above.

Harness outputs remain under `/var/tmp/a20-single/` on the M2: `response.json`, `control.txt`, `memory.jsonl`, and `server.log`. The invocation used `/var/tmp/a20_server_probe.py` and its existing `/var/tmp/a20site/sitecustomize.py` instrumentation.

## Long generation

Ran `mlx_lm.generate.stream_generate` with the same 4B model for exactly 2,000 generated tokens. The test processor masked EOS tokens so early termination could not shorten the run.

- Result: 2,000 tokens, 8,183 response characters, 50.42 s.
- Instrumentation recorded 8,028 memory events and no asynchronous evaluation failures.
- Maximum observed cache memory: 335,482,880 bytes (320 MiB); maximum active memory: 3,711,918,080 bytes (3.46 GiB); peak memory: 3,717,070,848 bytes (3.46 GiB).
- Output and profile remain under `/var/tmp/alloc-cache-longgen/` on the M2.

## Decode performance and numerical identity

Compared the prior `60f80d2` wheel with the `f9500ec` allocator wheel in the same oMLX venv. Each measurement used the same 2B d64 benchmark, 512-token prefill, 64 generated tokens, one warmup, and one measured pass. Five measurements were captured per wheel.

| Wheel | Decode tokens/s (five runs) | Median |
|---|---|---:|
| Prior (`60f80d2`) | 107.81, 107.89, 108.66, 107.63, 108.04 | 107.89 |
| Allocator (`f9500ec`) | 108.61, 95.85, 108.28, 108.19, 108.07 | 108.19 |

The medians differ by 0.28%. Four of five allocator runs are within the prior wheel's observed range; one allocator run measured 95.85 tokens/s and is retained here rather than excluded. All ten runs produced the same ordered-record digest:

`304d1237fefefa485c53403c826116b713f1467ce9a4ba38026c0841d2501372`

Benchmark JSON files remain under `/var/tmp/alloc-cache-2b-final/` on the M2. The fresh build checkout and wheel were removed after the build and measurements; the wheel hash above and benchmark results preserve their receipts.
