# omarchy-mlx v0.7.40

Prefill no longer runs out of GPU memory on a 16 GB 13-inch M1 while the KV cache grows.

The mlx-lm cache grew in blocks during prefill. Each growth built the larger buffer while the old one was still alive. On the 13-inch M1, Qwen3 4B 4-bit with an 8192-token prompt and the default chunk completed 1 of 4 tries on that path. A 16384-token prompt failed in 2 of 2 tries (chunk 512 and chunk 256).

This release allocates the cache once per prompt and picks the prefill chunk again after the cache is resident. With both changes the 8192-token prompt completed 6 of 6 runs in 2 tries, and the 16384-token prompt completed 2 of 2 runs with the default chunk. Decode speed is unchanged.

## What changed

- **KV cache allocated once per prompt (#390).** `generate_step` tells each full-precision `KVCache` how many tokens the call needs: the prompt plus up to 1024 decode tokens, capped by `max_tokens`. The first growth then allocates the whole buffer. `MLX_OMARCHY_KV_PRESIZE=0` restores the upstream growth.
- **Prefill chunk picked again once the cache is resident (#392).** The first chunk is chosen before the cache exists, so that choice cannot count it. After the first chunk, `generate_step` chooses again from the memory that is left, minus the same 1 GiB workspace the serve budget uses. The new choice is never larger than the old one. An explicit `prefill_step_size` skips it. Pre-sizing alone did not fix the default chunk on the 13-inch M1, so the two changes ship together.
- **Assistant server stops within five seconds on repeated stop signals (#388).** The first signal starts an orderly stop. Later signals are absorbed. The speech worker finishes its current chunk and exits instead of being killed in a GPU job. The runtime file is always removed.
- **Release tooling.** `scripts/verify-release-assets.py` removes its download directories on exit (#389). The HTTP failure test waits for the stop call (#391).

## Measured

13-inch M1, Qwen3 4B 4-bit, greedy, 128 new tokens, default chunk, packaged 0.7.39-2 environment with the patched files laid over it:

| context | old cache path | #390 and #392 |
|---|---|---|
| 8192 | completed 1 of 4 tries | completed 6 of 6 runs; decode 66.9 to 67.6 ms per token in 5 runs and 108.6 ms in one run that ran beside a neural-engine job; prefill 87.1 to 89.0 s; peak after prefill 5.81 GB |
| 16384 | out of device memory in 2 of 2 tries | completed 2 of 2 runs; decode 90.17 and 90.20 ms; prefill 269.45 and 273.76 s; peak after prefill 6.08 GB |

14-inch M2 Max, same model and settings, median decode time per token, 3 runs per row:

| context | #390 off (upstream growth) | #390 on | #392 |
|---|---|---|---|
| 8192 | 20.62 ms | 20.57 ms | 20.64 ms |
| 12288 | 24.42 ms | 24.42 ms | 24.37 ms |
| 16384 | 26.53 ms | 26.52 ms | 26.53 ms |

Token ids are equal in every pair. Peak GPU memory after prefill on the 14-inch M2 Max with the default chunk: 5.84 GB at 8192 tokens, 7.20 GB at 16384, 8.41 GB at 24576 and 9.92 GB at 32768.

Prefill time on the 14-inch M2 Max differs from -3 % to +7 % between rows of the same code. These changes make no prefill speed claim there. The fix is for the 13-inch M1.

## Known issues

- The cache reserve covers `generate_step`. The batched route of the served API (`BatchGenerator`) allocates the cache as before.
- Unchanged from v0.7.39: a process that is killed while it has two GPU submissions in flight logs a GPU page fault and a firmware timeout (linux-aurora #335).
- Unchanged from v0.7.39: the Kokoro primer check logged a kernel GPU timeout on the 16-inch M1 Max in 2 of 2 runs on the Mesa build the lab ran before v0.7.39, and in 0 of 1 runs on the Mesa build that ships since. One run does not show that the timeout is gone. The GPU recovered and the check passed.
