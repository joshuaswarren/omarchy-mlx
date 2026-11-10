# omarchy-mlx v0.7.39

Headline: a decoded token no longer copies the whole KV cache window. The cache writes each new key and value row in place, so the 72 buffer copies per token on Qwen3 4B drop to 0. The copy cost grew with the context, and so does the gain. Measured on the 14-inch M2 Max: decode is 8.2 percent faster at 512 tokens of context, 15.8 percent at 2048, 27.4 percent at 8192 and 36.6 percent at 16384. Tokens stay bit-identical.

## KV window write in place (#386)

During decode, each layer appended the new key and value row to its cache window by first copying the whole window. Qwen3 4B has 36 layers, so a token made 72 copies, and each copy grows with the context. A copy microbenchmark on the 14-inch M2 Max put those 72 copies at 1.33 ms per token at 1024 positions and 14.76 ms at 16640. The append now writes in place when it holds the last handle to the cache. If anything else holds the cache (a reader, a view, a snapshot or the caller), the copy stays. `MLX_OMARCHY_KV_INPLACE=0` forces the copy.

Measured on the 14-inch M2 Max, Qwen3 4B Instruct 4-bit, greedy, 128 decoded tokens, decode only (prefill excluded). A was main before this change, B was main with it; both wheels were built by one recipe, and each run checks that the loaded library matches its wheel. Pairs ran interleaved. Tokens were identical in every run and buffer copies per token went from 72 to 0.

| Context | Pairs | A, ms per token | B, ms per token | Faster |
| --- | --- | --- | --- | --- |
| 512 | 4 | 15.980 | 14.671 | 8.2% |
| 2048 | 4 | 19.127 | 16.083 | 15.8% |
| 4096 | 3 | 21.755 | 17.126 | 21.3% |
| 8192 | 3 | 29.465 | 21.370 | 27.4% |
| 12288 | 3 | 36.895 | 24.988 | 32.3% |
| 16384 | 3 | 42.585 | 26.978 | 36.6% |

A control with both arms on the same wheel at 8192 tokens measured -0.1 percent (pair ratios 0.994 to 1.005).

Measured on the 13-inch M1 (16 GB), same model, same method: tokens identical in every run and copies per token 72 to 0. At 2048 tokens of context decode was 16.8 percent faster (3 pairs, B over A 0.825 to 0.839). At 8192 tokens it was 34.9 percent faster (3 pairs, B over A 0.646 to 0.661, prefill in 512-token chunks). A neural-engine job ran on the same chip during both runs and slowed both arms: stock decode took 174 ms per token at 8192, where a quiet run of an earlier build took about 109 ms. Read these as ratios, not as speeds. The one 2048 pair that ran quiet measured 17.5 percent.

## Other changes

- GatedDelta decode (Qwen3.5 and Qwen3.8): the fused conv, delta and gated-norm route is the default on the 13-inch M1 (#352). On a 2B 4-bit model at 512 tokens it measured +0.5 percent decode, with 251 down to 215 dispatches per token. The install now applies that patch (#361). Other parts keep the stock route.
- GatedDelta decode keeps its recurrent state in bf16 from end to end (#377). `MLX_OMARCHY_GDN_STATE_F32=1` restores the float32 state.
- The default memory limit and garbage-collection ceiling follow the Vulkan heap (#359). A 58 GiB model on a 96 GiB M2 Max no longer drops its buffer cache on every allocation: the first 16-token step of gpt-oss-120b MXFP4 fell from 228 s to 1.2 s.
- Resident dictation starts again on the 16-inch M1 Max (#372) and uses the duration-aware mel for arbitrary audio (#373). The pinned contract now applies to the test fixture only.
- 16-inch M1 Max: the opt-in `MLX_OMARCHY_QMV_SMALL_T` accepts up to 8 rows (#342). Unset, nothing changes.
- Device discovery accepts the Borealis driver ID as an Apple GPU driver (#381). Not tested on Mac mini M4 hardware in this release.
- Custom-kernel translator: bf16 cast rounding, subscript balance, int64 stride reads and chained-cast collapse fixes (#374, #376, #383).

## Known issues

- Killing a process while it has two GPU submissions in flight still logs a GPU page fault and a firmware timeout (linux-aurora #335).
- A prefill of 8192 tokens with the mlx-lm default chunk of 2048 ran out of GPU memory on the 16 GB 13-inch M1, in the build without this change. The measured runs used a 512-token chunk. The cause is not isolated.
