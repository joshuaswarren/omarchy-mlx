# omarchy-mlx v0.7.41

rms_norm at width 2560 now uses the register-path kernel at decode row counts.

Width 2560 is the hidden size of Qwen3 4B. The register-path kernel already served widths 256 and 2048. This release adds width 2560. The path is on by default on the 13-inch M1. On the 14-inch M2 Max it is off by default. Set `MLX_OMARCHY_NORM_APPLE_2560=1` to enable it there. A change that turns the path on by default on the 14-inch M2 Max merged after the commit this release is built from, so it is not in this release.

## What changed

- **rms_norm at width 2560 uses the register-path kernel at decode row counts (#400).** The kernel serves width 2560 at 64 rows or fewer, the way it serves widths 256 and 2048. The fixed read loops are unchanged. A conditional tail covers the extra reads for 2560. The fused and the composed norm kernels share one enable, so both sum a 2560 row in the same order.

## Measured

14-inch M2 Max, Qwen3 4B 4-bit, greedy, 128 new tokens, packaged build of the tagged commit:

- A 519-token prompt, decode: 14.236 ms per token with the path off, and 13.814 ms per token with `MLX_OMARCHY_NORM_APPLE_2560=1`. The path is 0.422 ms (2.97 percent) faster. The token digests are equal.
- A 65536-token context, decode, against the previous release, on one boot: this release read 61.738 ms and 62.234 ms per token. The previous release read 61.732 ms and 62.003 ms per token on its clean runs. The mean is 0.19 percent higher, which is inside the spread of both sets. All five digests are equal. This is noise, not a regression.
- Contexts of 2048, 8192 and 32768 tokens, decode: 15.646, 20.675 and 38.148 ms per token. The earlier build of this release read 15.659, 20.579 and 37.976 ms. The digests and the memory peaks are equal.

Measurement record: commit 8c2a31715.
