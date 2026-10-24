# Split-K decode attention, value-phase pass 1 on G14C and G13C (2026-10-24)

The split-K decode attention route from the previous change costs 78 to 90 microseconds per call at short caches because it takes two dispatches and each lane chain walks 128 keys. A new pass 1 shortens that chain to 32 keys per lane group and reads the values in 8-byte loads. It takes about 19 microseconds off every call from 128 to 2048 keys on the Apple M2 Max and the Apple M1 Max. Qwen3-4B 4-bit decodes 4.6 percent faster on the Apple M2 Max and 4.7 percent faster on the Apple M1 Max at a 477-token prompt. At 1888 prompt tokens the gain is 2.8 and 2.7 percent.

The variant is the default from the first split-route length up to 2048 keys on the Apple M2 Max (G14C) and the Apple M1 Max (G13C). It stays off on the Apple M1 (G13G), where it loses above 512 keys and decode reads 0.5 percent slower at 477 tokens and 2.4 percent slower at 1888. The rule is the pure function `sdpa_split_pv_max_keys(device_name)` in `device.h`, with `sdpa_split_pv_default(device_name, k_len)` on top. `MLX_OMARCHY_SDPA_SPLIT_PV=0` keeps the old pass 1 and any other non-empty value forces the variant, at every length, whenever the value offset and both value strides are multiples of 4. Otherwise the old pass 1 runs.

The greedy token digest equals the digest of the old pass 1 in every round of every screen on all three chips. The kernel is not bit-equal to the old pass 1: each lane group sums four partial values in a fixed order. The unit test `omarchy_sdpa_split_tests` checks both pass-1 variants against a float64 reference at every shape and checks the fallback for unaligned values.

All tables are tab-separated files in `tables/`. Every number below is read from a log in this directory.

## Machines

The Apple M2 Max machine is a 14-inch model with Vulkan device Apple M2 Max (G14C B1). The Apple M1 Max machine is a 16-inch model (G13C C0). The Apple M1 machine is a 13-inch model (G13G). All three use the same Mesa build and Vulkan driver directory. Kernel strings are named in `tables/`. The Apple M2 Max ran on the egpu kernel 7.1.12-2-20261013.18-egpu-ARCH and the other two on stock kernels (7.1.12-2-12.6-sep-ARCH on the Apple M1 Max, 7.1.12-2-20261013.18-sep-ARCH on the Apple M1). Boot ids are in the first line of each log. Numbers are compared only inside one boot.

## Stack

The tree is the build of commit 3841edc1 (mlx_omarchy 0.32.4.dev202610240043+3841edc, libmlx prefix `2dbe64c6ae0e4811` on all three chips). The rule change in this commit adds only the part rule and its test, so the decode screens ran on the same build with the variant forced by `MLX_OMARCHY_SDPA_SPLIT_PV=1` against the same build with the variable unset, which at that commit meant the old pass 1. The third arm of each screen is the installed ledger wheel 0.32.4.dev202610231001+48908ca, shown for reference. The model is Qwen3-4B 4-bit (index sha256 prefix `f7825defe5865d17`).

## Kernel sweep

Files: `run-logs/microbench-*.log`, table `tables/microbench-chain-us.tsv`. The sweep times single-query grouped-query attention at 128 to 8192 keys in three arms: the fused kernel, the split route, and the split route with the variant. The chain column is the time per call when 36 calls depend on each other, the shape of a decode step.

On the Apple M2 Max the variant saves 18 to 21 microseconds from 128 to 1024 keys and 15 at 2048 keys. It loses 25 microseconds at 4096 keys and 42 at 8192. On the Apple M1 Max it saves 19 to 21 microseconds up to 2048 keys, 6 at 4096 and loses 14 at 8192. On the Apple M1 it saves 15 to 21 microseconds up to 512 keys and loses from 1024 keys on. The limit of 2048 keys on two chips is the last measured length where the variant wins on both. A decode step has 36 attention calls, so 19 microseconds per call is 0.7 milliseconds per token.

## Decode

Files: `run-logs/decode-*.log`, table `tables/decode-qwen3-4b.tsv`. The old-pass-1 arm is the build with the variable unset before this rule existed. Each screen runs five interleaved rounds of three arms, one process per sample, 128 generated tokens, greedy. The prompt is the ledger prompt and its length is nominal: the sentence repeats round(N/18) times. The samples report 443 and 1786 prompt tokens for the screens named 477 and 1888, on all three chips (`prompt_tokens` in `run-logs/decode-*-sample.out`). With 128 generated tokens the cache holds 443 to 571 keys and 1786 to 1914 keys.

At 477 prompt tokens the Apple M2 Max decodes 63.10 tokens per second with the old pass 1 and 66.00 with the variant. The Apple M1 Max decodes 61.40 and 64.30. The Apple M1 decodes 20.50 and 20.40 with the variant forced. At 1888 tokens the Apple M2 Max reads 57.90 and 59.50, the Apple M1 Max 55.00 and 56.50 and the Apple M1 16.40 and 16.00. Prefill speed does not move by more than 3 percent in any screen. The 1786-token screens stay inside the 2048-key limit throughout.

Two of the rules written before the screens were not met. The bar of +3 percent at both lengths holds at 443 tokens (+4.6 and +4.7 percent) but not at 1786 tokens (+2.8 and +2.7 percent), so the claim of this change is the gain at about 450 tokens and a smaller gain near 1800. The third arm (the installed ledger wheel) was to read within 2 percent of the old pass 1 arm. It reads 12 to 92 percent apart, for example 56.30 against 63.10 on the Apple M2 Max at 443 tokens, because that wheel predates the split route. It is reference only and plays no part in the comparison.

## Tests

`doctest/doctest-run-apple-m2-max.log` holds the run of the split test and the fused test on the Apple M2 Max at commit 3841edc1: 8 split cases with 411 assertions and 7 fused cases with 100490 assertions, none failed. The new case in this commit checks the part rule at its limits and runs without a GPU.

## What is not covered

The 7600-token teacher-forced numerics run of the previous change was not repeated for this kernel. The digest equality above covers 128 tokens at two prompt lengths on three chips. The first screens of the variant ran with a rule that was not yet written, so the decode numbers come from forcing the variable. The rule gates the same code path at the same shapes.
