# Split-K decode attention starts at 320 keys on G14C (2026-10-24)

On the Apple M2 Max (G14C) the split-K decode attention route now starts at 320 keys instead of 448. Qwen3-4B 4-bit decodes 3.9 percent faster at a 256-token prompt, where the cache holds 256 to 384 keys: 63.50 against 66.00 tokens per second. The greedy token digest is equal in every round.

The value-phase pass 1 from the previous change made the split route cheaper than the fused kernel at shorter caches. In the kernel sweep of that change the chain time per call at 320 keys is 89.7 microseconds for the fused kernel and 64.3 for the split route with the variant. At 256 keys the fused kernel still wins by 2 microseconds (65.7 against 68.0), so the start is the first measured length where the split route wins the chain shape of a decode step. In the independent shape the fused kernel is 2.6 microseconds faster at 320 keys (55.6 against 58.2). Decode is a chain, so the rule follows the chain.

The rule is `sdpa_split_min_keys(device_name)` in `device.h`. The Apple M1 Max (G13C) already starts at 320 keys and the Apple M1 (G13G) at 128, so only the G14C value changes. `MLX_OMARCHY_SDPA_DECODE_SPLIT=0` still forces the fused kernel at every length.

Table: `tables/decode-apple-m2-max-256-tokens.tsv`. Log: `run-logs/decode-apple-m2-max-256.log`.

## Machine and stack

The machine is a 14-inch Apple M2 Max (Vulkan device Apple M2 Max, G14C B1) on the egpu kernel 7.1.12-2-20261013.18-egpu-ARCH. The boot id is in the start line of the log. The tree is the build of commit 7104ec94 (mlx_omarchy 0.32.4.dev202610240500+7104ec9, libmlx prefix `6a268ec29aae0b63`). The `libmlx=` field in the log header is the installed ledger wheel (prefix `a9738db49ca96906`), not the tree. The model is Qwen3-4B 4-bit (index sha256 prefix `f7825defe5865d17`).

## Screen

Four arms ran five interleaved rounds, one process per sample, 128 generated tokens, greedy. The sample output reports 256 prompt tokens (`run-logs/decode-apple-m2-max-256-sample.out`); the nominal target of 270 was the sentence repeated round(270/18) times.

The `rule` arm is the tree with default routes. The `fused` arm is the same tree with the split route off. The `main` arm is the previous tree with the value-phase variant forced, which at 256 to 384 keys on G14C still runs the fused kernel because the old start was 448 keys. Medians: 66.00 for `rule`, 63.50 for `fused` and 63.50 for `main`. The two fused arms agree to 0.01, so the tree difference does not matter. The installed ledger wheel reads 63.60 and is shown for reference only.

The bar written before the screen was +3 percent over three or more interleaved rounds with equal digests. The rule arm reads +3.9 percent over `main` and the digest is equal in all 20 samples. Prefill speed is unchanged to 0.1 percent.

## What is not covered

Only the Apple M2 Max was screened, because it is the only chip whose rule changes. The gain applies to caches of 320 to 447 keys. A 128-token decode that starts at a prompt of P tokens covers P to P+128 keys, so it is unchanged only when P+128 is below 320 or P is at least 448. A 7600-token numerics run was not repeated: the kernels are the same ones that ran in the previous change, only the length where they start differs.
