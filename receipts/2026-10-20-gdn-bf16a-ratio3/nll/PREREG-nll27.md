# NLL accept-not-worse check: Qwen3.8-27B-4bit, GDN decode kernel default against MLX_OMARCHY_GDN_DECODE_TILE=0

Registered 2026-10-20, before any scoring run, by the author. Reason: on the 16-inch M1 Max the untiled kernel (GatedDeltaDecodeBF16Pf, arm U) decodes this model 5.75 and 5.76 percent faster than the default (GatedDeltaDecodeBF16, arm D) in 5 of 5 paired rounds on both prompts (~/scratch/w7q/dense27/out-timeP1-1 and out-timeP2-1). The two kernels are not bit-equal: on prompt P1 the generated ids differ. No default changes from a speed run alone when outputs change. This repeats the registered protocol of receipts/2026-10-19-gdn-bf16a-untiled/nll/PREREG.md on a second model.

## Instrument

Same tools, byte for byte: nll_pair.py sha256 abdf0dcdbf3b16eb..., nll_cmp.py 2ff32652d390afce..., text text-AB-e7c41800.txt sha256 e7c41800016e3620... (529411 bytes; 126996 tokens with this model's tokenizer, 73856 needed for 128 prefixes). A prefix is a 512-token prefill window followed by 64 decode steps (batch 1, true next tokens fed); the decode step logits score the true next token, nll = logsumexp(logits) - logits[target]. The prefill route is the same in every arm.

## Arms

D = default (no GDN variable). U = MLX_OMARCHY_GDN_DECODE_TILE=0. D2 = D again on prefixes 0 to 31, to prove the scoring is deterministic. One fresh process per window and arm, same boot, same text, same prefixes. No other GDN, MLX, HK or AGX variable. Windows: prefixes 0 to 63 and 64 to 127 for D and U (each ticket is limited to 30 minutes); nll_cmp concatenates contiguous windows. Order D0, U0, D1, U1, D2.

## Frozen inputs

16-inch Apple M1 Max (G13C), kernel 7.1.12-2-12.6-sep-ARCH, boot 2abdc10a, performance governor, driver prefix 40282811c6e14ef9. Wheel mlx_omarchy-0.32.4.dev202610181348+49e6b3c1, sha256 7654db6b519b221ab5b221e96e1bda3b28c6317b0969b78de2abf3d101f8fe59, libmlx prefix 960f0f358a2aba25 (checked by --expect-libmlx). Model mlx-community/Qwen3.8-27B-4bit snapshot 10c35caafbb80f7dc6a7a432cdd11af10a6d4818.

## Guards (a failed guard voids the run, and the result says so first)

1. nll_pair provenance matches libmlx 960f0f358a2aba25 in every process, exit code 0.
2. D2 on prefixes 0 to 31 is bit-identical to D on the same prefixes: equal NLL and top1 arrays.
3. Vacuity: U differs from D at some positions (nll_cmp VACUITY GUARD nonzero).
4. Route: the census of this wheel (out-census-1, hist-D-10 and hist-U-10) shows kernel 455 for D and kernel 480 for U, 528 dispatches each per 10 tokens. It is not repeated.

## Registered rule

d = nll_U - nll_D per position (positive = untiled is worse). The unit of replication is the prefix: dbar_p = mean of d over the 64 steps of prefix p; D_mean = mean over the 128 prefixes; SE = std(dbar_p, ddof=1) / sqrt(128); interval = D_mean +- 2 SE. ACCEPT-NOT-WORSE if the upper end D_mean + 2 SE is below +0.02 nats per token under the 2 SE form and also under the t(127) form (t = 1.979). If the upper end is at or above +0.02, the verdict is NOT ACCEPTED and U stays opt-in. If the interval is wider than the bound, the verdict is WIDER, the prefixes are extended, and the bound is not widened. The report prints D mean NLL, D_mean, cluster SE, both upper ends, positions that differ, and argmax disagreements.

## Limits

One model, one chip, one text. A pass says the mean per-token NLL on this text is not worse by more than 0.02 nats at the 2 SE level. It is not a quality benchmark, it does not cover long context (512-token window), and it does not cover other models. A pass does not change a default by itself: a default change is a separate step with a pinned rebuilt wheel of current main.
