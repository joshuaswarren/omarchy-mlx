# G14C: GDN decode kernel choice for Qwen3.8-27B-4bit, speed and NLL accept-not-worse

Registered 2026-10-20, before any G14C measurement or scoring run of this model, by the author. Question: on the 14-inch M2 Max (Vulkan device G14C), does the untiled prefetch kernel (GatedDeltaDecodeBF16Pf, arm U, `MLX_OMARCHY_GDN_DECODE_TILE=0`) decode Qwen3.8-27B-4bit (bf16 A_log, 16 key heads, 48 value heads) faster than the default tiled kernel (GatedDeltaDecodeBF16, arm D), and is it not worse in NLL? On the 16-inch M1 Max (G13C) it is faster by 5.8 percent and the default changed in PR 247. Today the G14C default for this shape is the tiled kernel (the default change is gated to G13C by `gdn_bf16a_untiled_default_part`). No G14C speed number for this model exists before this run.

## Instrument

Wheel: mlx_omarchy-0.32.4.dev202610200902+f334420, built from main f334420781a6ed29f5c5c521f32933497b0bd6ef, sha256 d05326b0835cf47b8a21fc4cf9c1f9f22e8813a00feb1963f67eb79bd4933d8c, libmlx prefix 1a2af63453eeb246. Same tools, byte for byte, as the G13C check: nll_pair.py sha256 abdf0dcdbf3b16eb..., nll_cmp.py 2ff32652d390afce..., text text-AB-e7c41800.txt sha256 e7c41800016e3620.... Speed: bench_decode.py 8220064c5ea3aaaf..., gdn_logits.py 055a90bd58ad9c3a.... Model mlx-community/Qwen3.8-27B-4bit snapshot 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 (model.safetensors.index.json sha256 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8, equal to the copy used on G13C). A prefix is a 512-token prefill window followed by 64 decode steps (batch 1, true next tokens fed); nll = logsumexp(logits) - logits[target].

## Arms and steps

D = no GDN variable. U = MLX_OMARCHY_GDN_DECODE_TILE=0. No other GDN, MLX, HK or AGX variable.
1. Census: D and U, 6 and 10 tokens, dispatch histograms (route guard).
2. Speed: prompts P1 "Explain in plain words how a bicycle works." and P2 "Write a short story about a lighthouse keeper.", 64 tokens, 5 rounds, fresh process each, D and U order alternating by round. Report medians, change of medians, wins, per-round change.
3. NLL: windows D0 (prefixes 0 to 63), U0, D1 (64 to 127), U1, D2 (D again on prefixes 0 to 31). Each ticket is limited to 30 minutes.

## Guards (a failed guard voids the run, and the result says so first)

1. Every process: provenance verified=match for the wheel above, exit code 0, one boot, performance governor.
2. D2 equals D on prefixes 0 to 31 (equal NLL and top1 arrays).
3. Vacuity: U differs from D at some positions.
4. Route: the census shows kernel 455 for D and kernel 480 for U, equal dispatch totals.

## Registered rule

d = nll_U - nll_D per position (positive = untiled is worse). The unit of replication is the prefix: dbar_p = mean of d over the 64 steps of prefix p; D_mean = mean over the 128 prefixes; SE = std(dbar_p, ddof=1) / sqrt(128); interval = D_mean +- 2 SE. NLL ACCEPT-NOT-WORSE if the upper end D_mean + 2 SE is below +0.02 nats per token, and also under the t(127) form (t = 1.979) computed by hand from the logged mean and SE. If the upper end is at or above +0.02, the verdict is NOT ACCEPTED. If the interval is wider than the bound, the verdict is WIDER, the prefixes are extended, and the bound is not widened.
A G14C default change is proposed only if the speed result is faster in at least 4 of 5 rounds on both prompts with a change of medians of at least 2 percent AND the NLL verdict is ACCEPT-NOT-WORSE. Otherwise the G14C default stays tiled and the result is recorded as a measured no.

## Limits

One model, one chip, one text, a 512-token window, two 20-token speed prompts. A pass is a not-worse check, not a quality benchmark. A default change is a separate PR with its own rebuilt wheel and two-build A/B.

## Amendment 2026-10-20, made before any valid G14C run

The first launch used an environment copied from a general serving venv: its mlx-lm (0.31.4.dev0) has no omarchy patches, so the decode ran the generic fallback ops (census: 63511 dispatches for 6 tokens, no GatedDeltaDecode kernel, D and U histograms identical). That launch produced no valid result and its outputs are kept as failed-out-*-unpatched-mlxlm. The environment is now the same lean venv as the G13C checks: mlx-lm 0.31.3 with the omarchy patches (mlx_lm/models/gated_delta.py sha256 prefix 0d5cd194449cd36e, identical to the G13C venv) and the wheel above. Every preflight pins that file hash. Rule, arms, guards and thresholds are unchanged.
