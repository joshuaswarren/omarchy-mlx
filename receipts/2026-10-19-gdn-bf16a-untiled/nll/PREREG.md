# NLL accept-not-worse check: Qwen3.6-35B-A3B-4bit, GDN decode kernel default against MLX_OMARCHY_GDN_DECODE_TILE=0

Registered 2026-10-19, before any scoring run. Requested after the speed and numerics run (untiled-vs-tiled/): the untiled kernel (GatedDeltaDecodeBF16Pf) is 8.1 percent faster in decode and not bit-equal to the default (GatedDeltaDecodeBF16). Rule: no default flip from a speed run alone when outputs change; run the paired-NLL accept-not-worse protocol of the row-reduce default (2 SE bound). If the check passes not-worse, the default changes; if not, the route stays opt-in.

## Instrument

The decode path under test is m == 1, so the instrument scores through decode steps. nll_pair.py (sha256 abdf0dcd...) does a teacher-forced decode of a real model on a fixed text: a 512-token prefill window, then 64 decode steps (batch 1) with the true next tokens fed; each step's logits score the true next token: nll = logsumexp(logits) - logits[target]. The prefill route is the same in every arm. nll_cmp.py (sha256 2ff32652...) pairs two runs. This is the instrument of the gather_qmm routing studies; it replaces the gateway-scored prose passages of the row-reduce check because the gateway path needs the 62-layer cluster and this change is a one-model decode kernel. The rule below is that check's rule with prefixes as the unit of replication.

## Arms

D = default (no GDN variable). U = MLX_OMARCHY_GDN_DECODE_TILE=0. D2 = D again, to prove the scoring is deterministic. One process per arm, fresh process, same boot, same text, same prefixes. No other GDN, MLX, HK or AGX variable.

## Frozen inputs

16-inch Apple M1 Max (G13C), kernel 7.1.12-2-12.6-sep-ARCH, driver prefix 40282811c6e14ef9. Wheel mlx_omarchy-0.32.4.dev202610181348+49e6b3c1 (sha256 7654db6b519b221ab5b221e96e1bda3b28c6317b0969b78de2abf3d101f8fe59), libmlx prefix 960f0f358a2aba25, venv of that build. Model mlx-community/Qwen3.6-35B-A3B-4bit snapshot 38740b847e4cb78f352aba30aa41c76e08e6eb46. Text text-AB-e7c41800.txt (sha256 e7c41800...). Prefixes: windows 0 to 127 (128 prefixes, 8192 scored positions per arm). Timing is not measured by this run.

## Guards (a failed guard voids the run, and the result says so first)

1. nll_pair provenance matches the wheel libmlx prefix (--expect-libmlx 960f0f358a2aba25) in every process, exit code 0.
2. D and D2 bit-identical: equal NLL arrays and equal top1 arrays.
3. Vacuity: U differs from D at some positions (nll_cmp VACUITY GUARD nonzero), so the route switch changed the computation.
4. The census of the same wheel (untiled-vs-tiled/hist-*.txt) already shows the kernel swap 455 to 480; it is not repeated here.

## Registered rule

d = nll_U - nll_D per position (positive = untiled is worse). The unit of replication is the prefix: dbar_p = mean of d over the 64 steps of prefix p; D_mean = mean over prefixes; SE = std(dbar_p, ddof=1) / sqrt(128); interval = D_mean +- 2 SE. Also reported: the t(127) form (t = 1.979), the position-level SE, and the argmax disagreements.
ACCEPT-NOT-WORSE if the upper end D_mean + 2 SE is below +0.02 nats per token under both the 2 SE and the t forms. ACCEPT-BETTER is the same check with the whole interval below 0 and is not required. If the upper end is at or above +0.02, the verdict is NOT ACCEPTED and the route stays opt-in. A wider interval (2 SE above +0.02 with the mean below it) is reported as INCONCLUSIVE; it does not accept. nll_cmp.py also prints its own two-sided verdict at the +-0.02 bound; it is reported, not used for the decision.
A pass says the mean per-token NLL on this text and model is not worse by more than 0.02 nats at the 2 SE level. It is not a quality benchmark and does not cover other models.

## Ordering and limits

Order D, U, D2, sequential under flock /tmp/m1-gpu.lock, one boot. One model, one chip, one text. The 20-token prompts of the speed run are not used here. No guard-ticket receipt.
