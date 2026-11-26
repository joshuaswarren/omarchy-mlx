# GLM DSA sparse prefill and cross-layer indexer sharing

Assessment: 2026-10-23. Source review and existing receipts only; no new build,
model run, download, dependency update, or configuration change.

## Decision

Track mlx-lm [#1997](https://github.com/ml-explore/mlx-lm/pull/1997) and
[#1410](https://github.com/ml-explore/mlx-lm/pull/1410) as a **model-level
compatibility and numerical-qualification opportunity**, not an established
Linux speedup or a demonstrated fix for GLM-4.5-Air prefill cost.

The earlier support assessment did not verify adoption in the deployed mlx-lm
model code. The [core lock](../mlx.lock) pins MLX 0.32.3 at `9c3d35571ac4`, not
mlx-lm; its date cannot prove whether either model-library change is present.
A family name or a separately installed mlx-lm version is also insufficient to
identify the loaded architecture, indexer schedule, or cache implementation.

## What changes upstream

| Item | Mechanism | Qualification consequence |
|---|---|---|
| #1997: sparse DSA prefill | Gather selected latent/rotary key-cache rows and attend over indexer top-k keys instead of masking over all keys; compute indexer scores in blocks. The proposal depends on #1410. | Relevant to DSA long-context prefill, not evidence of a decode gain. The recorded upstream comparison reports 98.68% top-1 agreement and nonzero KLD: this is not proof of exact arithmetic parity with our reference. |
| #1410: cross-layer indexer sharing | Schedule full versus shared indexer layers, pass previously selected top-k indices to shared layers, and omit an unused indexer cache there. | Changes cache ownership and layer-to-layer state. The described default retains ordinary DeepSeek-v3.2 behavior; reported GLM-5.2 conversion/generation checks do not establish full-model numerical equivalence or local adoption. |

For GLM-4.5-Air, first establish whether its actual loaded model path uses these
DSA/indexer mechanisms. Neither the existing assessment nor the receipts below
supply that proof. Do not transfer GLM-5.2 or another architecture's reported
prefill improvement to a GLM-4.5-Air row.

## What the existing receipts actually establish

The historical [DSA family receipt at `f42d188b8`](https://github.com/joshuaswarren/omarchy-mlx/blob/f42d188b8/receipts/2026-10-04-omlx-family-glm-dsa/README.md)
covers graph-level indexer/top-k/sparse-attention work against composed
references, not qualification of these two mlx-lm changes. Its README still
contains pending-M2 wording; use the recorded CSVs for the completed run:

- [Eight M2 prefill shapes](https://github.com/joshuaswarren/omarchy-mlx/blob/f42d188b8/receipts/2026-10-04-omlx-family-glm-dsa/parity_speed_m2.csv)
  have zero candidate-versus-composed output difference, including the recorded
  pooled-ratio case. Timing ratios are elapsed-time ratios, not speedup factors;
  every recorded candidate median is higher than the composed median.
- [Six larger M2 shapes](https://github.com/joshuaswarren/omarchy-mlx/blob/f42d188b8/receipts/2026-10-04-omlx-family-glm-dsa/large_parity_m2.csv)
  likewise have zero difference and finite outputs. They do not show a win:
  `H=32, L=1024, K=4096` records 73.53 ms versus 45.97 ms, about 1.60 times
  the elapsed time. These are operator/shape observations, not full-model TTFT.
- The [custom-kernel inventory](custom-kernel-inventory.md) describes bounded
  translation and optional sparse-attention fallbacks. A family name in that
  inventory is not coverage of #1997's complete graph or #1410's shared-cache
  lifecycle. Its dated corpus is not a current full-model qualification.

This assessment has no matched GLM-4.5-Air before/after model receipt. Existing
Qwen model receipts and vocabulary-tail tests cannot substitute for one.

## Track impact

- **Vulkan MLX:** qualify the selected-key gather, indexer score blocks, masks,
  reductions, and sparse attention at the actual shapes/dtypes; separately
  verify model-library cache semantics. No backend port is delivered here.
- **Cluster serving:** relevant only when the served model takes this path.
  Prefix reuse, cache fork/reset, and layer ownership need explicit checks;
  neither upstream result establishes a distributed speedup.
- **ANE:** no direct ANE kernel or ABI change is identified by these proposals.
  That is a scope statement, not an ANE compatibility claim.
- **Mesa:** these proposals do not supply a driver change. Existing shape
  results do not qualify new shader/compiler/dtype combinations.
- **macOS parity:** advancing MLX or mlx-lm can change the native reference
  denominator. Freeze both libraries and the model configuration, or establish
  a new matched reference before comparing Linux and macOS numbers.

## Gates before any later implementation or performance claim

1. Record the actual model/configuration, mlx-lm source revision, MLX pin,
   indexer-sharing schedule, dtype, and cache mode; prove the target takes DSA.
2. Check selected indices and causal/chunk offsets, then shared-layer cache
   ownership through prefix reuse, fork, reset, and continuation. Shape-only
   operator parity does not cover that lifecycle.
3. Compare model outputs and logits against the frozen reference. Any relaxed
   numerical bound needs explicit acceptance; upstream top-1 agreement is not
   permission to replace an exact-output gate.
4. In a separately authorized window, measure prefill/TTFT and unchanged decode
   with matched hardware, versions, prompts, and cache conditions. Separate
   score/gather/attention costs before choosing a backend optimization.

These are future acceptance conditions, not a request to start another run.
No GLM parity or speedup claim is made by this documentation change.
