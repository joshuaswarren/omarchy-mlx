# Numerics gate

A kernel or route change is judged against an **independent reference**, never against the kernel it replaces. The old kernel is a noisy implementation too: agreement with it measures how far two roundings drift apart, not which one is closer to the truth.

## Reference

Use one of:

- an fp64 or fp32-dequantized CPU forward of the same weights and text;
- the macOS (Metal) MLX implementation on the same quantized weights and text.

Run the same text through the old kernel and the new kernel with identical harness code. Validate the harness first: S=1 teacher-forced PPL of the old kernel must match the prefill PPL of the same text (a decode-loop bug shows up as a wildly different PPL).

## Checks

A change passes when its error against the reference is equal to or better than the old kernel's error against the same reference:

1. **Per-op error:** fp32/fp64 output and state error for captured operands is no worse than the deployed op.
2. **Token choices:** teacher-forced top-1 agreement with the reference is at least the old kernel's agreement (within the 95% binomial band at the sample size). Flips whose top-2 margin is at most 2 bf16 ULP at the magnitude of the top logit do not count as failures: they are near ties. For normal bf16 values the spacing is 2^(floor(log2(|x|))-7), for example 0.125 in [16, 32) and 0.0625 in [8, 16); the boundary is inclusive.
3. **Perplexity:** route-sensitive S=1 PPL against the reference is within run-to-run noise of the old kernel's deviation (report both numbers).

Bit agreement between the old and the new kernel is **not** a bar. A reorder-only change (a different fp32 summation order, a lane-parallel reduction) cannot match the old kernel bit for bit, and in routed-MoE models a one-ULP logit change can flip an expert choice and cascade through the layers. Report the old-vs-new agreement for information.

Record in the receipt: reference used, text and token count, the table (kernel x agreement vs reference, PPL vs reference, mean |dlogprob| vs reference, flips and their margins), and the harness validation.

## Precedents

- v0.7.26 9B fused GDN route (old gate, new kernel vs old kernel): per-op fp64 identical; teacher-forced agreement 99.51% (5,095/5,120); six first-divergence gaps of one ULP; S=1 PPL -0.072%. The v0.7.27 default remains ON; MLX_OMARCHY_GDN_RAW_REPEAT=0 selects the composed route. An independent G13 run measured 99.53% and -0.079%.
- 2026-10-06 DeepSeek-Coder-V2-Lite routed-MoE gather (subgroup kernel, 2.0-2.1x decode), 527-token S=1 text, Metal reference: deployed kernel 96.02% agreement with the reference, PPL +0.245%, mean |dlogprob| 0.0749, 21 flips; Sub kernel 97.15%, PPL +0.416%, mean |dlogprob| 0.0727, 15 flips; old-vs-new agreement 97.34%. Sub agrees with the reference more often and was merged. Receipt: receipts/2026-10-06-moe-layer-decode/.

Analyzer code: tools/dfuse/bf16_ulp.py, free_run_gaps.py, and compare_tf9.py.
