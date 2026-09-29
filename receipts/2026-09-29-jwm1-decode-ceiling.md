# 2026-09-29 jwm1 (M1, T8103): where the decode gap to macOS sits, and what does not move it

State: live stack = main 560a64424 + q/k + conv mlx-lm patches + soundfile; decode64/128/256 = 40.6 / 39.8 / 38.9 tok/s vs macOS 49.36 / 49.26 / 49.33 (0.82 / 0.81 / 0.79).

Measured (jwm1 M1, `tools/q4-bw-bench`, frozen base copy of the decode GEMV, 2B decode shapes, bf16 mix; two passes):
- Probes: copy 58.9 GB/s, read 59.7 GB/s (wall).
- In-chain (24 dependent 4-dispatch layer sets): 49.8 GB/s = ~84% of the read probe. One large dispatch (lm_head, 286 MB) reaches 56 GB/s (95%).
- Variants: xpack equal to base; unroll and wg128 are ~26% worse in-chain (36.8 GB/s) although neutral in isolated timing; all bit-exact (56/56 equality records, 0 mismatches).
- Isolated per-shape wall GB/s from this tool are dominated by per-submission overhead (0.2-0.5 ms) and must not be read as bandwidth.

Levers tested today on the live stack (each bit-identical unless noted, decode64 medians):
- gated dependency barriers `MLX_OMARCHY_GATED_BARRIERS=1`: 1.002x (18% of barriers skipped).
- CPU-cluster frequency floor (min = max): 1.0015x decode, Parakeet -0.7% (M1 Max's +4.3% does not transfer).
- shipped: q/k rms_norm_scaled +2.0%, GDN conv +1.5-2.0% (patches in this repo).
- not shipped: fused gated norm (deviation from the composed chain cut from ~36% to ~1e-5 of elements; composed sigmoid carries its own deterministic 1-ulp errors on ~26% of inputs, fused kernel matches the exact chain; needs a table-based sigmoid to be bit-identical).

Arithmetic (inference, from the profile and the bench): the ~973 MB per-token read set at ~49.8 GB/s is ~19.5 ms of the ~24.6 ms token; the ~344 non-qmm dispatches take the remaining ~5 ms
(removing a dispatch has measured ~5-6 us each). macOS is 20.3 ms/token total. Reaching it needs BOTH qmm near the roof (~95%, -2 ms) and roughly half the small-op time (-2.5 ms);
the existing GEMV variants do not deliver the first, and the exact fusions left are worth ~1% each.

Full data and entries (private notebook, not in this repo): jwm1-parity H1, H2, H8, H10, H11, H12, H15, H17.
