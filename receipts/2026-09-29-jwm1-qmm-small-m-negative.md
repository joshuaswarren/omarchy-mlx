# jwm1 (T8103): small-M and large-M Q4 qmm schedule probes - all exact, none faster (2026-09-29)

Hardware: Apple M1 (G13G), stock Honeykrisp, Qwen3.8-2B MLX 4-bit (bf16 x/scales/out), live venv wheel 1daa1ad5d as control. Branch with all experiments (env-gated, default off): `agent/qmm-coopmat-prefetch`.

## Motivation

The jwm1 TTFT cell is 0.196 s vs macOS 0.1248 s for 11-17 token prompts. A one-forward curve (fresh cache, `mx.eval(model(ids))`, median of 10) shows the cause: 1 token = 48 ms, every M >= 2 costs ~145-175 ms up to M~16, then ~3.7 ms/token (M=512: 1974 ms). Extrapolated intercept of the linear region ~95 ms. A GPU profile of a 4-token forward puts 75% of GPU time in `QmmPrefillCoopmatM16BF16X32` (187 calls per forward, mean 526 us). Per call, `mx.quantized_matmul` at M = 2..16 costs 560-760 us on the byte-heavy shapes (up_proj 132 us at M=1) and 260-300 us even for N=16: the floor is inside the coopmat kernel (profiled: N=16, K=2048, M=4 kernel mean 343-428 us).

## Probes (each: pre-registered, output hash sweep of 96 shape x M cases against the live route, then per-call timing; all interleaved control/candidate in one GPU-lock hold)

| Probe | Change | Exact (96/96) | Result |
|---|---|---|---|
| H35 | hoist per-K-chunk global loads (packed words, A tiles) in the X_F32 coopmat kernel | yes | 5-55% slower (up_proj M=4 565 -> 870 us); N=16 floor unchanged |
| H36 | STEP_K 16 -> 64 in the 16-row kernel | yes | 1.6-1.8x slower; N=16 floor unchanged |
| H37 | route M <= 16 to the staged-A bf16 coopmat kernel (no cast) | yes | slower at M=16 (923 vs 565 us), N=16 floor unchanged |
| H38 | new scalar-FMA small-M kernel (`qmm_fma_smallm.comp`, bf16 io, 16 masked rows, 1 col/lane); fused and precise accumulate | yes, both | 2.3x slower at M=16 |
| H39 | shared-memory x staging, COLS 1/2/4 | yes, all | COLS=1 best but still 1.4x slower at M=16 (817 vs 567 us); COLS 2/4 worse (register pressure) |
| H40 | bf16 twin of the f16 scalar-FMA prefill kernel (`qmm_fma.comp -DBF16_IO`) for M >= 32 | yes | 1.5-2x slower than coopmat (up_proj M=512 20.4 vs 11.0 ms; 0.63 vs 1.18 TFLOP/s) |

Findings worth keeping:

- The ascending-k scalar f32 chain (`acc += x*w`, weight dequantized as `scale*float(nibble)+bias`) is **bit-identical** to the coopmat chain for bf16 activations (fused and NoContraction variants), for N in {16, 512, 2048, 6144}, K in {2048, 6144}, M in {2..512}. FMA kernels are exact drop-ins; on the M1 they are just slower than the coopmat kernel with the schedules tried.
- The per-call floor of the coopmat kernel is not per-step load latency, barrier count or step size (H35-H37); it is inside the kernel's sequential K chain in one workgroup per 32-column tile.
- The GPU governor is not the cause of the compute gap: after >= 0.5 s of continuous load the mel stage and a 1024^3 matmul pulse are constant (19.7-20.3 ms and 4.9 ms up to 8 s of load). The GPU's firmware DVFS parameters are in the device tree (`apple,perf-tgt-utilization` 0x55 etc.); an idle gap >= 50 ms costs ~+10 ms on the first work and only near-continuous heavy load holds the fast state (light keepalive and CPU-side pinning do not).

Not shipped. No route left in this family except a non-bit-identical multi-row GEMV kernel (numerics decision) or ISA-level work on the coopmat kernel.
