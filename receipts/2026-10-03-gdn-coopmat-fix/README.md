# 2026-10-03 — GDN coopmat prefill: exact-input acceptance on current main; FLT_MIN gate floor + wave-barrier hardening; captured-operand fixtures

Lane: GdnFix (worker), hosts jwm1 (iteration) / jw16 (deploy perf) / jw14m2 (27B e2e).
Wheels: baseline `0.32.4.dev202610031219+3700b88` (origin/main), fix
`0.32.4.dev202610031228+e8113bc` (this lane), distinct stamps asserted.

## TL;DR

The release-blocking fused-coopmat GDN prefill defect (layer0 state err
0.0298, layer12 head-29 NaNs, 27B `[0]*8` logits) was measured on diag wheel
`0.32.3.dev202610022039+diag.83eb57a`, which predates `2d1ee300e` (exact
Neumann inverse `(I-N)(I+N^2+N^4+N^6)`, overflow-form removal) and
`77cc02811` (decay/gamma domains). On current origin/main the fused route
already passes every exact-input bar: replaying the captured 27B operands,
layer0 fused state max-err vs fp64 is **1.4265e-5** (bar 5e-5), layer12
**1.7184e-6** (bar 1e-5), zero NaN/inf, batch≡base bit-identical, y within
bf16 noise of the scan route. This lane (a) proves that acceptance, (b)
hardens the two remaining shader defects — the 1e-6 gate floor (real gates
reach 5.9e-11; floor mis-states decay ~1.7e4, and g→0 would NaN on
`inf - inf`) and two spec-UB shared-memory races in the batch kernel's
state-update waves — and (c) adds the captured operands as checked-in
fixtures so the fp64 sweep can never again pass while real inputs fail.

## Exact-input replay (jwm1, T6001... no: T8103 G13G, `/tmp/m1-gpu.lock`)

Captured operands: GdnMaskless 2026-10-03T10:53Z M2 capture, layer0 sha
`7e9325fb…`, layer12 sha `704b5990…`, q/k/v `[1,351,48,128]` bf16-origin,
g f32, zero initial state. Reference: per-token gated-delta rule in float64
(same recurrence as `test_gdn_maskless_correctness.cpp`); validated by
reproducing the scan arms' M2 numbers digit-for-digit (layer0
2.5773428e-5, layer12 5.5866747e-7).

| arm | layer0 state | layer12 state | NaN | y vs fp64 | y vs scan |
|---|---|---|---|---|---|
| scan (NO_COOPMAT=1) | 2.577e-5 | 5.587e-7 | 0 | 0.0151 / 6.1e-5 | — |
| masked all-True | 2.577e-5 | 5.587e-7 | 0 | same | — |
| fused batch (default) | 1.4265e-5 | 1.7184e-6 | 0 | same | 6.1e-5 / 1.22e-4 |
| fused base (GDN_BATCH=0) | 1.4265e-5 | 1.7184e-6 | 0 | same | 6.1e-5 / 1.22e-4 |

Fix wheel vs baseline wheel: bitwise-identical states on every arm (the
clamp error is common-mode across prefix differences, so its differential
residue sits below f32 state resolution on these captures; 1-ulp y change).
The floor's value is faithfulness for g→0 and sub-floor gates: layer12
head 29 has 18 tokens with g < 1e-6 (min 5.89362e-11).

## Shader changes (`e8113bc72`)

1. `log(max(g, 1e-6))` → `log(max(g, 1.17549435e-38))` in both coopmat
   prefill shaders. All exponents stay log-differences ≤ 0 (max |log| 87.3,
   exp ≤ 1); g = 0 (softplus underflow) now decays fully instead of
   NaN-ing on `inf - inf`.
2. Batch kernel wave loop: `subgroupBarrier()` between the `kd`
   `coopMatStore` and the FMA reads of `rt_b` (the store's lane-to-element
   mapping is implementation-defined; the FMA reads cross-lane elements),
   and a wave-close barrier before the next wave's `kgc_all` staging
   overwrites pending `coopMatLoad` reads. Both are UB races that have not
   bitten on Mesa lockstep hardware; they remain latent on any scheduler
   change.

## Tests (all green on jwm1, 2/2 cases, 152/152 assertions, three route configs)

- Sweep: T = 63/64/65/96/352 × rep 1/2/3, maskless + all-True, fp64
  reference (pre-existing case, kept).
- NEW fixtures `overlay/tests/omarchy/fixtures/gdn_coopmat/`: two-head
  (NaN head 29 + control head 8) subsets of the real layer0/layer12
  captures (largest file 359 KB, total 2.1 MB), bf16 u16 bit patterns +
  f32 g, manifest with source sha256. New case `GDN coopmat prefill
  matches fp64 on captured 27B operands`: state bars 5e-5 (layer0) /
  1e-5 (layer12), y over-quanta (8 bf16 quanta) count zero, NaN/inf zero,
  maskless + masked.
- Route configs: default (batch fused), `MLX_OMARCHY_GDN_BATCH=0` (base
  fused), `MLX_OMARCHY_NO_COOPMAT_GDN=1` (scan) — all 152/152.

## End-to-end (jwm1, fix wheel + mlx-lm 0.32.0 + ssm-maskless patch)

128-token chat prompt, greedy 8 tokens, arms by env only:

| model | fused-batch | fused-base | scan | masked |
|---|---|---|---|---|
| 2B | `a5620574d9fcec9d` | same | same | same |
| 9B | `a5620574d9fcec9d` | same | same | same |

ids `[90700, 8340, 25, 271, 16, 13, 220, 2972]` in every arm. The 54-token
variant of the same prompt (below the 64-token fused gate) is also
three-arm identical, as expected — both rows are informative only because
the 128-token prompt fires the coopmat route (two distinct fused kernel
binaries agree with scan and masked).

## Status / remaining

- [x] exact-input replay both routes, batch=1/0, maskless/all-True
- [x] sweep doctest + captured fixtures in-repo, three route configs
- [x] 2B e2e parity (jwm1)
- [ ] jw16 same-cell prefill 512/1024/2048 vs deployed baseline
      1103.6/1256.9/1324.7 (awaiting jw16 window)
- [ ] M2 27B prompt600 greedy ids `[1596,1144,310,5707,310,1156,13,2570]` +
      finite logits; 9B/2B digests (M2 window after 15:00Z)
- [ ] serve A/B with ssm-maskless patch (27B TTFT n>=5, padded 2-seq
      exactness, zero-CPU spot check)
- [ ] deploy/ship decision (v0.7.22)
