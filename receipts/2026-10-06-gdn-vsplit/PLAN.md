# GDN HOIST pass-B Dv split (MLX_OMARCHY_GDN_VSPLIT) — pre-registered plan

Pre-registered 2026-10-06 before implementation. Worktree `agent/GdnVsplit` @ origin/main
d428ade1d. Offline analysis only; jwm1 (T8103 G13G, Honeykrisp) runs belong to w71.
Companion private notebook entry:
`~/.local/share/apple-silicon-lab/entries/GdnVsplit/20261006T0332Z-omp-studio-local-gdn-vsplit.md`.

## 1. Decomposition: what is Dv-independent, what is not

State S[dk, dv] = [B=1, Hv, Dv=128, Dk=128] f32. Pass B groups Dv into 8-column slabs; one
simdgroup (32 lanes) owns one slab. Today local_size = (32,4,1): 4 sg/WG, 32 Dv columns/WG,
grid (Hv, Dv/32, 1) — **4 workgroups per head already**, not one. The occupancy problem is
per-WG shared bytes (27,904 B declared; the 32,000 B figure in the ticket is the batch
kernel's gate constant `kGdnCoopmatBatchSharedBytes`), which still allows only 1 WG per
32 KiB G13G core.

| Piece | Matrix shape | Dv-dependence | Where today |
|---|---|---|---|
| gamma/beta chunk prefix sums | [8 tokens] | independent | recomputed per sg (`subgroupInclusiveAdd`) |
| K.K^T, Q.K^T (kkt/qkt) | [8 tok, 8 tok] | independent | pass A (`gated_delta_prefill_kktqkt.comp`), f32 snap, one 128-float tile per sg |
| N = tril(diag(beta)KK^T)Gamma, Tinv = (I-N)(I+N^2+N^4+N^6), msum, eye | [8 tok, 8 tok] | independent | recomputed per sg in pass B (5 MMAs + round trips) |
| m2 = (gamma beta K).S, tmp = (gamma Q).S | [8 tok, 8 dv] | per-slab | pass B loop 1, 16 MMAs over kk |
| delta = Tinv.(diag(beta)V - m2) | [8 tok, 8 dv] | per-slab | pass B |
| out/y = QKt_tri.delta + tmp | [8 tok, 8 dv] | per-slab | pass B |
| state walk kd = (dec.K)^T.delta, state = state.gamC + kd | [8 dk, 8 dv] x 16 kk | per-slab | pass B waves |

Every shared array in pass B is indexed `[...][sg]` — **zero cross-sg data flow**; the WG
barriers order only same-sg reuse. The split therefore re-packs the same 16 sg per head
(Dv/8) into more, smaller workgroups. Total sg work per head is unchanged, so the per-sg
recompute of the Dv-independent N/Tinv chain does NOT grow (16 recomputes per head per
chunk before and after). Note the task framing corrected: the split does not shorten the
serial chunk walk (every sg still walks all ceil(T/8) chunks); the lever is occupancy —
more WGs resident per core to overlap the MMA/LDS-bound walk.

Decision on Tinv/kkt recomputation: the split keeps pass B exactly as is (per-sg recompute;
kkt/qkt read from pass A's snap). Sharing the token matrices across slabs would need a
third dispatch + cross-WG ordering (i.e. re-association risk) for ~5 of 25 MMAs — rejected.

## 2. Bit-exactness audit, per matmul

An output element y[t, dv] and state[dk, dv] are produced by exactly one sg. Claim: the
per-element accumulation order along K is a function of the sg-local tile sequence only,
which the split does not touch.

- m2/tmp: element (r, dv) = sum over kk = 0..15 of an 8x8x8 MMA (ascending kk, K=8 per
  MMA). MMA shape and order unchanged; N (dv width) never enters a reduction.
- delta, out_t: single 8x8x8 MMAs (out_t accumulator-initialized with tmp). Unchanged.
- kd waves: kd_kk = MMA(kgc_kk column-major, delta) — one MMA per kk (K = 8 tokens); state
  update per element `state = state * gamC + kd` in lanes. Unchanged.
- kkt/qkt: values are pass A's f32 tiles (exact round trip). The split only remaps WHICH
  snapshot slot a sg reads: `slot = slab >> 2`, `sub = slab & 3` with `slab = gy*SGPW+sg`.
  At SGPW=4 this reduces to the current `slot = gy, sub = sg` exactly.
- gamma scan: per-sg over its own 32 lanes. Unchanged.
- No coopmat tile shape changes anywhere (all 8x8x8); N-shrink affects only how many
  tiles a WG carries, not any reduction. **Verdict: bit-identical by construction.**
  The one place a mistake could bite is the snap slot arithmetic — covered by the host
  model and the jwm1 hash gate below.

## 3. Change shape

- New shader `shaders/gated_delta_prefill_coopmat_hoist_vsplit.comp` (copy of the hoist
  shader; original untouched) with `#define SGPW` from `-DGDN_VSPLIT={2,4}`:
  local_size_y = SGPW, shared arrays [4] -> [SGPW], `dv0 = (gy*SGPW+sg)*8`, snap base
  slab-mapped. Two pipelines: `gated_delta_prefill_coopmat_hoist_vsplit{2,4}_bf16`.
- New ComputeKernels `GatedDeltaPrefillCoopmatHoistVsplit{2,4}BF16` (append-only, before
  `Count`), compute.cpp includes+cases, CMakeLists omarchy_shader entries,
  scripts/check_gdn_variants.sh expected set.
- primitives.cpp hoist branch: `MLX_OMARCHY_GDN_VSPLIT` parsed once (static); 2 or 4
  selects the kernel and pass-B grid (Hv, Dv/(8*SGPW), 1). Everything else — including
  pass A and the default route — byte-identical when the var is unset/0/other.

## 4. Shared-memory and occupancy budget (declared bytes)

Per sg: s_state 4 KiB + staging (kgc_all 1 KiB, s_kg/s_qg 512 B each, v_s/rt_a/rt_b
256 B each, gamma/beta 32 B each) = 6,976 B.

| SGPW | per-WG shared | WGs per 32 KiB G13G core | grid (Hv, gy, 1) at Dv=128 |
|---|---|---|---|
| 4 (current) | 27,904 B | 1 | (16, 4, 1) |
| 2 | 13,952 B | 2 | (16, 8, 1) |
| 1 | 6,976 B | 4 (27,904 B total) | (16, 16, 1) |

Register estimate: unchanged per lane (same sg-local live set; SGPW only shrinks the WG).
Occupancy beyond shared (simd slots) is a hardware question — falsifier below.

## 5. Offline gates (this session)

1. `glslc -O --target-env=vulkan1.3 -DGDN_VSPLIT={2,4}` clean + `spirv-val` clean on both
   variants (same flags as CMake omarchy_shader), plus scripts/check_gdn_variants.sh.
2. Host model (fp32, exact per-element order): split schedule vs unsplit schedule bitwise
   equal for T in {512, 519, 1024} with non-zero initial state — validates the
   decomposition and the snap slot mapping; reference (fp64 token-matrix form) within
   fp32 tolerance.
3. test_gdn_maskless_correctness.cpp: the reported namespace defect is ALREADY FIXED on
   main (04f825676) — verified, no edit needed. ninja that target on a build dir if one
   exists — none exists on this box, so the C++ evidence is the -fsyntax-only below.

## 6. On-hardware gates (w71, pre-registered, not run here)

- gdn_hash.py: default (HOIST) vs MLX_OMARCHY_GDN_VSPLIT=2 and =4 — identical y AND state
  hashes at T=64 (route untouched), 128, 512, including a non-zero initial state case.
- Pinned digests must hold: d64 eee1cf9635d6d4eb, d128 0756351401f5b3fb,
  d256 393a1cf303e9f362, pf512 509c19201275dfcc (env unset).
- gdn_micro2.py kernel micro at T=512/1024, HOIST vs VSPLIT2 vs VSPLIT4 (disjoint medians,
  3+ reps, quiet host): prediction ~-20..-40% kernel time at T=512 if the occupancy-shadow
  estimate transfers (2 WGs/core = the shadow's 2.3x upper bound is NOT expected; LDS
  issue pressure grows with resident WGs). Falsifier: <5% kernel-time change at T=512 on
  G13G kills the candidate; any digest/hash mismatch kills it outright.
- pf512 e2e qwen38 protocol as the final gate before any default flip (not requested here).

## 7. Kill switch

`MLX_OMARCHY_GDN_VSPLIT` unset, 0, or any value not in {2,4} selects the shipped HOIST
route exactly (same kernel enum, same grid, same bytes).
