# Jw16DecodeNorm — fresh census on the deployed Fuse6 stack, NORM_SUBTREE norm kernel, Q4_WV2 decode GEMVs (REFUTED)

## Identity
- Lane: Jw16DecodeNorm (worker, omp-studio-local; drove jw16 via ssh `jw16mbp1-linux`).
- Notebook entry: `entries/Jw16DecodeNorm/20261001T054855Z-jw16-decode-norm-gemv.md`.
- Branch: `agent/jw16-decode-norm` (off `origin/main` e3bb6501f; pushed; carries the diag cherry-pins + Q4_WV2 cherry-pick + NORM_SUBTREE kernel).
- Wheels: `diag.748ffd3a0` (census) in `/var/tmp/norm-venv`; `748ffd3a0` (release) in `/var/tmp/norm-venv2`.
- Boats: serving `/var/tmp/v072-venv-fused` = `1e7cf5c45` (Fuse6 land) UNTOUCHED throughout.
- Boats: `origin/main` UNTOUCHED (this receipt commit is the only main-facing change).

## Acceptance criterion
Lane assignment: "A further landed+deployed decode gain with all pins, OR measured per-lever refutations with the new census, and ratios vs macOS window 5." This lane completes on the REFUTATION path; both candidate levers (norm in-kernel latency, GEMV weight-load transport) are bit-exact artifacts that realize ~0% decode gain on the deployed stack.

## W1 census (deployed 127.68.3.dev202610010525+1e7cf5c45, boot 48cb8bc1)
- Barrier reason (slope d64→d128 / 320 tokens): 240.0 tracker decisions/token = **220.0 emitted RAW barriers/token + 20.0 skipped** (waw=0 war=0 everywhere, tracker RAW-tight). Fuse5 measured 252/232/20 on the Fuse4 stack; F6's land design delta is EXACTLY the −12 barriers/token the F6 gate windows recorded.
- Named class census (5 passes × 128 tokens, profiled shares only — NO timing claims):
  - `QmmVecQ4MultiSubgroupBF16` 92.0/tok (35.7% share, mean 11.6 µs) — the GEMV multi group.
  - `FastRmsNormBF16` 50.6/tok (19.8%, 11.7 µs) — the lone-largest remaining headroom class.
  - `GatedDeltaDecodeBF16` 18.4 (6.9%, 11.2); `FastNormGatedOnlyBF16` 18.6 (6.7%, 10.9); `GdnConvDecodeBF16` 18.4 (6.7%, 10.9) — the GDN trio.
  - `QmmVecGreedyBF16` 8.2/tok = the 8-stage greedy lm_head (4.9%, mean 18.0 µs → 147 µs/tok; macOS word kernel equivalent ~232 GB/s vs Linux 254.6 GB/s measured here).
  - `FastRopeNormBF16` 12.4 (4.5%, 10.8) — F3 fused rope+norm.
  - `SdpaDecodeNativeBF16Hd256` 6.1 (2.2%, 10.9); `QmmVecQ4MultiOutgateBF16` 6.1 (2.2%, 10.9).
  - Tail `~9.4/tok` (TakeBF16 2.1@24.3, QmmPrefillCoopmat 1.4@23.3, CastBF16F32 2.4@13.1, ElementwiseU32 1.5@19.1, CopyGeneralU32 1.0@25.2, TakeU32 1.0@23.2).
  - SliceUpdate: GONE from decode (F6). Total dispatches 247/tok. `[kv-plan]` lines = 18 (warmup-prefill refusals; DecodeFuse6-documented).
- Per-class cost model (count × (12.2 µs boundary + profiled kernel mean)), ranked top: GEMV-multi 2188 µs/tok, RMSNorm 1209, GDN trio ~1283, FastRopeNorm 285, greedy head 247, SDPA+outgate 282, tail ~307.

## W2 NORM_SUBTREE (subgroup shuffle tail on strides 16..1; 4 barriers/row removed) — REFUTED
- Bit-exact: `dg_bitcheck.py` 3084/3084 rows identical with `MLX_OMARCHY_NORM_SUBTREE=1` vs serving, including all 1440 norm controls. Subgroup shuffle pairing reproduces the shared tree's `(t, t+s)` association.
- Carts: d64 ctl 105.84 / cand 105.64 / ctl2 105.49 / ksoff 105.88 — cand inside ctl noise. d128 105.55/105.54; d256 104.28/104.18; d512 99.89/99.63. Digests at all four pins both arms.
- Realized gain: ≈0% (d64 −0.19%, d128 −0.01%, d256 −0.10%, d512 −0.26%). The 4-barrier/row saving hides entirely under the drain/turnover structure — consistent with c3's load-prefetch lesson and the corrected census's "drain hides kernel time". The env flag default stays OFF; bit-exact kernel stays on the pushed branch as a receipt of the design.

## W3 Q4_WV2 for decode (MLX_OMARCHY_Q4_WV2=1) — REFUTED
- Bit-exact: 0 differing bitcheck rows vs serving (excl version). `gemv_shapes.py` decode rows (M=1) GB/s, serving vs `cand env-off` vs `cand +Q4_WV2=1`:

  | shape             | serve | cand-off | +WV2   | Δ wv2 |
  |-------------------|-------|----------|--------|-------|
  | gate_up 6144×2048 | 174.7 | 175.5    | 172.3  | −1.4% |
  | down  2048×6144   | 184.7 | 183.2    | 183.5  | −0.6% |
  | qkv   4096×2048   | 156.2 | 155.1    | 153.9  | −1.5% |
  | qkvz  8192×2048   | 180.1 | 182.9    | 183.4  | +1.8% |
  | out   2048×2048   | 118.0 | 118.1    | 117.4  | −0.5% |
  | lm_head 248320×2048 | 254.6 | 252.9 | 247.7  | −2.7% |

  Decode GEMVs are latency/x-re-read bound, not weight-load-instruction bound; the uvec2 pair staging (per Jw16PrefillGap4 L1) wins on prefill M=512/2048 amortized but does not transfer to M=1 decode. The cells (d64 105.40 vs 105.69; d256 104.18 vs 104.05) confirm the micro A/B at model rate. Env flag stays default-OFF.

## Ratios vs macOS window 5 (deployed, this boot's control arms)
| length | Linux (ctl, this boot) | macOS window-5 | ratio |
|--------|-----------------------:|---------------:|------:|
| d64    | 105.84                 | 179.72         | 0.589 |
| d128   | 105.55                 | 179.08         | 0.590 |
| d256   | 104.28                 | 178.72         | 0.584 |
| d512   | 99.89                  | 177.02         | 0.564 |

(Deployed Fuse6 deploy-verify ratios: 0.588 / 0.590 / 0.584 / 0.562 — consistent with this boot within ±0.5%.)

## What is not this lane
1. The barrier-elision lever (Jw16BarrierElide) is CLOSED at both driver (rw10) and MLX tracker (this census: RAW-only, 0 avoidable on this stack); ORDER-level lever (level-batched emission ~42 levels/token vs 220 emitted barriers/token, ceiling ~2.25 ms/token) is driver/queue territory.
2. Norm-prologue fusion into a GEMV dispatch — REFUTED in Jw16DecodeGap2 W6 (−28.7%, bit-exact).
3. Q4 GEMV rows-per-workgroup widening — REFUTED in Jw16DecodeGap2 W7/W7b (slower; + a `QMM_VEC_Q4_WORD` `ROWS_PER_SLOT>1` landmine).

## Cleanup
- jw16 scratch `/var/tmp/{norm-build,norm-venv,norm-venv2,norm}` removed via a copied script (the branch push preserved the source + scripts).
- Serving stack (`v072-venv-fused` 1e7cf5c45) and `origin/main` (e3bb6501f, Fuse6) UNTOUCHED apart from this receipt commit.
- All artifacts mirrored under `~/.local/share/apple-silicon-lab/artifacts/Jw16DecodeNorm/{census,norm,gemv}/` with SHA256SUMS.

## Acknowledgement
jrj authored the opposite-direction velocity attribution that made the "realized ~0%" measured "every class at the ~11 µs floor" call land. Bit-exact artifacts on the agent branch (NORM_SUBTREE kernel + Q4_WV2 dispatch gate) keep the door open for a future driver-side enabler (level-batched emission, decode-shape uvec4 weight staging for M>1, or a different kernel-side rewrite) to recover their latent class-ceiling value.