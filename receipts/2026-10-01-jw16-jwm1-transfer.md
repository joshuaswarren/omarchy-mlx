# 2026-10-01 — jw16 jwm1-lane transfer: nosched + shared-offset fold candidate measured, under the landing bar, not landed

Lane: Jw16JwmTransfer. Question: do the jwm1 (T8103) proven Mesa wins that are missing on jw16 (T6001) still pay on jw16's current stack, and do they clear the cumulative landing rule (>= +2% on decode d64 or pf2048, outside control range, non-negative elsewhere)?

## Candidate

mesa-1 branch `jw16/transfer-cand` (= deployed `jw16/hwmat-vec2-on` tip `485bd380d85` + cherry-picks):
- `f6faaa917e5` shared-memory offset fold into lload/lstore index immediates (jwm1 h81 win, +1.8% pf512 there)
- `8fa3a53dc98` skip the pre-RA pressure scheduler for compute (jwm1 h84 win, +1.4..1.8% decode there)
- `d128c070740` scope M1 compute codegen defaults to G13 + shader-cache key bits for all AGX env toggles

Conflicts resolved toward deployed jw16 semantics: vec2 fragment loads stay default-ON without the v3 `width_matched` guard; only the cache-key bits were taken in `hk_shader.c`. Build: `/var/tmp/transfer-icd/asahi_icd_transfer.json` -> `libvulkan_asahi.so` sha256 `bd3b9f0c33527d53c1f6936f3e5de710a929e43d65cc26fb92afd54f03ce5339`, driver reports `git-d128c07074`.

## Gates (all green)

- Decode record digests identical on candidate: d64 `c84b3e7a`, d128 `07c515e0`, d256 `c6aabbf0`, d512 `5c120987`; pf records `100a61b62470` on all pf cells.
- Logits gates finite and equal: T512 `f771c4265f88`, T1024 `ce24f3b4ce42`, T2048 `b8c4e14f8f8a`.
- kb (kernel OUTPUT hashes, 23 kernels): 23/23 identical to deployed — fold+nosched are output-bit-exact on jw16. With kill switches (`AGX_SCHED_COMPUTE=1 AGX_LOCAL_FOLD_OFF=1`) the candidate tree also produces byte-identical kernels to the deployed ICD (23/23), so both wins are individually reversible per env.
- Battery digest `dbf704971617` on all 16 runs; every window restored llm-inference with health 200 + completion probe.

## Measured performance (same-boot control vs candidate)

- Cells n=5: d64 105.89 -> 107.92 (+1.92%), d128 105.52 -> 107.75 (+2.11%), d256 104.19 -> 106.30 (+2.03%), d512 99.81 -> 101.53 (+1.72%); pf512 1090.5 -> 1100.2 (+0.89%), pf1024 1236.1 -> 1243.8 (+0.62%), pf2048 1309.4 -> 1314.9 (+0.42%).
- 8-pair interleaved battery: candidate won 8/8 pairs; ctl median 105.56 vs cand 107.66 (+1.99%), paired per-pair median +2.02%.
- Reversed-order paired d64: cand 107.43 vs ctl 105.49 (+1.84%); pooled d64 +1.88%.

## Verdict: NOT LANDED — under the pre-registered bar

d64 +1.88..1.92% and pf2048 +0.42% are positive, outside the control range, and reproducible (17/17 paired windows won), but the rule names d64 or pf2048 at >= +2%: missed by ~0.1pp on d64. d128 (+2.11%) and d256 (+2.03%) clear +2% but are not the named metrics. No falsifier fired (no digest/logits divergence, nothing inside the control range). Deployed ICD stays `9d949d4-vec2`; the candidate stays on pushed branches (`jw16/transfer-cand` on origin and the jw16 clone) for a future GO or bar review.

Context: MesaParity's v3 (+2.1% decode, +2.1% pf2048) was measured on the 2026-09-29 wheel (pf2048 baseline ~1005 tok/s); the deployed 2026-10-01 wheel (DecodeFuse4 + F6 fused chain) already reads ~1309 tok/s pf2048 on the same ICD, so the same Mesa-level prefill win is worth +0.42% today — the bottleneck migrated into the wheel. v3's remaining unique delta (`width_matched` vec2 narrowing) is a no-op for the production f32 width-matched shapes.

## Also fixed in passing

`/var/tmp/pp/pp_cells.sh` could never run as committed: the pf PF-RATE `python -c` line inside the single-quoted `bash -c` payload contained literal single quotes, breaking the outer quoting so the outer `set -u` shell expanded `$c` and aborted before any cell (`pp_cells.sh: line 55: c: unbound variable`). One-line fix: python-payload single quotes replaced with escaped double quotes. Verified by stubbed repro before and after.

## Transferred / rejected ledger (jwm1 -> jw16)

Already on jw16 (no action): hwmat-vec2 (deployed ICD line), qmm FULLN + all wheel-side wins (deployed wheel `0.32.3.dev202610010525+1e7cf5c45`), CCTRACE removal (absent), GDN tiled decode path (per-chip selection; jwm1's GDN_PF vec4 win is untiled-path-only and cannot apply to G13C).
Rejected with measured reasons: cdm-defer (refuted on G13X by MesaLaunchCost: -3.9% + wrong digest), Mesa v2 as a whole (pf -5.6%), nosched+fold COMBO (this receipt: +1.9% d64, under bar), norm load batch (null on jwm1), postra scheduler (null), gemv scale hoist (null), gemv magic convert (never measured on jwm1; equivalent magic dequant rejected in h113), shared-memory vectorize (null), GDN row split / scalar prefetch (null on jwm1), rmsnorm subgroup tail (null), qmm f16 operands (inexact, slower in slow mode), qmm m16 split-K / staged-A (rejected on jwm1 TTFT), qmm fast/slow mode (diagnostic, not steerable).
