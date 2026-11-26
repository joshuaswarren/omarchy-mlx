# Prefill cast/copy census on G13: producer-shared casts, an exact per-pass cast memo, and a wall-flat negative

Date: 2026-10-04. Chip: Apple M1 Max (G13C/T6001), kernel 7.1.13-3-2-ARCH,
host redacted per the public-repo rule (the private lab notebook holds the
unredacted transcript, `entries/PrefillCast/20261004T010009Z-*`). Continues
receipts/2026-10-03-prefill-axes/README.md (whose pf512 time breakdown named
`CastBF16F32` 17.9 % of GPU busy as the largest non-qmm prefill item and
proposed producer-fused widening) and the prefill-gap receipt (staged-A
closed negative). Provenance printed beside every measurement
(`scripts/mlx_provenance.py`, verified=match); A/B arms are the SAME diag
wheel selected by env (`MLX_OMARCHY_QMM_CAST_DEDUP`), satisfying the
distinct-builds rule by construction; idle gates (load1 < 0.5, PSI cpu
some avg10 <= 0.05) recorded per row and only green rows counted;
`flock /tmp/m1-gpu.lock` per run inside `gpuwin` windows.

## Census (4-join instrument, 2 reps + warmup; per-prefill = n/9)

Per-op-kind GPU-busy shares, 512-token prefill, bf16-hybrid qwen snapshots:

| kernel | 2B n | 2B share | 9B n | 9B share |
|---|---|---|---|---|
| QmmPrefillCoopmatBF16X32FullNRasterG4 | 1254 (139/pass) | 24.6 % | 1672 (186/pass) | 25.5 % |
| CastBF16F32 | 1998 (222/pass) | 17.8 % | 2448 (272/pass) | 16.7 % |
| CopyGeneralBF16 | 1326 (147/pass) | 13.8 % | 2200 (244/pass) | 18.1 % |
| QmmPrefillCoopmatM16 (both twins) | 420/pass-eq | 7.1 % | 62/pass-eq | 6.0 %+ |
| FastRmsNormBF16 / FastNormGated* | — | 12.1 % | — | 9.8 % |
| GatedDeltaPrefill + ConvDw1d (GDN) | — | 5.4 % | — | 5.5 % |

The 2B table reproduces the prefill-axes anchors exactly (qmm 24.6 %,
cast ~18 %, copy-family ~13.5 %). Dispatch counts are M-independent;
cast per-pass counts grow with model width (222/pass at hidden 2048,
272/pass at hidden 4096).

Cast call-site structure (bindings dataflow from the profiler NDJSON):
every cast input is a producer buffer read by MULTIPLE quantized matmuls —
the input-layernorm output feeds the q/k/v preps, the post-attention norm
feeds gate/up — producers are `RMSNorm` (`FastRmsNormBF16`),
`RMSNormScaled`, and prior `QuantizedMatmul` outputs. 9 additional tiny
(4 KB) casts per pass feed compiled-tape `FusedChainF32` nodes.

Copy-family attribution (next lane, NOT widening): `Contiguous` layout
copies of qmm/Silu outputs (~1.5 MB each), repeatedly-read `Full`-init
constants (36 KB), RMSNorm outputs — all bf16->bf16; exact elimination of
these requires strided-read kernels, not cast elimination.

## Lever: env-gated per-pass cast memo (`MLX_OMARCHY_QMM_CAST_DEDUP=1`, default OFF)

The qmm prefill cast block memoizes the bf16->f32 widening per evaluator
pass (LRU-2 keyed encoder/array-id/buffer/offset/bytes, strong refs;
cleared at every completion join, so an entry can never outlive the pass
that wrote it — a doctest pins one-pass three-consumer bit-identity AND
post-join freshness). Bit-exact by construction: identical cast values,
one dispatch instead of two/three.

Census delta (profiler, same stream): 2B casts 1998 -> 1188 (222 -> 132
per pass), busy 26.38 -> 15.48 ms (-41 %); 9B casts 2448 -> 1368 (272 ->
152 per pass), busy 38.76 -> 21.29 ms (-45 %). qmm dispatch counts
unchanged in both — the lever engages exactly at the shared-producer
sites and nowhere else.

## A/B: bit-exact everywhere, wall-flat everywhere

5 alternating pairs ctl-first per cell (green-gate medians; green-pair
counts in parens; every pair's digests identical in EVERY row, green or
not; pair-0 full-logits digests shown, arms always byte-equal):

| model | cell | ctl tok/s | dedup tok/s | ratio | pair-0 digest |
|---|---|---|---|---|---|
| 2B | pf512 (5,5) | 1434.3 | 1434.3 | 1.0000 | `21fabf57cc83299e` |
| 2B | pf1024 (5,5) | 1457.4 | 1457.2 | 0.9999 | `36151ab8fa09e9f7` |
| 4B | pf512 (7,8) | 535.0 | 535.1 | 1.0002 | `20d847e89f8cc5d4` |
| 4B | pf1024 (5,6) | 513.5 | 513.2 | 0.9994 | `fb2c02e44d7c8653` |
| 9B | pf512 (4,5) | 310.3 | 310.5 | 1.0006 | `a0c7edfe6520dccc` |
| 9B | pf1024 (1,2) | 306.8 | 306.4 | 0.9987 | `5ad0b6eb2cce5097` |

No cell reaches the 1.5 % landing bar; the full spread is -0.13 % ..
+0.06 %. The 2B pair-0 digest equals the prefill-axes cross-context 2B
pin on the LANDED wheel, i.e. this branch's default-off path reproduces
main's generation digest from a different build. Ambient-load caveats: a
competing lane compiled through two windows; red rows were rerun or
excluded by the recorded gates — the 9B pf1024 cell holds 1-2 green
pairs (its verdict is unambiguous: flat, digests exact).

Verdict: NOT landed. The lever is exact and does exactly what it claims
(-40..45 % of cast dispatches and busy), and that moves prefill wall by
nothing on G13C at M in {512, 1024}: at prefill the wall is not
GPU-busy-bound (the PrefillGap instrument finding: the wall is dominated
by dependent-dispatch turnover, which scales with dispatch count — the
memo removes 90-120 of ~940 dispatches/pass, ~0.3 % of wall). The
literal producer-fused f32 x (norm writes f32 directly) remains blocked
by construction in eager dispatch: retyping a producer's graph output
changes downstream rounding points, so digests cannot be preserved; only
this memo fragment is exact.

## Standing suites (hardware, this branch)

omarchy_matmul_family_tests 24/24 cases incl. the new "qmm prefill cast
dedup is bit-identical and never stale" (82.9 M assertions, bit-equality
vs the uncached route at qwen layer shapes); omarchy_runtime_tests,
omarchy_fused_chain_tests, omarchy_primitive_tests all SUCCESS;
omarchy_capability_sim_tests profile matrix 5/5 rc=0.

## Artifacts and provenance

Diag wheel `0.32.4.dev202610040123+diag.226602fe3` built on the G13C
host (glslc 2026.3); backend code byte-identical to test commit
`d4fc6708d` (test-only diff after the wheel stamp). Model shard sha256s:
2B `b0d5de68…`, 4B `2a73c6c2…`, 9B `a68b8755…`+`b0a770bf…`. A/B JSONLs,
census NDJSONs, analysis script, and SHA256SUMS in the private lab
`artifacts/PrefillCast/cast-census/`. Source: branch
`agent/prefillcast-producer-f32` (code), receipt + ledger merged to main.

## Closed axes (do not re-run)

Per-pass bf16->f32 x cast memoization at prefill M (this receipt): exact,
wall-flat on G13C, env `MLX_OMARCHY_QMM_CAST_DEDUP` retained for
experiments. The remaining prefill-parity levers are unchanged from the
prior receipts: dependent-dispatch turnover (dispatch-count/fusion
levers), honeykrisp coopmat lowering, and the non-widening copy family
(attribution above).
