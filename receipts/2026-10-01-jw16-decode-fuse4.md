# 2026-10-01 — jw16 decode: F4 (GDN q/k norm into the conv epilogue) + F3 + F6 landed together as one cumulative default-on wheel

Actor: DecodeFuse4. jw16 (M1 Max, T6001), boot 36539f5a throughout. Branch
`agent/jw16-decode-fuse4` off `agent/jw16-decode-fuse3` (63e909139), rebased
onto origin/main (duplicates of main's F3 lineage dropped; tree diff vs the
gated wheel = main's unrelated file additions only). Pre-registered in the
private notebook `entries/Jw16DecodeFuse4/20261001T014220Z-jw16-decode-fuse4.md`
with the exact F4 math contract; artifacts under `artifacts/Jw16DecodeFuse4/`.

## What changed (tip 58405acc2)

- **F4** (be4dc8893 pre-rebase 646fdae5f): `gdn_conv_decode.comp` gains a
  flags-bit-1 epilogue — per-workgroup two-half 128-row rms-norm*scale over
  the stored bf16 silu values, reproducing `fast_norm_gated.comp` mode 1
  exactly (strides 64→1 per half = the mode-1 tree association; the stride-128
  step only ever adds +0.0; mean = block/128.0; `inversesqrt(mean + eps)`;
  `value * (norm * 1.0)` precise — SPIR-V disassembly verified the association
  and its NoContraction pins; normed = widen(bf16 store); scale = widen(bf16
  store(push f32)); out = bf16 store(normed*scale)). `GdnConvUpdate` carries
  the qk tuple (`qk_key_dim/qk_scale_q/qk_scale_k/qk_eps`); the omarchy
  wrapper refuses non-fuseable geometry loudly (count % 256, key_dim % 256 —
  refusal BEFORE dispatch: the in-loop shared-memory barriers deadlock
  otherwise); the fallback composes the exact model ops (per-head reshape +
  `rms_norm_scaled`, scalar wrapped at the array dtype off-bf16).
  Removes 36 dispatches + 18 barriers/token on the GDN layers.
- **F3** (from DecodeFuse3, `agent/jw16-decode-fuse3` @ 63e909139): the
  wrapper fence + `MLX_OMARCHY_ROPE_NORM_FUSE` gate + python patcher +
  rope-norm iso v3 — measured +0.68% solo (below the old +1% solo bar),
  landed per the new combined ruling.
- **F6** (c46523ff3 pre-rebase 2a86cd6d3): ONE-LINE widening of the
  fused_chain SliceUpdate-pair planner from f16-only to {f16, bf16}. All
  downstream machinery was already bf16-capable (rope kv_window, GEMV
  sum_window, geometry checks, abort unwinds); the merged-pair dispatch
  kernel stays f16-only, so a refused bf16 pair falls back to the two
  ordinary SliceUpdate dispatches (exact). **Measured result: inert on this
  stream** — the pairs now form (`[kv-plan] pair N: classifiable=1`) but the
  direct plans classify `kinds=0,0` (neither side's producer/geometry
  satisfies the direct-write conditions), so 0 of the combined gain is F6.
  Receipt = the kv-trace in the W3 artifacts; kept as a verified-inert
  prerequisite for the direct-write path (kill switch MLX_OMARCHY_KV_DIRECT=0).
- Both env gates default **ON** at the land commit (kill switch =0 runs the
  exact eager chains): `MLX_OMARCHY_ROPE_NORM_FUSE`, `MLX_OMARCHY_GDN_QKNORM_FUSE`.

## Gates (wheel 0.32.3.dev202610010211+1248d88fe; same content as the landed tip)

- **Iso**: qknorm 12/12 PASS (9 bf16 legs bit-identical incl. inf/nan/±0/±88/
  1e30/33σ cases, b2 batch, kd512/kd256 geometry, scale stress; both fence
  legs raise; f16-fallback leg exact after the dtype-wrap fix).
  rope-norm v3 25/25 PASS (F3 path on the combined wheel).
- **Whole-model bitcheck**: 3084 rows identical serve vs candidate venvs.
- **Combined paired cells** (n=5 interleaved, same window, ctl2 re-arm;
  W3 2026-10-01T02:2xZ, boot 36539f5a, restore health_ok=1
  probe_finish=length active=active):
  d64 ctl 101.40 / ctl2 101.33 / cand 103.85 → **+2.42%**, outside ctl
  min-max; d128 101.52/103.80 (**+2.25%**); d256 99.89/102.24 (**+2.35%**);
  d512 95.74/98.15 (**+2.52%**). All four digests at the pins in every arm
  (c84b3e7af640 / 07c515e0338b / c6aabbf0a51d / 5c120987f0e5); kill-switch
  arm inert (101.28, pin).
- **F4-only isolation** (W2, wheel 646fdae5f): d64 +1.98% (103.41 vs ctl
  101.40/ctl2 101.31), d128 +1.96%, d256 +1.71%, d512 +1.91% — F4 alone
  clears the old +1% bar; the combined ruling's +2% bar is met with F3.
- **Race battery**: `gated_battery.sh 8` pairs, cand = fuse4-venv (X=1) with
  both flags, ctl = serving. **16/16 runs `ordered_records_sha256 =
  dbf704971617`** (contract held both arms, no divergence); per-run medians
  cand 103.40-103.88 vs ctl 100.84-101.28 — disjoint ranges; every window
  restored health_ok=1 probe_finish=length.
- **Full cells** (run-linux-cells.sh, candidate venv, both flags):
  d64 103.92 c84b3e7af640 / d128 103.77 07c515e0338b / d256 102.71
  c6aabbf0a51d / d512 98.04 5c120987f0e5 — all four whole-word full-cell
  digests at the production pins; **pf512 records 100a61b6247096f5 on all
  5 runs**; logits gates **f771c4265f88 (T=512) / ce24f3b4ce42 (T=1024) /
  b8c4e14f8f8a (T=2048)** all finite=True; restore health 200 + completion
  finish_reason=length + is-active.

## Ratios vs macOS window-5 (179.72/179.08/178.72/177.02)

Combined candidate arm (W3 window): d64 103.85/179.72 = 0.578,
d128 103.80/179.08 = 0.580, d256 102.24/178.72 = 0.572,
d512 98.15/177.02 = 0.554 (was 0.564/0.566/0.559/0.541 at the
pre-F4 serving stack).

## Rollback

- Runtime: `MLX_OMARCHY_ROPE_NORM_FUSE=0 MLX_OMARCHY_GDN_QKNORM_FUSE=0`
  (exact eager chains, digest-pinned) or `MLX_OMARCHY_KV_DIRECT=0`;
  deploy backup venv swap + restart for the whole wheel.
- Code: revert the land commit(s); F3's fence commits remain independently
  revertable.
