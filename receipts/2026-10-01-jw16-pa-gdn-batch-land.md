# 2026-10-01 — jw16 prefill: GDN round-trip-diet batch kernel (MLX_OMARCHY_GDN_BATCH) landed and deployed

Actor: PaLand. jw16 (M1 Max, T6001), boot 61e5e399 throughout the lane's windows.
Branch `agent/jw16-pa-land` = G4 batch commit 673fd3135 (Jw16PrefillAlgo)
cherry-picked onto origin/main and flipped to default ON (code commit
0aa148382; rebased twice over mid-flight main moves 8528e6eab → 3f9d66c06).
Pre-registered in the private notebook
`entries/Jw16PaLand/20261001T211448Z-jw16-pa-gdn-batch-land.md`; artifacts
under `artifacts/Jw16PaLand/` (w1-base/con/coff, w1b-base/con/con-dd,
w1c-base/con, w4-deployed, each with SHA256SUMS). Continues
Jw16PrefillAlgo: the batch kernel passed every gate there but missed that
lane's self-imposed +2% pf2048 bar; Main overrode the bar (bit-exact +
kill-switched + positive at every length with disjoint ranges lands, the
Mesa transfer ICD ruling). The G3 shuffle-scalar kernel (02b7889f8, refuted
20x) is NOT landed — dropped at the cherry-pick.

## What changed (code commit 0aa148382)

- New shader `gated_delta_prefill_coopmat_batch.comp` (byte-identical to the
  gated branch's file): bit-exact restructure of the coopmat GDN prefill
  kernel — double-buffered chunk staging (one barrier per chunk step instead
  of two, staging loads overlap the matrix work) and a 4-slice wave state
  update (per-step barrier count 48 → 4; the FMA reads only the lane's own
  store-mapped round-trip elements). Per-element arithmetic and its order
  are unchanged.
- Dispatch gate in `primitives.cpp` defaults ON following the shipped
  `kv_direct_enabled()` pattern (unset → enabled); kill switch
  `MLX_OMARCHY_GDN_BATCH=0` restores the exact deployed kernel path.
- 32000 B shared-memory capability gate unchanged; the batch kernel is
  selected only where the coopmat GDN prefill path already fired
  (T ≥ 64, no mask, B=1, coopmat-f32-8, subgroup 32). Tree diff vs main is
  exactly the five G4 files (CMakeLists, compute.cpp/.h, primitives.cpp,
  the new shader); CMakeLists conflict with main's VjpKernels block
  resolved by union (append-only class).

## Gates (all on boot 61e5e399, one contiguous window per set)

Base re-cert first on every set (serving wheel 1e7cf5c45): kb 17/17 hashes,
logits gates T512 `f771c4265f88` / T1024 `ce24f3b4ce42` / T2048 `b8c4e14f8f8a`
all finite, pf records `100a61b62470` in all runs, d64 `c84b3e7af640` — pins
held on the new boot, no other lane's regression absorbed.

- Bit-exactness: kb 17/17 sha_y AND sha_state identical base vs candidate on
  all three wheels (gated-branch wheel b8e54e00b over 8528e6eab, rebased
  wheel b8e54e00b over 3f9d66c06 content, final wheel 0aa148382), and the
  final wheel's hashes identical to the gated branch's wheel. The rebase
  onto the VJP/release commits changed nothing at runtime.
- Kill switch: `MLX_OMARCHY_GDN_BATCH=0` arm — kb 17/17 identical to base,
  gdn_T2048 8.001 ms vs base 7.956 (inert); with the flag on, 7.046–7.073 ms
  (−11 to −12%).
- Logits gates: exact digests above, finite, both arms, every window.
- pf ordered records: `100a61b62470` in all 75 pf runs across the lane
  (15 + 10 + 10 + 15 + 15 + 10), both arms everywhere.
- Decode digests: d64 `c84b3e7af640` pinned both arms everywhere;
  d128 `07c515e0338b` / d256 `c6aabbf0a51d` / d512 `5c120987f0e5` pinned on
  the rebased candidate wheel (w1b-con-dd); d64 + d256 re-pinned on the
  DEPLOYED stack (w4). d64 medians base/cand across windows:
  107.79/107.86, 107.71/107.46 — inside noise, dispatch provably untouched
  (T=1 never selects the prefill kernel).

## Paired prefill (n=5 per cell per arm, same window)

| window | pf512 base→cand (med) | pf1024 | pf2048 |
|---|---|---|---|
| W1 (gated wheel) | 1094.9 → 1100.3 (+0.49%, overlap) | 1245.1 → 1260.8 (+1.26%, disjoint) | 1315.8 → 1326.8 (+0.84%, disjoint) |
| W1b (gated wheel) | 1094.9 → 1105.8 (+0.99%, overlap 0.8 tok/s) | 1242.3 → 1257.7 (+1.24%, touching) | — |
| W1c (final wheel) | 1092.9 → 1103.3 (+0.95%, overlap 1.0 tok/s) | 1241.6 → 1257.9 (+1.31%, disjoint) | 1313.9 → 1328.0 (+1.07%, disjoint) |

Positive medians at every length in every window (9/9); pf1024/pf2048
disjoint everywhere; pf512's min-max ranges overlap by ≤ 6 tok/s (≤ 0.5%)
in this lane's windows — Jw16PrefillAlgo's own w2 window had pf512 disjoint
(+1.12%). Recorded as window noise on the shortest cell, not hidden: the
landing decision is Main's override, quoted above.

## Deploy

- `deploy_wheel.sh` into `/var/tmp/v072-venv-fused`:
  0.32.3.dev202610010525+1e7cf5c45 → **0.32.4.dev202610012238+0aa148382**
  (wheel sha256 `65f1809ef467de82…`). Rollback point
  `/var/tmp/v072-venv-fused.pre-20261001T175018`.
- mlx_lm patch set byte-identical pre/post deploy (generate.py
  `bc4903b5bda7d705…`, qwen3_5.py `88f803754e951f46…`; greedy-prune,
  last-logits, conv-silu, rope-norm, qknorm markers in the same three
  files).
- Deploy-verify (w4, NO env vars — batch kernel active by default):
  pf512 1103.6 / pf1024 1256.9 / pf2048 1324.7 tok/s (+0.98/+1.23/+0.82%
  vs the pre-deploy base in the same boot/window family); records
  `100a61b62470` ×15; d64 + d256 digests pinned; logits gates finite and
  exact; health 200 + authenticated completion finish_reason=length +
  is-active. Boot unchanged.

## Ratios vs macOS window 5 (pf last-logits 1742.68 / 1793.60 / 1826.74 tok/s)

Deployed: pf512 **0.633**, pf1024 **0.701**, pf2048 **0.725**.

## Kill switch

`MLX_OMARCHY_GDN_BATCH=0` — inert-verified at every kb hash and inside base
noise on the final wheel; restores the exact deployed pre-deploy kernel
selection.
