# 2026-10-05 — GDN prefill chunk-walk: where the 87 ms pf512 gap goes, and the barrier-diet candidate

Lane: GdnPrefill2 (worker). Hosts: the T6021 machine for build/doctest/A-B
(kernel 7.1.13-3-1-ARCH, Mesa 26.3.0-devel git-7faf04c065, Vulkan 1.4.359,
glslc 2026.3, Apple M2 Max 38-core, public-host placeholder); jwm1 (G13G)
numbers are w71's h283/h290/h292 via Main. Private notebook:
`entries/GdnPrefill2/20261005T054400Z-jw14m2-gdn-chunkwalk-ablation.md`
(pre-registered before any M2 work); artifacts
`artifacts/GdnPrefill2/2026-10-05-chunkwalk/` (SHA256SUMS sealed).
Branch: `agent/GdnPrefill2` @ baed8f456. Wheel under test:
0.32.4.dev202610050436+ca195e6 (lane build; provenance verified=match).

## The gap, decomposed

| quantity | Linux jwm1 (G13G, 8-core) | Linux T6021 (38-core) | macOS pair (Metal) |
|---|---|---|---|
| pf512 forward (2B, 18 layers) | 1205 ms | — | 1119 ms |
| kernel-only GDN 18-call chain, T=512 | 7.85-7.90 ms/call | 2.138 ms/call | 3.29 ms/call |
| glue share (compute_g+sigmoid) | 2.2% | ~6% of full entry | — |
| per 8-token chunk | ~123 us | 33.4 us | ~51 us (busy window) |
| T=1024 kernel-only | 15.99 ms/call | 4.14 ms/call | 6.51 ms/call |

Sources: jwm1 column w71 h292/h283 (this session's ground truth); T6021
column this lane (gdn_micro2.py, 3 gated repeats, load1 0.04-0.09, logs in
artifacts); macOS column H290.

Headlines:
1. Kernel-only is 97.8% of the jwm1 gap; glue is 2.2% — profiling the glue
   or the surrounding ops cannot pay.
2. The chunk loop is ALREADY fused: one dispatch per call
   (GatedDeltaPrefillCoopmatBatchBF16, 64 workgroups x 128 threads) with an
   in-kernel chunk walk. There is no per-chunk dispatch to remove.
3. The per-chunk cost is occupancy/latency-bound, NOT math-bound: the same
   kernel is 3.7x slower per call on the 8-core part than the 38-core part,
   while total MMA work per chunk is ~1 us. The G14C number is already at or
   under the busy-window macOS Metal pair. THE 87 ms IS G13G-SPECIFIC.
4. Why Vulkan pays what Metal does not: Metal's gated_delta_fused_chunk
   (C=8) keeps the whole chunk body in simdgroup-register tiles and needs
   ONE threadgroup barrier per chunk. Vulkan cooperative matrices have no
   lane-addressable elements, so every operand-role change is a shared round
   trip: the shipped batch kernel's chunk body costs ~38 workgroup barriers
   + 16 subgroup barriers + ~25 8x8x8 MMAs + ~35 shared round trips, all
   serial inside one workgroup. At jwm1's ~2.4 us per barrier-class event
   that is the 123 us.
5. Stage ablation (T6021, lane-only stubs, T=1024 fast-op, ship 72.79 ms/18):
   NO_LOOPK 62.47 (−14%), NO_LOOPS 59.85 (−18%), NO_NEUMANN 71.65 (−1.6%),
   NO_DELTA 73.00, NO_OUT 73.14 (both ±0), NO_STATE 45.73 (−37%), SKELETON
   19.68 (−73%). On the occupancy-rich part only MMA-heavy stages show; the
   middle chain (Neumann/delta/out) is fully hidden. On G13G the barrier
   chain dominates instead (one workgroup per core cannot overlap the serial
   RTs), which is what the diet and the follow-up levers target.

## What landed (agent/GdnPrefill2, env-gated, default OFF)

`gated_delta_prefill_coopmat_batch2.comp` +
`MLX_OMARCHY_GDN_BATCH2=1` (kill switch: unset or =0 restores shipped
dispatch; shipped kernel bytes untouched — new enum
GatedDeltaPrefillCoopmatBatch2BF16, additive plumbing only). Barrier diet:
- three workgroup barriers removed where the next chain barrier provably
  orders the shared-memory hazard (N loads / a1 store; msum load / qkt
  store path; delta-tinv loads / kd store);
- the v_s staging folds under the tinv-load barrier;
- the msum round trip rides rt_a so the N^4 load and the msum store do not
  share an array (fourth barrier removed);
- one kernel-start barrier covers the state restoration (the shipped
  kernel's per-chunk gamma barrier used to).

Net ~38 -> ~34 workgroup barriers per chunk; per-element arithmetic and
per-accumulator order unchanged.

### Bisect evidence (the interesting negative)

A gamma/beta subgroup-shuffle restructure (distribute the prefix by
subgroupShuffle instead of shared+barrier) was tried first and REVERTED: the
captured-operand doctest proved a 32-lane subgroupInclusiveAdd with trailing
zeros is NOT bit-equal to the shipped 8-active-lane scan on this stack
(batch2a shuffle-only produced byte-identical wrong-vs-ship hashes as full
batch2; the diet was innocent). Durable rule: subgroup scan values depend on
the active-lane count; never distribute GDN prefix sums by shuffling a
widened scan.

### Gates (all on the T6021 lane build)

- Captured-operand doctest: BIT-IDENTICAL 4/4 (fixtures gdn_coopmat
  layer0+layer12, out AND state, byte-compare, same wheel OFF vs ON).
- 2B greedy digests, same wheel OFF vs ON: d64 9ffb5e0a859a6a6c ==,
  d512 9789a28bbbb5723a == (token-identical).
- T6021 perf: NEUTRAL (37.51/37.89 vs 37.63/37.51 ms T=512 across 3
  interleaved pairs) — expected; the lever targets the G13G barrier regime.
- jwm1 bundle sent to Main (w71): branch sha baed8f456, A/B protocol,
  pins d64 eee1cf96 / d128 07563514 / d256 393a1cf3 / pf512 509c1920 must
  hold exactly with =1. Default flips ON only on >= 1.5% T=512 kernel-only
  there + exact pins.

## Not pursued (recorded so nobody re-derives)

- Two-pass scan (NO_COOPMAT_GDN=1): 8.2x worse than coopmat on T6021 — dead
  end for prefill T>=64.
- C=16 chunks: halves the chunk count but changes chunk-form rounding —
  digests move; requires an owner re-pin under the numerics gate. The math
  (Tinv = (I-N)(I+N^2+...+N^14) needs 4 MMAs vs 3) stays available if the
  owner ever re-pins.
- Cross-slice N/Tinv sharing via grid barrier (the middle chain is identical
  across the 4 Dv-slices of a head): real 3/4 saving on the chain but needs
  grid sync — GridBarrier/GduBar territory, next ticket if the diet shows
  the G13G barrier model holds.

## Reproduce

```bash
# T6021 lane (any host with the golden kit):
/var/tmp/golden-clone-tree.sh /var/tmp/gdn2-wheel
/var/tmp/golden-clone-venv.sh /var/tmp/gdn2-venv
cd /var/tmp/gdn2-wheel && git fetch origin agent/GdnPrefill2 && git checkout FETCH_HEAD
bash golden_rebuild.sh   # then pip install dist/*.whl into the cloned venv
# A/B (NEVER nest flock inside a gpu-turn ticket; gpu-turn holds the lock):
<venv>/bin/python gdn_micro2.py                    # ship
MLX_OMARCHY_GDN_BATCH2=1 <venv>/bin/python gdn_micro2.py
```

## Addendum (same session): the kkt/qkt hoist lands its gates (branch agent/GdnPrefill2-b)

Pass A `gated_delta_prefill_kktqkt.comp` (grid Hv x Dv/32 x chunks): the
state-independent K.K^T / Q.K^T accumulators for EVERY chunk in parallel,
each workgroup running the shipped loop-1 sequence per simdgroup (ascending
kk, same staged tiles and row/column loads — the redundant per-slice
computation is what preserves the accumulation order). Tiles ride the Snap
binding as f32 (exact round trip): layout (chunk*Hv + head)*(Dv/32)*512 +
sg*128, kkt at +0, qkt at +64. Pass B
`gated_delta_prefill_coopmat_hoist.comp`: the batch2 recurrence with the
loop-1 kkt/qkt MMAs replaced by tile loads. `MLX_OMARCHY_GDN_HOIST=1`
opt-in (default OFF; kill switch unset/=0).

Gates (T6021 lane build @ agent/GdnPrefill2-b 14558faaa):
- Determinism: T=64 hoist x3 identical, hash == ship (the first cut had a
  scratch-stride bug — 128 not 512 f32 per (chunk,head,wgY) block — that
  let concurrent pass-A workgroups overwrite each other's tiles: T=64 hoist
  hashes differed run to run; T<=16 scheduled stably and hid it. The
  bit-identity doctest plus a run-to-run determinism check are now standing
  gates for every multi-dispatch kernel here.)
- Captured-operand doctest: BIT-IDENTICAL 4/4 (out AND state, both layers).
- 2B greedy digest, same wheel OFF vs ON: d512 9789a28bbbb5723a ==
  (token-identical).
- Perf (3 interleaved pairs, clean gates): T=512 fast-op 33.98/34.52/34.88
  vs ship 37.56/37.47/37.88 ms (-9.3% median); T=1024 65.71/66.39/67.13 vs
  73.27/73.31/73.40 (-10.0%). Disjoint medians, every pair wins.

Decode T=1 profile (item 3, M2, production raw-gates entry, fallback-lines 0
= fused fires): tiled 0.102 ms/call raw; perrow forced 0.095 — and jwm1
0.095 full — decode does NOT scale with part size: it is fixed-cost-bound
(per-call turnaround + 2 MB f32 state round trip at ~20 GB/s effective; the
kernel body is 1 dispatch, 64 wgs, 2 barriers). Lever class for the ~1
ms/token decode gap: neighborhood dispatch fusion at T=1 (conv1d-update +
gating + GDU + norm -> 1-2 dispatches; 229 dispatches/token on the 2B), not
kernel-internal work.

G13G expectations (theory, to verify with w71): the hoist removes 2 of 4
loop-1 MMAs plus the raw k/q staging from the serial chain but keeps 16
loopS barriers — a smaller relative win than on T6021; the diet composes
(additive, both envs independent).
