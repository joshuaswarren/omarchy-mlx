# 2026-10-05 GridBarrier — hop removal via persistent kernels with in-kernel grid barriers (design note + measurements plan)

Lane: GridBarrier (worker). Branch `agent/GridBarrier` (this file's tree).
Continues jwm1-parity H136/H137a/H137b. Notebook pre-registrations:
`apple-silicon-lab/entries/GridBarrier/20261005T033000Z-jwm1-gridbarrier-stress-h288.md` (H288)
and `20261005T033500Z-jw14m2-gridbarrier-h289.md` (H289).

## Premise and what is already measured (prior receipts, not re-run here)

- Decode's remaining Linux-vs-macOS gap is dependent-dispatch turnover: ~27 us per dependent
  dispatch on Linux vs ~15 us on macOS (chain-dep-bench; RingTurnaround 2026-10-04: 21.0 us
  real chain / 12.2 synthetic vs macOS 12.5 / 4.8; unordered floor 1.3 us) x ~90 dispatches/token.
  The CDM_BARRIER word at the dispatch boundary is firmware drain semantics; no bit is
  droppable (RingTurnaround: {4,5,6,8} RAW load-bearing, {0,1,2} required, 0x178 stochastic
  corrupt). Removing the hop, not cheapening it, is the remaining lever.
- H136: in-kernel grid barrier (atomic counter + generation, bounded spin) costs
  0.72-1.24 us at G <= 64 (32-thr WGs), jwm1. H137a: 128-thr WGs co-resident to G = 192,
  slope 1.885 us; timeout at 256. H137b: 2-stage persistent Q4 GEMV pair BIT-EXACT at every
  fitting G but +25 us/pair SLOWER (ref 104 vs fused 129 at G=64): the register-heavy
  production GEMV has real-kernel residency 80 <= ceiling < 96 WGs of 128 threads, so the
  persistent form streams at ~G/256 of full MLP. Parallelism-bound, not sync-bound.
  Barrier correctness at device scope already shown bit-exact on AGX for ~7k crossings
  (H137b exactness pass, coherent hand-off bindings).

## What is new in this lane (the open questions)

1. H288 (jwm1, w71 bundle): the >= 1e7-crossing producer-consumer stress with random data and
   per-value validation, PLAIN bindings first (the production-relevant crux — MLX kernels do
   not mark buffers coherent), coherent arms for comparison, at G = 64 (local 128) and
   G = 128 (local 32). Plus probe re-verify with a dense G list filling the H137a 192..256 gap.
2. H289 (M2/G14C after w73 reopen): probe (residency ceiling + cost vs G at local 128,
   G to 512), stress at the working G, and the H137b pair A/B with PAIR_GS up to 384 — the
   decisive question: with ~4.75x the cores, does the real GEMV kernel's residency cover the
   256 tiles so the persistent form stops losing MLP?
3. If H289 Q4 passes (bit-exact AND fused <= ref - 10 us): pre-register H290, the fused
   layer-tail prototype (MLP tail: norm -> gate/up GEMV -> swiglu -> down GEMV + residual, or
   attn o-proj + residual + norm; 2-3 internal barriers), bit-exact vs the separate kernels
   with identical accumulation orders, then the 2B decode A/B on the affected chip(s).

## Tool (branch agent/GridBarrier)

`tools/gridbarrier-bench/bench.cpp`: probe mode (H136/H137a method: G workgroups, R barriers,
slope of min wall R=2001 vs R=1, 5 reps, 2^17-poll bounded spin, stop escalation at first
timeout flag) and stress mode (parity double-buffered per-WG slots, 128 lanes/WG, per-round
write of hash(round,wg,lane), grid barrier, every lane validates every other WG's slot and
counts violations with first-bad (round, producer, consumer, got, want) diagnosis; done_wgs
completion counter; plain vs coherent binding variants; push-constant round count).
`tools/gridbarrier-bench/pair/`: H137b pair bench + generator + production base shader for
the M2 leg (`run-pair.sh`, PAIR_GS env).

## Safety

Every spin bounded (~40 ms) with sticky timeout flag and clean exit; escalating G stops at
the first timeout (H137b's incident fix); hardware runs under the GPU lock / gpu-turn tickets
with `timeout 900`; journal + `systemctl --failed` recorded after every run. No new failure
mode is introduced beyond H136/H137a/H137b's already-chartered one, and the spin bound is the
proven-safe 2^17, not the original 2^22.

## Results

(filled after the hardware legs)
