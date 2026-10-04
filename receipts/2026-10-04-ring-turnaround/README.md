# 2026-10-04 RingTurnaround — jw16 (M1 Max) per-dispatch ring turnaround: whole-word CDM_BARRIER reductions at decode depth — DOCUMENTED NEGATIVE, word axis closed on G13X

Lane: RingTurnaround (worker). Host jw16 (T6001/G13X), ssh alias per fleet config.
Serving stack UNTOUCHED end to end: venv `/var/tmp/v072-venv-fused`
(0.32.4.dev202610031046+b581d5c, mlx_provenance verified=match in both windows), system
ICD `libvulkan_asahi.so.1432df0196-transfer` (hwmat-vec2-on lineage, dependency-tracked
per-launch CDM barrier; `strings` confirmed `HK_CDM_BARRIER_MASK` + `HK_PERFTEST` in the
installed .so). Every cell inside `gpuwin.sh` windows; RESTORE health_ok=1
probe_finish=length active=active after each. Notebook (pre-registered before any window):
`apple-silicon-lab/entries/RingTurnaround/20261004T003435Z-jw16-ring-turnaround-cdm-word.md`;
artifacts `apple-silicon-lab/artifacts/RingTurnaround/20261004/` (w1/w2 dirs + scripts,
SHA256SUMS.local/.remote).

## Question

Why is dependent-dispatch ring turnaround ~21 us/op on Linux (Honeykrisp + drm/asahi +
firmware submit path) vs ~12.5 us/op macOS in a decode token chain, and can any whole-word
reduction of the G13X dependent-dispatch CDM_BARRIER hold bit-exact at all decode depths
and recover >= 1.5%?

## Method

Env-only whole-word arms via `HK_CDM_BARRIER_MASK=<hex>` on the serving ICD (zero builds,
byte-exact word overwrite incl. block-type bits, same mechanism as DecodeGap3). Cells:
`qwen38-mlx-bench.py --limit 1 --new-tokens {64,128,256,512} --prefill-tokens 0 --warmup 1
--passes 1`, serving env pins NORM_APPLE=1 GDN_BATCH=1 GDN_F16_STATE=0, greedy digest
(`ordered_records_sha256`) checked per cell against the DrainFix/DecodeBw p1 pins.
Alternating ctl/arm pairs; gates (uptime>=360s, load1<0.5, PSI cpu some avg10=0.00) at
window top with bounded settle and per-cell bounded re-checks.

## Boot event (disclosed up front)

jw16 REBOOTED at 2026-10-04T00:38Z between lane prep and W1 (boot 4b353848 ->
757cd3f3-693d-49be-8183-aa2578d95a61, same kernel 7.1.13-3-2-ARCH; `last -x reboot` cited
in the notebook entry). Not caused by this lane (gpuwin has no reboot path; the lane never
issued reboot/power commands). Serving venv + ICD verified unchanged after the reboot.
Consequences: all ctl cells re-established pins on the new boot — **every pre-reboot pin
reproduced bit-exact post-reboot** (d64 cb3e8770..., d512 619360bf... etc. across 13 ctl
cells) — and all A/B comparisons are internally paired within the new boot.

## W1 — single-run probe map (window 2026-10-04T00:46-00:48Z)

| arm (word) | bit set | d64 tok/s | d256 | d512 | digests |
|---|---|---:|---:|---:|---|
| ctl 0x17f | {0,1,2,usc,4,5,6,8} | 108.88 | - | 102.38 | OK (pins) |
| 0x178 | {usc,4,5,6,8} (drop 0-2) | 110.73 (+1.70%) | 109.37 | 104.65 (+2.22%) | OK x3 (1 run each) |
| 0x177 | {0,1,2,4,5,6,8} (drop usc) | 108.91 (+0.03%) | 107.60 | 102.54 (+0.16%) | OK x3 — NEUTRAL |
| 0x108 | {8,usc} (drop 4,5,6) | - | - | - | **DIGEST-MISMATCH at d64** (3a4e6a6e...), window aborted by design |

Reading at W1: 0x178 looked like the DecodeGap3 d64-clean arm finally holding all depths —
the pre-registered H1. 0x177 answers the mislabeled DecodeGap3 arm (their "drop usc only"
0x187 = {0,1,2,7,8} actually dropped 4,5,6 too): the true usc-only drop is clean but free
(the usc bit costs ~nothing at cell level; the study's 4.6 us/launch chain estimate does
not translate to cells). 0x108: bits 4/5/6 are load-bearing for the RAW wait.

## W2 — landing pairs + chain slopes (window 2026-10-04T00:49-00:50Z, ABORTED by the digest gate)

- Chain slopes (this lane's instrument, `chain_slopes.py`, min-of-reps wall over n=32..160):
  ctl add 6.045 us/op / gemv 36.33 us/op (65.2 GB/s); 0x178 add 5.741 / gemv 35.971.
  Instrument note: absolute add slope is ~2x below DecodeGap3's 12.2 us/op — the
  construction is NOT faithful across lanes (theirs: different chain/reps aggregation);
  only within-lane ctl-vs-arm deltas are reported, and they are small.
- Decode pairs, d64 (only depth reached): r1 ctl 109.18 / arm 111.20; r2 ctl 109.28 / arm
  111.50 (+1.85% / +2.03%, digests OK).
- **r3 arm: DIGEST-MISMATCH at d64 (304d1237fefe... != cb3e8770...).** Window aborted by
  the gate; service restored, health 200 + finish=length.

## Verdict

**H1 REFUTED: 0x178 is nondeterministically corrupt on the dependency-tracked G13X
driver** — 2 clean d64 runs (W2 r1-r2), then a divergence on the third, all at load
0.53-0.63; DecodeGap3's two d64-clean runs and this lane's W1 single-run map were exactly
the short-gate blindness the standing rules warn about. The corruption class is the
same stochastic-divergence signature as 0x170 at length ("a different wrong digest every
run"), now demonstrated at d64 for the usc-bearing variant. A word that is bit-exact only
~2 of 3 runs is dead for landing; nothing was deployed.

Pre-registered landing rule outcome: **NO candidate met the bar.** With this lane:

| word | set | status on current stack |
|---|---|---|
| 0x17f | {0,1,2,usc,4,5,6,8} | minimal correct (13 ctl cells bit-exact across a host reboot) |
| 0x170 | {4,5,6,8} | corrupt at length (DecodeGap3) |
| 0x178 | {usc,4,5,6,8} | **stochastic corrupt at d64 (this lane, W2 r3)** |
| 0x187 | {0,1,2,7,8} | corrupt at d64 (DecodeGap3; was mislabeled "drop usc only") |
| 0x177 | {0,1,2,4,5,6,8} | clean, NEUTRAL (+0.16% d512) — usc bit is free but removable |
| 0x108 | {8,usc} | corrupt at d64 — bits 4/5/6 are load-bearing RAW-wait |
| 0x0F7 / 0x1F0 | swap 7/8 | clean, NULL (DecodeGap3) |

## Where the ~21 vs ~12.5 us/op lives (quantified, per-gap)

- Unordered dispatch floor: 1.3 us/op (nocdm add slope, corrupt; DecodeGap3) — launch
  cadence is NOT the cost. Trivial 1-WG cadence on G13G: 4.52 us/dispatch,
  barrier-independent (dispatch-floor receipt).
- The CDM_BARRIER word cost on chains: ~10-11 us/op (add class) to ~18-20 us/op
  (memory-kernel class) over the floor; the load-bearing part is the {4,5,6,8} RAW
  drain handshake (dropping any of 4/5/6 corrupts: 0x108), and the {0,1,2}+usc
  cross-launch coherency set is REQUIRED at every dependent boundary on this driver
  (0x170/0x178 stochastic corruption; 0x177 shows usc alone is not the corrupting
  ingredient — the corrupting ingredient is dropping bits 0-2).
- macOS pays 4.77 us/op (add chain) for the same full-correct dependency on the same
  silicon (Jw16DecodeGap W2 macOS window, same-machine measurement; no macOS windows
  were available to this lane). The ~7-8 us/op excess is firmware drain semantics of
  this barrier vocabulary — DecodeGap3's conclusion, now confirmed with three
  additional words including the never-tested 0x177 and 0x178-at-depth.
- Per-edge attribution inside the token (prior receipts, unchanged by this lane):
  Turnover gap_attr — profiled gap scales with next dispatch grid (26 us before <=64 WG,
  79-81 us before 256+) and prior-kernel duration; Jw16BarrierElide — 226 RAW
  barriers/token, every one a true RAW at hardware granularity; host is not the binder
  (DrainFix strace census). Stream census from the command builder (this lane, code
  reading of mesa-1 e7631595df6): per dependent dispatch = CDM_LAUNCH_WORD_0 (4B) +
  WORD_1 (4B) + CDM_GLOBAL (12B) + CDM_LOCAL (12B) + CDM_BARRIER (4B) = 36B + one
  USC-words block (~48-80B, host-written pool); no per-dispatch timestamps; stream
  links/terminates are per-CS (2 CS/token), not per-dispatch; `hk_ensure_cs_has_space`
  requests 0x2000+link+0x800 so a ~229-launch token crosses at least one 64KB chunk
  link. None of these is the 21 us — the barrier drain is.

## Provenance

- mlx-omarchy serving wheel 0.32.4.dev202610031046+b581d5c, mlx_provenance verified=match
  in both windows; mlx_lm 0.31.3; kernel 7.1.13-3-2-ARCH; boots 757cd3f3 (both windows;
  pre-reboot boot 4b353848 for context only).
- Mesa reading: joshuaswarren/mesa-1 honeykrisp-omarchy-v3 @ e7631595df6 (shipped
  candidate), asahi/hwmat-vec2-on @ 283bf35c055 (serving lineage), serving ICD build
  1432df0196. A candidate branch/worktree (agent/ring-turnaround off e7631595df6) was
  prepared for the fix and REMOVED unused when H1 failed — no Mesa commit, no ICD build,
  no serving change.
- Protocol deviations, disclosed: (1) mid-plan load gate accepted sub-1.0 loads due to a
  `0.*` pattern bug (cells ran at load 0.53-0.63, above the 0.5 protocol bound); the
  defect is a speed-protocol limitation only — greedy digests are deterministic and the
  mismatch that killed the arm is load-independent (ctl cells at the same loads stayed
  bit-exact); (2) two 16-s aborted launches before W1 (script bugs: missing mkdir, then
  the same gate bug at window top), no data; (3) the host reboot mid-lane (above).

## Post-state

- Nothing landed anywhere: serving venv/ICD/wheel byte-identical to lane start, health
  200 + completion probe after every window. Branch `agent/ring-turnaround` (this repo)
  carries this receipt + `tools/ringturnaround/` (w1_inner.sh, w2_inner.sh, w2_run.sh,
  chain_slopes.py, pins.py) for reproducibility.
- The remaining per-dispatch levers on G13X are the ones the ledger already names:
  fewer RAW edges via fusion (DispatchFuse lane), the BATCH_WORK=0 cross-submit lever
  (+1.5%, Turnover, pending the owner cap decision), and firmware-semantics work on the
  {4,5,6,8} drain handshake — which requires understanding what bits 4/5/6 wait on
  (untested bits 9-20/24/26 remain unexplored but are additive-only under the mask
  vocabulary and cannot replace the load-bearing set without a semantics breakthrough).
