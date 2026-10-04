# 2026-10-04 PrefillSens — injected-work sensitivity of the pf512/pf1024 wall on G13C: which kernel class actually prices prefill

Date: 2026-10-04. Chip: Apple M1 Max (G13C/T6001), kernel 7.1.13-3-2-ARCH,
host redacted per the public-repo rule (private lab notebook holds the
unredacted transcript,
`entries/PrefillSens/20261004T104500Z-jw16-prefill-injected-work-sensitivity.md`).
Continues the prefill-axes / prefill-cast-census / PrefillFuse wall-decomp
thread: the GPU_PROFILE critical path was declared invalid (inferred path
exceeded the measured span), so this lane built a profiler-free instrument
instead. Provenance printed beside every measurement
(`scripts/mlx_provenance.py`, verified=match); one diag wheel
(`0.32.4.dev202610041551+diag.0efaaac10`, built on the host from branch
`agent/prefillsens-repeat` commit `0efaaac10`), arms are env-selected twins
of the SAME wheel, so the distinct-builds rule is satisfied by construction;
idle gates (load1 < 0.5, PSI cpu some avg10 <= 0.05) recorded per row and
green rows only counted (red-gated rows are labeled as such below);
`flock /tmp/m1-gpu.lock` per run inside gpuwin windows; service
stop/restore + health probe (`health_ok=1 probe_finish=length
active=active`) after every window.

## Instrument

`MLX_OMARCHY_DEBUG_REPEAT_<CLASS>=k` (backend, default OFF,
`overlay/mlx/backend/omarchy/encoder.cpp`): re-issues every dispatch of one
kernel class k-1 extra times back-to-back — same pipeline, descriptor set,
push constants, and grid, so the kernel recomputes identical values from
unchanged inputs; generation digests MUST stay bit-identical (asserted
every pair; a digest mismatch voids the arm). `MLX_OMARCHY_DEBUG_EMPTY=k`
records k binding-free null-kernel dispatches after every dispatch.
Knobs refuse wave-sched mode (injection would silently no-op) and update no
work budget, hazard tracker, profiler, or dispatch counter. Commit
`0efaaac10` (branch `agent/prefillsens-repeat`), shader `debug_null.comp`,
append-only profile id `DebugNull`.

Validation: EMPTY=3 delta -0.4 ms, not disjoint from control; CAST=9 digests
bit-identical to control; qmm dose response k=2/3/5 measured 1.483/1.460/1.483
ms per added dispatch. This validates dose linearity, not an additive model of
the entire uninstrumented wall.
Sysfs `gpu_busy_percent` does not exist on this kernel (probe: no busy node
under /sys/class/drm; drm fdinfo has no engine stats), so busy attribution
is census-anchored (PrefillCast census, same chip, NDJSON re-analyzed:
per-class dispatch counts 2B pf512/pass — qmm 187, cast 223, ew 192,
copy 153, rms 115, gdn 36, sdpa 12, rope 12, softmax 6; busy/dispatch qmm
28.0 us, gdn 24.4, sdpa 24.5, rms 16.1, copy 15.3, rope 15.5, softmax 18.3,
ew 13.3, cast 13.2; total 16.5 ms busy/pass over 939 dispatches).

## The table (2B, qwen3.8-hybrid; pf512 medians of green alternating ctl/arm pairs, 1 warmup + 3 timed reps, digest-identical every pair)

| class | n/pass | busy us/disp | marginal wall per ADDED dispatch pf512 | pf1024 | wall/busy (pf512) | implied ctl-wall share | verdict |
|---|---|---|---|---|---|---|---|
| QMM (prefill coopmat) | 187 | 28.0 | **1483 us** (k=2/3/5 dose-linear; +1109 ms on k=5, disjoint) | **3010 us** (red-gated, tight) | **53x** | ~277 ms = 75 % | THE critical-path class |
| GDN (prefill scan+conv) | 36 | 24.4 | **1190 us** (+341.7 ms, disjoint) | 2365 us (+681.0 ms, disjoint) | 49x | ~43 ms = 12 % | second critical class |
| SDPA (qk/pv coopmat + softmax composition) | 12 | 24.5 | 272 us (+26.1 ms, disjoint) | not run | 11x | ~3.3 ms | minor |
| COPY | 153 | 15.3 | 43.1 us (+52.8 ms, disjoint) | 86.4 us (red-gated) | 2.8x | ~6.6 ms | hides in gaps |
| RMS | 115 | 16.1 | 42.0 us (+38.6 ms, disjoint) | not run | 2.6x | ~4.8 ms | hides in gaps |
| ROPE | 12 | 15.5 | 35.9 us (+6.9 ms, disjoint by 0.1 ms) | not run | 2.3x | ~0.4 ms | negligible |
| CAST | 223 | 13.2 | 21.8 us (+37.6..38.8 ms, disjoint) | 42.9 us (+76.6 ms, disjoint) | 1.65x | ~4.9 ms | hides (PrefillCast wall-flat reproduced) |
| EW (elementwise/silu/swiglu/chains) | 192 | 13.3 | 21.3 us (+32.7 ms, disjoint) | not run | 1.6x | ~4.1 ms | hides |
| SOFTMAX | 6 | 18.3 | ~57 us (red-gated) | not run | ~3x | ~0.3 ms | negligible |
| MATMUL (plain) | ~0/pass | — | ~0 (no repeats fired) | — | — | 0 | class absent at 2B prefill |
| EMPTY control | — | 0 | ~0 (NOT disjoint; -0.4 ms) | — | — | 0 | binding-free inserts free |

Ctls: pf512 370.7-372.0 ms across windows (365-377 spread); pf1024
722-723 ms. Every pf512->pf1024 ratio of per-dispatch marginal cost is ~2.0x
(qmm 1483->3010, gdn 1190->2365, cast 21.8->42.9, copy 43.1->86.4): the
marginal cost scales with the dispatch's M-sized work, not a fixed launch
overhead.

## Reading (labeled inference where marked)

1. The pf512 wall is per-big-dispatch-gap bound, not GPU-busy bound. Adding
   qmm-class dispatches costs ~53x their busy; adding cast/ew dispatches
   costs ~1.6x their busy. The PrefillCast cast-memo wall-flat negative and
   the prefill-axes raster +1-3.4 % are both explained: kernel-time
   reductions in classes whose marginal cost ~= busy buy almost nothing,
   because their dispatches hide inside the gaps left by the big classes.
2. The qmm and GDN injections have much larger wall effects per added
   dispatch than their census busy duration. The measured ratios identify
   these as strong wall-time levers; they are not a complete wall-time
   decomposition because the added-work sensitivity can include dependencies
   and interactions with other classes.
3. Lever ranking for wall (not busy): (a) reduce the NUMBER of qmm
   dispatches (fusion — e.g. a prefill multi-weight q/k/v one-dispatch
   variant on the QmmVecQ4Multi decode precedent: each removed qmm dispatch
   is worth ~1.5 ms at pf512, ~3.0 ms at pf1024); (b) same for the GDN
   prefill pair (~1.2-2.4 ms each); (c) kernel-efficiency work on cast/
   copy/rms/ew is near-worthless for wall (PrefillCast's -41 % cast busy =
   0.0 % wall confirms); (d) the qmm gap itself did not yield to the raster
   twins (kernel-side), so the remaining qmm axis is dispatch-count or
   driver/ICD per-launch dependency cost, not the kernel.
4. Micro finding: 200-op dependent rms chain, REPEAT_RMS=2 = 1.30x ctl
   per-op (51.9 -> 67.7 us), NOT 2x — small single-digit-workgroup ops
   partially co-schedule; the "dependent = serialized turnover" model from
   PrefillFuse window 3 holds only for big-grid dispatches. EMPTY inserts
   are free (no hazards). Dependent-dispatch turnover is priced per unit of
   dispatch work-size, not per dispatch.

## 9B attempt (mlx-community Qwen3.5-9B 4bit, pf512, ~8.2 s passes)

QMM=5 was attempted in the final window. Captured QMM and control rows were
all red-gated by ambient load (zero green timed rows); they are not a valid
9B sensitivity estimate. GDN and SDPA 9B cells were not run. Therefore the
2B measurements do not establish the 9B coefficients.

## Provenance and artifacts

- Wheel `0.32.4.dev202610041551+diag.0efaaac10` (cp314 aarch64, built on
  the host from a fresh clone of `agent/prefillsens-repeat` = `0efaaac10`),
  `mlx_provenance.py` verified=match at every window top; prompts
  `qwen38-2b-prompts.jsonl` sha256 prefix `9299a3b2fc136a4c`; model shards
  head-hashed before the 9B windows.
- Raw rows and scratch archival are not yet confirmed in this receipt; the
  working files remain on jw16 under `/var/tmp/prefillsens`.
- Output digests were identical across the recorded 2B control/arm comparisons
  (`080a0ce2a3eeace4` and `01c02431b17a0616` prefixes).
- Load-gate honesty: W2/W5/W7/W8 pf1024 arms and one W3 pass ran under
  ambient lane load (llm-inference serving bursts); those rows are
  red-gated and labeled; where cited they corroborate green-gated cells
  with tight spreads (e.g. qmm5 pf1024 2967.5-2978.5 vs ctl 717.5-729.6
  across load 0.6-0.8 — GPU-bound walls are load-insensitive here), never
  carry a verdict alone.

## Closed axes (do not re-run)

Per-dispatch injected-work sensitivity at 2B pf512/pf1024 on G13C (this
receipt): instrument validated; qmm/GDN and hiding-class sensitivity measured
as tabulated, but not a complete additive wall decomposition.
Kernel-time-only levers on the hiding classes (cast/copy/rms/ew) are closed
for wall at prefill (three independent negatives now: cast memo, this
table, raster). Open: qmm/gdn dispatch-count fusion, green-gated 9B
confirmation, and separation of class cost from dependency/driver effects.
