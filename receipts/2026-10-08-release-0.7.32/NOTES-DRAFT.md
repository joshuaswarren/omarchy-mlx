# v0.7.32 notes DRAFT (working document; nothing cut, no tag until FREEZE)

Base: v0.7.31 (9b5c938fe, published 2026-10-07). Range: v0.7.31..origin/main
= 84 commits, 184 files, +16827/-67 (as of 2026-10-08, draft started;
re-run the range command at FREEZE).

## Shipped (user impact)

### Upstream correctness fixes (Vulkan backend primitives/shaders, landed after v0.7.31)

- **bool MaskedScatter native** (85c4a0b9a, merge f3dc90c94, upstream
  #4635): dedicated masked_scatter_bool.comp shader + dispatch; verified on
  G13G hardware, matrix row updated (receipts/2026-10-08-bool-masked-scatter/README.md,
  landed via 1e87dc129).
- **ArgReduce NaN propagation** (b42203b66): argreduce_suffix.comp now
  propagates NaN like the reference — argmax/argmin over rows containing
  NaN previously returned wrong indices deterministically (upstream suite
  test arg reduce NaN failed 6/6 before). Documented as
  OMARCHY-ARGREDUCE-NAN in docs/known-defects.md (:1057, FIXED same day);
  test coverage in overlay/tests/omarchy/test_reduce_ops.cpp; audit
  receipt receipts/2026-10-08-audit-followups/README.md (b0fe81f50).
- **empty Sort / ArgSort short-circuit** (e7f56fc3d): empty arrays
  short-circuit before the dtype refusal instead of erroring; defect entry
  in docs/known-defects.md (same receipt b0fe81f50).
- **Scan::eval_gpu parent-aliased-output guard** (a3ccf3a90): shared-buffer
  view allocation guard in Scan eval_gpu prevents aliasing a parent
  buffer's output; failing-before test 7c0317160 (tests: CumSum with a
  pre-allocated shared-buffer output); receipt b0fe81f50.

### Serve performance
- **mlx-lm last-logits for the dense qwen3 family** (0bdba12e8, both patch
  lines): cached prefill computes the quantized head only for the final
  prompt position on qwen3 dense models. w71 measured 4B pf512 +9.0%,
  digests equal, fp64-reference error equal
  (receipts/2026-10-08-qwen3-last-logits). Proven on pristine wheels:
  g17-patch-series dev-box run 2026-10-08, both lines rc=0 + idempotent,
  "applied: mlx-lm-last-logits-qwen3.patch" in both
  (receipts/2026-10-08-release-0.7.32/g17-0732/).
- Assistant wake word (338f63902): opt-in openWakeWord ONNX listener,
  `--wake-word`; bytes coercion + per-detector peak counter (3ccab703c);
  ALSA 'pulse' device with default fallback (573037bf3).

### Chip support
- M3/M4 family: chip table rows, M3-aware ICD error, unified m3m4 kit with
  README, data-only owner steps (284524e5d merge: 50170f716, dbd36ada4,
  91434d7fd, be7e3bb70).

## Bonsai status update (2026-10-08, revises the earlier "held out")

- **G13G (jwm1): hardware-QUALIFIED.** Full matrix green including the wide
  route: 20/20 pytest on the G13G ticket after the wide redesign (q1 qmv,
  wide q1 m2-m5, q2 m2/m5, dequant f32+bf16, preconditions, roundtrip),
  ctest 104/49/32 suites green in-ticket (2.77M+ assertions), dispatch
  trace hits Bonsai ordinal 547 with zero CPU-stream lines; build
  qualification receipts/2026-10-08-bonsai-build/README.md; wide-route
  fix + full matrix receipt 725317295.
- **G14C (M2): leg QUEUED on the M2** (in flight at draft time).
- **G13C (jw16): PENDING** (leg ticket 689aab921 staged; not run).
- Shipped-eligible on G13G evidence now; the release notes claim stays
  scoped "hardware-qualified on G13G" until G14C + G13C legs report, per
  the free-only rule.

## Tooling / gates (visible to maintainers)

- custom-kernel translator fixes + unit tests (152110af6, 8ac5c0c11,
  8d4275ae7, 9898d9722, ec5886ba4, 671bd48cb, fb6fe7c01, 4be08b70a
  KEEP_SHADER debug env).
- kernel-recheck harness: 26 upstream sources pinned byte-exact, GPU runner
  with refusal/crash/wrong classification, byte-exact dumper, MSL
  replication dumps + translate tool, mx-composed arbiter, tolerance floors.
- release-gates: gate_lock gpu-turn-aware (76fa98735, d9fa66fd8),
  WHEEL_IDENTITY logged from every gate (feb57d020), sha-pinned assets +
  auto-staged tag tree (97d31bbf0), re-cloning install trees (02b44d779),
  g7b vendor-tar extract logging (878858a65), EXPECTED_TAG_SHA required
  (83f0ddd0b), g7c heredoc bound to python not tee (ea1227088), compute
  wiring static gate (53bbe82c9), CI backend syntax gate (dea4303af,
  263b5fe6c, 733057e2b green), matmul_family G14C LDS-pin fix (884636d5e).
- v0.7.31 receipt/history commits (a4a3c6bd3, 09da0917a jw16 leg B PASS,
  e9fc09fcd, 8ae198be7, 2fe4680e2, and the SUMMARY/gate-log landings) are
  history, not release content.

## Docs
- TensorFold recipe facts fixed (02e027713), MiniMax H3 int8 recipe
  (37ced5128), upstream layout addendum (6dcc32387).

## v0.7.32 gate plan (revised 2026-10-08: shader/primitive fixes force the standing battery per chip)

The upstream correctness fixes (MaskedScatter, ArgReduce NaN, empty Sort,
Scan guard) all touch shaders/primitives, so v0.7.32 gates MUST include:

- **Standing battery (all suites in the AGENTS.md list) on EACH chip**:
  - G13G on jwm1,
  - G13C on jw16 (reachable again via `ssh 16m1mbp`; home wiped — needs
    restaging: release assets + gate scripts, as FILL),
  - G14C on the M2,
  on a fresh build from the main tip (not the 0.7.31 wheel).
- Plus on the new wheel: g1 fresh-home install on EVERY host, g17
  patch-series on every host (both lines rc=0 + idempotent), g16/g16b
  legs P+B per host, g-g13c legs P+B, g-hold, jw16-gates full plan, and
  the M2 full gate set.
- FREEZE only after: the G14C Bonsai leg + a main battery (full standing
  battery) on jwm1 AND the M2, built from the main tip.
- Scheduling: idle-guard backlog queues are the way to book host time
  (~/src/omarchy-mplus-private/tools/idle-guard/README.md): one file per
  host+engine on omp-studio-local, tab-separated ticket lines; the 60 s
  systemd timer submits the head of each backlog as a real detached
  gpu-turn ticket when the host has been idle 3 min, never disturbing
  MEASURE work; lanes append their pre-approved tickets to the queue files
  instead of grabbing hosts directly.

## Open before FREEZE
- G14C Bonsai leg on the M2 (queued).
- G13C Bonsai leg on jw16 (reachable again; restage first — home wiped).
- Full standing battery per chip on a fresh main-tip build (jwm1 + M2
  minimum before FREEZE; jw16 restaged then run).
- Confirm nothing else user-visible lands that is missing here.

Base range at this draft revision: v0.7.31..origin/main = 104 commits
(b0fe81f50 tip). Re-run the range command at FREEZE.
