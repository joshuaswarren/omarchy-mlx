# v0.7.32 notes DRAFT (working document; nothing cut, no tag until FREEZE)

Base: v0.7.31 (9b5c938fe, published 2026-10-07). Range: v0.7.31..origin/main
= 84 commits, 184 files, +16827/-67 (as of 2026-10-08, draft started;
re-run the range command at FREEZE).

## Shipped (user impact)

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

## NOT listed as shipped yet (needs hardware proof before FREEZE)

- **Bonsai family** (dequant dispatch fix 576223963, qmv byte-walk constant
  0e0871392, wide-kernel row addressing ba71943eb + scale/bias deconflation
  902d40936, 21f57e231 MLX 0.32.3 patch drift, fbd8cfb9d enum entries,
  f4a659358 runtime-M const): G13G hardware parity matrix receipt exists
  (b865a929b) but the **wide-route redesign has NO hardware proof**. Rule
  from Main: only list as shipped if G13G+G13C hardware receipts exist.
  G13C leg ticket is staged (689aab921) but not run. Until both legs are
  green on hardware, Bonsai stays OUT of the shipped section.
- Bool MaskedScatter: NOT landed (no commit in range).
- Kernel recheck translator fixes: the harness/translator work landed
  (795015644, 353da309a, 123469dec, cd8670ca0, 9a7e83285, a1158b1df,
  72ebe853e, 73838b26d) but these are dev-tooling; they affect what ships
  later, not this release's runtime. Listed under tooling, not user impact,
  unless Main says otherwise.

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

## v0.7.32 gate plan (additions per Main, 2026-10-08)

- g1 fresh-home install on EVERY host (dev box, jwm1, jw16, M2) and g17
  patch-series on every host again (both lines rc=0 + idempotent, now
  including mlx-lm-last-logits-qwen3 on pristine wheels).
- All v0.7.31 gates repeat (g2..g12, g14, g16/g16b P+B, g-g13c P+B,
  g-hold, jw16-gates full plan) on the 0.7.32 wheel once it exists.
- Scheduling: idle-guard backlog queues are the way to book host time
  (~/src/omarchy-mplus-private/tools/idle-guard/README.md): one file per
  host+engine on omp-studio-local, tab-separated ticket lines; the 60 s
  systemd timer submits the head of each backlog as a real detached
  gpu-turn ticket when the host has been idle 3 min, never disturbing
  MEASURE work; lanes append their pre-approved tickets to the queue files
  instead of grabbing hosts directly.

## Open before FREEZE
- Bonsai G13C leg + wide-route hardware proof (decides shipped vs not).
- Confirm nothing else user-visible lands that is missing here.
