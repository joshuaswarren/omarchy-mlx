# 2026-10-05 — mlx-omarchy v0.7.28 release (rope-norm B>1 hotfix)

Release: v0.7.28, tag `v0.7.28` = `5c15fbaea` (= origin/main at V28 GO),
annotated, pushed at the cut sha before any gate ran. Tag never moved.
Published 2026-10-05 as stable + latest, aarch64-only:
https://github.com/joshuaswarren/omarchy-mlx/releases/tag/v0.7.28

## Hotfix: batched B>1 requests crashed with the default qwen3 rope-norm fold

v0.7.25, v0.7.26 and v0.7.27 carried a defect in the default-ON qwen3 dense
q/k-norm+rope fold (`mx.fast.rope_rms_norm`). Every request batch with more
than one sequence (B>1) crashed:

```
ValueError: [broadcast_shapes] (B,8,L,64) and (B,1,L,128)
```

The shared fallback composition of the C++ op read the norm weight (D values)
as rotary frequencies (D/2 expected), so every B>1 leg built wrong-size
trig tables and failed to broadcast. The B=1 path was not affected: it
used the fused GPU kernel and stayed bit-exact.

Who was affected: the oMLX backend (`jundot/omlx`) with default settings.
`omlx` serves with `completion_batch_size = 8` by default, so two or more
concurrent HTTP requests to a default `python -m omlx.server` hit the
crash. A bare `python -m mlx_lm.server` (batch defaults 32/8) was exposed
the same way. The managed `mlx-omarchy-serve` mlx-lm backend and the
assistant app pin decode and prompt concurrency to 1, so they were not
affected. Any batched B>1 loop over qwen3 dense models (evaluation,
training-side inference) crashed the same way. The B=1 decode path was
unaffected and stayed bit-exact on every chip.

The fix (`48ce2b25f`): the fallback composition now uses real frequencies
when a norm weight is parked in the inputs vector; the fused kernel
already read per-request offsets. Full root cause and evidence:
`receipts/2026-10-05-rope-norm-batch-fix/README.md`. The interim v0.7.27
patcher fence that refused every non-integer cache offset is removed;
the fold stays default ON; the kill switches stay
(`MLX_OMARCHY_ROPE_NORM_FUSE=0`, and the q/k-norm patcher keeps its own).
A dated correction line has been added to the v0.7.25, v0.7.26 and v0.7.27
release pages.

## Changes since v0.7.27 (grouped)

- **Rope-norm B>1 fix** (`48ce2b25f` + fence removal): the hotfix above,
  with a behavior regression test, an op-level check script, and M2 +
  dev-box verification harnesses.
- **mlx-lm tool-call argument parsing fix** (`cf4ef86be`, `46584f42e`):
  backport of mlx-lm #1904, shipped on the applied 0.31.3 and 0.32
  patcher lines; idempotent patch-series rerun; self-contained regression
  test (`793546208`).
- **Nine upstream MLX backports** (`d0cc18bf1`, `8aa962f55`, `1019a9edf`,
  `2e2643191`, `03c607f93`, `7a4853c82`, `a5740771f`, `4c2100f1b`,
  `b96ba8890`, wired backend-generic by `3f144a1b1`): view-offset,
  eval-cleanup deadlock, shared-buffer reshape contiguity, view last-axis
  stride, view contiguity flags, multioptimizer empty group, as-strided
  contiguity, aligned array pointer, vmap scatter axis. Triage and
  build-fix receipt: `96d9f7cce`.
- **TensorFold port** (`efced3a29`): translator extensions for the
  Metal-kernel source dialect (casts, subgroup builtins, pointer guard,
  pragma/device strip), the `mx.fast.int8_matmul` W8A8 Vulkan op with a
  composed fallback, lavapipe-verified tests. The custom-kernel
  translation cache now keys on the translator's own source hash
  (`33ff979ed`), so stale cached GLSL from an older binary cannot load.
- **qmm prefill: padded shared B-tile stride default ON** (`d078129f0`,
  twins `ab2af3d62`, routing fix `4907c5157`): measured +2-3% prefill on
  the T6021 host and +0.9% on the G13G host, bit-exact digests; kill
  switch `MLX_OMARCHY_QMM_LDSPAD=0`.
- **Opt-in per-SoC GPU base-pstate overlays**: ship inert; opt in through
  `/etc/omarchy-mac-boot/dtb-overlays.opt-in` (paths follow
  omarchy-mac-pkgs#3). Docs under `docs/gpu-base-pstate-*.md`.
- **oMLX-on-Linux compat layer** (`017ca25ca`, `a0fcaff59`, `4b7e2ddfc`,
  `bc81c2842`, `69801282a`): oMLX v0.7.0 and the TensorFold `mx.metal`
  gates route to the omarchy backend on Linux; the installer runs the
  platform gate before the compat series. Live T6021 smoke receipts:
  `19952daef`, `910a1c975`.
- **Hardware probe** (`bce97ef2a`, `ed3b40b79`): `mx.device_info()` gains
  `memory_size`, `max_recommended_working_set_size`, `gpu_cores`,
  `marketing_name`; `set_wired_limit` is an honest no-op off-Metal;
  interior NULs in `chip_compatible` become spaces.
- **Kernel-battery translator fixes** (`ca195e620`, `2f57a1a26`,
  `585ffba29`, `732c333a7`, `17f2fd605`, `0fa118e50`, `ba88ad971`,
  `e2dc22ead`, `986f77bd8`, `83b45e82d`, `dcc9f0641`, `ebe05fc50`,
  `58cd779b3`): bf16 handling, const-int wrapping, device-pointer
  aliasing, rint/roundEven mapping and more, proven on the T6021
  minimax_m3 battery.
- **Receipts and docs**: numerics-gate closure for the batched==single
  digest equality on Qwen3-4B (`ad4f5c898`), golden-build recipe
  (`9f4f4f747`), rope-norm batch-fix evidence, DFlash/oMLX matrix rows,
  minimax_m3 family plan.

## Build provenance

- `mlx_omarchy-0.32.4.dev202610050725+5c15fba-cp314-cp314-linux_aarch64.whl`:
  fresh full build on the T6021 gate host, 163 s wall, `-j4 nice 10`,
  whole-encoder bundle staged and pin-verified, source commit
  `5c15fbaea` (cut sha = origin/main).
- x86_64 cp311 wheel: NOT SHIPPED on v0.7.28. The emulated build
  (`debian:bookworm` amd64 container under the T6021 host's
  qemu-x86_64 binfmt) was attempted repeatedly, was too slow, and
  held the T6021 build slot for hours; it was killed at the publish
  decision (2026-10-05 ~11:25Z, Main) and the release cut aarch64-only.
  v0.7.27 carried a dev-box-built x86 wheel; v0.7.28 does not — the
  x86 dev box was off-limits this release. The stale waitrun and
  container processes were stopped on the T6021 host before cleanup.
- `omarchy-mlx-vendor-wheels-v0.7.28-cp314-aarch64.tar` (+ `.sha256`):
  built by `packaging/vendor-wheels.sh` on the T6021 host against
  `packaging/requirements-lock.in`; 36 vendored wheels verified against
  the generated lock; the release wheel copied in via `--wheel`.

## Hotfix verification (PASS)

- **Op-level matrix** B{1,2,4} x L{1,17} x offset{int 0, int 5,
  per-request array} on the release wheel, Qwen3-4B shapes, bf16:
  fused output bit-exact against the eager `rms_norm -> rope` chain in
  18/18 rows, 0 errors (the shipped v0.7.27-era wheel errored the 4
  B>1 array-offset rows).
- **Behavior tests** through the real patched mlx_lm `qwen3.py` with
  left-padded batched caches: patched logits bit-equal to the eager
  chain for B in {1,2,4}; the fold dispatches by default;
  `5/5` passed (`tests/test_rope_norm_batch_behavior.py`).
- **B=1 decode digest A/B** (fleet bench, same window, d64
  limit 5 / passes 2 / warmup 2) on the T6021 host: published v0.7.27
  control wheel vs release wheel. 2B 76ec87cb6ceda7113… == 76ec87cb6ceda7113…
  (== v0.7.26 M2-ledger pin; rates 107.3/106.9 tok/s). 4B
  42d27a8cbe93df49… == 42d27a8cbe93df49… (rates 64.0/64.0). 9B
  80274aa790426468… == 80274aa790426468… (== v0.7.27's recorded T6021 9B
  d64 route-ON digest; rates 39.05/39.17). B=1 unchanged on all three
  models. The 4B e2c919be pin belongs to the jw16 serving harness corpus
  and is covered by the jwm1 legs (see below).
- **Numerics gate** (`docs/numerics-gate.md`): per-op fp32/fp64 error
  no worse than the deployed path (opcheck row); B=1 greedy digests
  unchanged; S=1 PPL not addressed by this hotfix. The full
  three-criterion bar is on the rope-norm B>1 fix, not on a decode-route
  change.

## Release battery (draft-first, against the DRAFT bytes at `5c15fbaea`)

GitHub CI note for traceability: the "Community collectors" workflow was red
at the cut commit `5c15fbaea` and at the v0.7.27 cut `6edd258f8` with the
same two environment-dependent failures (macos-14 `test_collect.py`
AneSectionUnavailableMarks; ubuntu pytest imports `mlx` in a runner without
mlx installed). Pre-existing, not introduced by this delta.

T6021 gate host (M2 Max, T6021; same stock boot as the v0.7.25/26/27
batteries; throwaway gate HOMEs under GATE_ROOT=/tmp):

| gate | result |
|---|---|
| g1 clean install + serve entry | RC=0 |
| g2 online 9B | RC=0 |
| g3 online 4B + SSE card | RC=0 |
| g4 offline | RC=0 |
| g5 Laya pin | RC=0 |
| g6 TTS codec | RC=0 |
| g7a packaged ICD | RC=0 |
| g7b system install | RC=0 |
| g8 Kokoro default smoke | RC=0 |
| g9 speak-queue | RC=0 |
| g10 primer + TTFA | RC=0 |
| g11 card on the 9B | RC=0 |
| g12 streamed Kokoro | RC=0 |
| g14 routing default | RC=0 |

14/14 RC=0 (run-all without RUN_JW16; the jw16 rows are the recorded
skips below). Post-window rerun from scratch on the rebuilt wheel tree,
complete 2026-10-05 ~11:4xZ.

M2 substitutes for the jw16 GPU legs (jw16 mid-reinstall, skipped per
this release's policy; the notes say so plainly):
- g7d fresh-image parakeet transcribe: PASS on jwm1's live ANE (the
  substitute was unnecessary because the T8103 ANE + packaged parakeet
  stack are live on jwm1; cache stage ran before g7c per the gate order).
- g13 GDN doctest (`omarchy_gdn_maskless_correctness_tests`): PASS,
  exit 0, run on the T6021 host from the release wheel build tree
  (relaunch pid 78827, start 11:27:11Z).
- g15 trig contract: PASS — `omarchy_eq_math_tests` 7/7 (g15a, exit 0)
  and `omarchy_trig_reduction_tests` 4/4 (g15b, exit 0), same run,
  done 11:28:28Z.
- g7c packaged ANE worker verify: PASS on jwm1 (live ANE; the substitute
  was unnecessary because the T8103 ANE + packaged parakeet stack are
  live on jwm1).

jwm1 (T8103, G13G) legs through w71 — results:
- d64 greedy digests 2B/4B/9B on the release wheel vs the current
  baseline wheel in the same window: v0.7.27 == v0.7.28 bit-exact on
  2B (`eee1cf96…`), 4B (`42d27a8c…`) and 9B (`80274aa7…`). 3/3 PASS.
- `g7c` packaged ANE worker verify: PASS on the live T8103 ANE (cache
  stage before g7c).
- `g7d` fresh-image parakeet transcribe: PASS on the live ANE.
- `capsim` (omarchy_capability_sim_tests profile matrix): 5/5.
- Standing M1 battery (overlay/tests/omarchy/ suite binaries built with
  `MLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON`): 25/26 PASS. The one
  red is `omarchy_fast_ops` (two rope-refusal-message CHECKs); the
  same two CHECKs are red against the v0.7.27 baseline wheel in the
  same window, so this is not a regression and stays on the published
  v0.7.28 surface. `omarchy_ane_runtime_tests` does not link without
  `MLX_OMARCHY_ANE_DEVICE`; that gate is omitted here.

## Assets (final)

- `mlx_omarchy-0.32.4.dev202610050725+5c15fba-cp314-cp314-linux_aarch64.whl`
  sha256 `68bb536fa4879ff6367b35617e800e9a3cfe1d35563458474ebdf33e2ce4c92d`
- `omarchy-mlx-vendor-wheels-v0.7.28-cp314-aarch64.tar`
  sha256 `defbc78d3abe4f799213f416768915dc3176abe7bd18a3465073c610f47a7f32`
- `omarchy-mlx-vendor-wheels-v0.7.28-cp314-aarch64.tar.sha256`
  sha256 `be49684e3b50b0d8cd216d499b5ab9134e3b0757b1f995da6b3469ee1ce79d5f`
- `SHA256SUMS` (final 3-asset coverage; the provisional 4-asset file that
  still listed the x86 wheel was replaced before publish)

## Decode number (the honest pin in the release notes)

4-bit Qwen3-4B d64 greedy decode 64.47 tok/s, digest
`42d27a8cbe93df49f3c5089e5ddd3109bd4dc513f545ac5559a9e57bc3b2a0e9`,
measured 2026-10-05 11:21:57Z on the T6021 host in a fresh throwaway
HOME installed from the release assets via the tag installer
(`/var/tmp/rel0728-install-test.log`, fleet bench, 5 prompts, 2 passes;
digest == the release wheel's A/B 4B d64 digest). A publish-time
re-test installs from the PUBLISHED release URL; see Publication.

## Publication

- Draft verify (before publish):
  `scripts/verify-release-assets.py v0.7.28 --platforms linux_aarch64`
  -> `VERIFIED: every uploaded asset matches what the release claims`
  (rc=0; sha256, version identity, build commit, feature strings,
  platform tag, SHA256SUMS coverage all PASS; the `--platforms` flag
  is a one-flag script addition recording the aarch64-only asset set
  in the invocation, landed with this receipt).
- Published with `gh release edit v0.7.28 --draft=false --latest`.
- Published-release verify: same command, rc=0, VERIFIED (identical
  PASS set; release state "published, stable").
- Publish-time install test (fresh HOME on the T6021 host, tag
  installer, default release base = the published GitHub URL, installer
  downloads the wheel and verifies it against SHA256SUMS):
  /var/tmp/rel0728-pubtest.log on the T6021 host — PASS. INSTALL_EXIT 0;
  installer smoke `mlx-omarchy 0.32.4.dev202610050725+5c15fba: matmul OK`
  + ANE smoke OK; `mlx-omarchy-info` full report (Apple M2 Max G14C B1,
  Honeykrisp); one short generation on the installed venv
  (mlx_lm, Qwen3-4B-Instruct-2507-4bit, 8 tokens, temp 0) under a
  5-minute gpu-turn ticket: output `OK.`, GEN_EXIT 0 (11:40:38Z).
  Throwaway HOME deleted after capture; log copy archived in the lab
  notebook at artifacts/Release0728/2026-10-05-v0728-battery/ (with both
  verify outputs and the final SHA256SUMS).

## Known issues (unchanged, open)

- SDPA fused-VJP value defects remain open: qL=1 maskless dk/dv are
  zero; B=1, kL=5 also fails.
- `compile()` of the sin*cos tape segfaults on the G13 build host; that
  leg stays out of the test suite.
- Kokoro voice output remains unqualified.
