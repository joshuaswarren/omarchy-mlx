# Bonsai build qualification (serve-qualification plan, 2026-09-20)

2026-10-07/08. Lane: BonsaiBuild. Static phase run on the dev box (x86
container, CPU-only); hardware phases pending the GLM cluster window.

## Task A — static ComputeKernel wiring gate (landed)

Problem: commit 5f05b1ee2 landed Bonsai shader dispatch rows referencing
17 `ComputeKernel::Bonsai*` values that `overlay/mlx/backend/omarchy/compute.h`
never declared, so the feature never compiled. Nothing static caught it:
the wiring spans four places (compute.h enum, compute.cpp `shader_bytes`
switch, CMakeLists `omarchy_shader` targets, overlay-wide
`ComputeKernel::` references) with no cross-check.

Fix: commit `53bbe82c9` (on origin/main) adds:

- `scripts/check_compute_kernel_wiring.py` — stdlib text scan (~0.1 s
  warm, ~1 s cold; budget 5 s). Fails on:
  1. a referenced `ComputeKernel::<Value>` missing from compute.h;
  2. a `shader_bytes()` row whose symbol has no `omarchy_shader()` target;
  3. an `omarchy_shader()` target no row returns (counts drift);
  4. a referenced value with no dispatch row and not a sentinel
     (`Custom`, `Count`) — `create_pipeline` would throw
     `invalid_argument` at runtime;
  5. a `_r<N>`-suffixed shader target without `-DROWS_PER_SLOT=<N>` on
     its build line (spec-constant drift).
  Declared-but-unwired values pass as named INFO (append-only
  GPU-profile id placeholders; deleting them would shift profile ids).
- `tests/test_compute_kernel_wiring.py` — 8 unittest cases on synthetic
  fixture trees pinning every failure class plus the dead-entry INFO
  contract; no compiler, no network. All pass.
- `scripts/hooks/pre-push` — runs the gate before the privacy check;
  fail-closed, so a non-compiling feature cannot be pushed.

### Evidence

FAIL on the 5f05b1ee2 tree (`git archive 5f05b1ee2 overlay` into a
scratch tree, gate pointed at it), exit 1, all 17 names with file:line:

```
compute-kernel-wiring: 1 problem(s)
  - referenced but not declared in compute.h:
    BonsaiQ1DequantBF16 (...compute.cpp:1510)
    BonsaiQ1DequantF16  (...compute.cpp:1508)
    BonsaiQ1DequantF32  (...compute.cpp:1506)
    BonsaiQ1QmvSubgroupBF16 (...compute.cpp:1497)
    BonsaiQ1QmvSubgroupF16  (...compute.cpp:1495)
    BonsaiQ1QmvSubgroupF32  (...compute.cpp:1493)
    BonsaiQ1QmvTreeBF16 (...compute.cpp:1504)
    BonsaiQ1QmvTreeF16  (...compute.cpp:1502)
    BonsaiQ1QmvTreeF32  (...compute.cpp:1500)
    BonsaiQmvWideSubgroupBF16R2 ...R3 ...R4 ...R5 (compute.cpp:1512-1521)
    BonsaiQmvWideSubgroupF16R2  ...R3 ...R4 ...R5 (compute.cpp:1524-1533)
exit=1
```

PASS on the landed main tree (commit `53bbe82c9`), exit 0:

```
compute-kernel-wiring: ok — 565 declared, 562 referenced, 560 dispatch
rows, 559 shader targets, 3 dead entries
compute-kernel-wiring: info: ... MatmulBf16FmaL16C4
compute-kernel-wiring: info: ... QmmPrefillFmaL16C4F16
compute-kernel-wiring: info: ... QmmPrefillFmaL8C4F16
exit=0
```

The push itself exercised the new hook: the dead-entry INFO lines print
in the push transcript (`git push origin HEAD:main`,
`2fe4680e2..53bbe82c9`).

### Review findings (static, on the landed tree)

- Enum/dispatch/shader wiring is consistent: 562 referenced values are
  all declared; the 560 covered rows return 559 distinct symbols and
  the target set is exactly equal (no orphan, no missing). One symbol
  is shared by two enum values by design: `SelectComplex64` and
  `SelectI64` both ride `select_complex64` (32-bit word transport).
- Three declared values are wired nowhere: `MatmulBf16FmaL16C4`,
  `QmmPrefillFmaL16C4F16`, `QmmPrefillFmaL8C4F16` (compute.h:644-647).
  Dead profile-id placeholders; never dispatched, so harmless, but the
  gate now names them so the pile cannot grow silently.
- `shader_bytes` has no `default:`; uncovered values reach the trailing
  `throw std::invalid_argument` — a referenced-but-unwired value is a
  runtime crash, which check 4 now blocks at push time.
- The 6 duplicate `case MatmulDirectBF16*` labels are the
  `#ifdef MLX_OMARCHY_BF16_DIRECT` / `#else` alternates (return vs
  `break`+throw); no compile hazard.

## Task C — CI compile gate (implemented, landed with this receipt)

The cheap text gate cannot see what only a C++ frontend sees. Commit in
this push adds `.github/workflows/omarchy-syntax.yml` +
`scripts/ci/omarchy_syntax_check.py`: GitHub Actions (ubuntu-latest,
CPU-only) runs `scripts/prepare-mlx.sh` against the mlx.lock pin,
configures the omarchy backend (`-DMLX_BUILD_OMARCHY=ON` +
`-DMLX_BUILD_METAL=OFF`...), builds the `omarchy_shaders` target (559
embedded SPIR-V headers), then rewrites every omarchy backend compile
from `compile_commands.json` to `-fsyntax-only` and runs them in
parallel. No link, no GPU, no wheel, `.work` cached on `mlx.lock`.

Measured on a 16-core dev box (16 jobs): prepare 7.6 s, configure 6.2 s,
shader headers 8.7 s wall (~80 s CPU), syntax check 12.5 s wall (~46 s
CPU), 22/22 omarchy translation units pass. Confirmed on the real
runner: run
[37652359143](https://github.com/joshuaswarren/omarchy-mlx/actions/runs/37652359143)
(workflow_dispatch, ref main @ 263b5fe6c) — every step green, 1 m 42 s
total. A first push-triggered run stalled 9 min in apt on a runner
mirror flake (37651222111, canceled; not a workflow defect — the retry
installed in seconds). The gate runs on every push to main and every
PR; not too heavy.

Negative proof: with the 5f05b1ee2 overlay staged over a fresh prepare,
the same driver fails in 27.6 s end-to-end with the real compiler
errors, exit 1, 20/22:

```
FAIL compute.cpp
.../mlx/backend/omarchy/compute.cpp:1493:25: error:
    'BonsaiQ1QmvSubgroupF32' is not a member of
    'mlx::core::omarchy::ComputeKernel'
FAIL primitives.cpp
.../primitives.cpp:12418:37: error:
    'BonsaiQ1QmvSubgroupF32' is not a member of
    'mlx::core::omarchy::ComputeKernel'
omarchy-syntax-check: 20/22 translation units pass
```

Documented limitation: the gate validates the preprocessor state of the
configured build. The bf16 direct GEMM path is enabled only when the
shader compiler accepts GL_EXT_bfloat16 (the configure-time probe
decides), so a glslang-12 runner checks the disabled state; glslc
coverage comes from the jw16 gate builds.

## Task B leg 1 — CPU build, ctest and wheel on jwm1 (done 2026-10-07)

Per Main's reroute (jw16 root 90% full): built on **jwm1** (Apple M1, T8103,
chip G13G, aarch64, Omarchy aurora 12.3, kernel
`7.1.12-2-12.3-sep-ARCH`, boot id `16f875bd-bbea-4a4b-af71-6ed0c7f1617c`,
8 cores / 15 GiB). The three test binaries are destined for **jw16** (M1
Max, T6001, G13C) — portable per Main (shader SPIR-V is embedded; mlx is
linked into the binaries); both chips are stated here as required.

- Source commit: `09da0917a5d0394bd9e4c0a4c42116aa5e08f818` (origin/main tip
  at 2026-10-07T21:36Z).
- Toolchain: GCC 16.1.1 (aarch64), cmake + ninja, glslc 2026.3 and glslang
  16.4.0 (shader compilers), mesa 1:26.2.3, vulkan-icd-loader 1.4.357
  (ICDs present: asahi, lvp, virtio — none exercised; CPU-only build, no
  Vulkan device opened).
- Exact command:
  `CMAKE_BUILD_PARALLEL_LEVEL=4 nice -n 19 ionice -c3 bash repo/scripts/ticket_bonsai_cpu_build.sh ~/bonsai-build`
  (`nice`/`ionice` because fill-run is absent on jwm1; verified with `which`).
- Wall time: 21:36:56Z → 21:41:58Z = **5 m 02 s** (build + shaders + ctest;
  1232/1232 ninja targets).
- Peak RSS: **5415 MB** (10 s sampler over system used-memory; limit 15 GiB).
- ctest (100%, 124.7 s total):
  - `omarchy_runtime_tests` — Passed 38.80 s
  - `omarchy_primitive_tests` — Passed 0.90 s
  - `omarchy_matmul_family_tests` — Passed 85.0 s
- Binary sha256 (identical after the dev-box relay and on jw16
  `~/bonsai-run/`, copied with `scp -l 320000`):
  - `32ecb234307e4c7f7c000ef2d6027a9c4762f9a7a154b6174e3e50b5692246dc`
    omarchy_primitive_tests
  - `9a8cecb130687bb91051dbd9e4bdb17e94b798532feb1e8f0ddf93ceb89d3c96`
    omarchy_runtime_tests
  - `bcf2dc4becb8b513e51a42dbfffd586dd3f907826a0cd54c778fba7e4729ec8e`
    omarchy_matmul_family_tests
- Compile diagnostics: 354 warnings, all the same class — GCC 16.1.1
  deprecation of `std::atomic_load/atomic_exchange(shared_ptr*)` from
  upstream `mlx/error.h` (pre-existing upstream code, warnings only).
- Numerics: the three suites' internal expectations are the bar at this
  stage; all pass. Op-level pytest parity bars and dispatch-trace evidence
  require the wheel (below) and a GPU ticket — next legs, not yet run.

**Wheel: built after staging the pinned bundle (Main's decision).**
The first attempt refused at the bundle gate (log preserved). Main
identified
the candidate `~/.local/share/coreglass/whole-bundle/` on jw16; the
runtime pin (`parakeet-runtime-pin.json`) demands manifest
`08769793f8ee3299381f499bf90537d620e23635000a6dbc54b9b5c6a55a54ab` +
program `13c744231524d440b0a774155343df9ade0bbcbc37edc4b1ccf9698e580d5453`
and BOTH jw16 candidates — the coreglass dir and the v0.7.31 release
extract (`~/v0.7.31-assets/bundle-extract/.../parakeet-encoder-whole`) —
hash to exactly those bytes (458,022,314 B). The coreglass copy traveled
jw16 → dev box → jwm1 with sha256 verified at every hop (`rsync
--bwlimit=40000`, `ionice -c3`; no direct jwm1→jw16 key exists).

- Wheel: `mlx_omarchy-0.32.4.dev202610072150+09da091-cp314-cp314-linux_aarch64.whl`
- Size: 416,712,074 B; sha256
  `4e2b0ca80e74ce9ddbefb20f60ef3f8537dd26561ebde4989e8b35c1f2c43aa7`
  (identical on jwm1, dev-box relay, and jw16 `~/bonsai-run/`).
- Source commit: `09da091`; python tag cp314 (jw16 runs 3.14.7 — install
  compatible); the bundle gate staged the pin-exact bytes
  (`[bundle] staged ... (manifest 08769793…, program 13c74423…)`,
  wheel2.log).
- The build script's own NEXT-STEP note applies before any release:
  `scripts/verify-release-assets.py <tag>` must print VERIFIED if this
  wheel is ever uploaded.

**GPU parity leg (G13G ticket, same host's GPU; the G13C run follows on
jw16 when its queue allows).** Three gpu-turn tickets, wheel
`0.32.4.dev202610072312+6de45e2` (dispatch fix + gate fixes), fresh venv
per ticket, `OMARCHY_BONSAI_GATE=1` (skips are failures):

| route | result |
|---|---|
| q1 dequant k512 gs64 f32 | PASS (after the dispatch fix) |
| q1 dequant k1024 gs128 bf16 | PASS (after the dispatch fix) |
| q1 qmv — all dtypes/shapes (f32/f16/bf16, k512-k4096) | FAIL — wrong values |
| qmv wide q1/q2, m2-m5 | FAIL — wrong values |
| gate preconditions | PASS (GPU probe works; zero silent skips) |
| ctest suites (CPU leg) | 3/3 pass (leg 1; GPU paths not exercised there) |

Bug 1 — dequant under-dispatch (fixed, commit in this receipt): the
shader maps one row per workgroup but the encoder dispatched
ceil(n*k/256) groups; k<256 left rows 1..n-1 unwritten. Fix: dispatch
exactly n groups. Hardware-verified: dequant f32 and bf16 arms green.

Bug 2 — RESOLVED for q1 qmv (commit 0e0871392): the shader's
`VALUES_PER_BYTE = 32u / BITS` computed codes per WORD (32) where the
byte walk needs codes per BYTE (8), so bytes_per_row was K/32 and the
kernel read only the first quarter of each packed row; the inline
comment even stated the intended value ('8 for bits=1'). Hardware
sweeps pinned it exactly: one-hot x probes matched byte 0 codes at
positions 0..7 and byte 1 codes at positions 32..39 — the 4x k-stride
signature. After the fix, ALL q1 qmv arms pass on hardware (f32/f16/
bf16, k512-k4096, gs32-128) and the dequant arms stay green.

Bug 2b — qmv wide wrong values on hardware (OPEN, not fixed): the
wide route fails in every arm (q1 m2-m5, q2 m2/m5). Three distinct
defects identified by audit against the observed outputs
(shaders/bonsai_qmv_wide.comp):

  1. line ~140: the row stride is `row_base[0] * (words_per_row / 4u)`
     — the shader's own comment derives the stride as bytes_per_row/4
     which EQUALS words_per_row; the extra /4 re-divides, so rows read
     at a quarter of their true word offset (overlapping wrong data).
  2. lines ~168-183: the 1-bit loop reads `input_w.values[w_row + w]`
     with w_row from row_base[0] only — the ROWS_PER_SLOT-1 sibling
     rows of the slot never have their words read; every slot row
     accumulates the first row's weights.
  3. lines ~207/225: results are written at
     `output_offset + column + r` while the loops compute
     dot(x[r], w[column]) — the write indexes the (M, N) output buffer
     as if it were (N, M), transposing/scrambling everything after
     element [0][0] (which is why ACTUAL[0][0] equals DESIRED[0][0]
     in every wide failure).

A correct wide kernel needs a small redesign (per-r weight addressing
plus (m, n) output indexing); it is not a safe same-session patch.
Until it lands, the wide route must be considered NOT hardware-
qualified; the gate keeps these arms red by design.

Original bug 2 (qmv) evidence record:
every qmv
and wide arm mismatches its composed reference by large margins
(e.g. wide m2 first-row elements off by up to 5.65; qmv single-column
got 0.2232 vs reference 0.6676). Probes (all under gpu-turn, logs
preserved):

- device identity confirmed: `Apple M1 (G13G B1)`, driver `Honeykrisp`,
  api 1.4.362 under `VK_DRIVER_FILES` -> the coreglass
  `vulkan-4b-bbbfa36dce` ICD (mesa-git-sha bbbfa36dce7, the 0.7.31 gate
  ICD); no packaged ICD exists on this host, so the override is
  required.
- one-hot x sweeps: the kernel's perceived code vector matches the
  packed truth only in the first 16 of 64 positions at k=64; bits in
  bytes 2..7 are never seen (40/64 match, non-permutation).
- single-bit packed sweeps: bits in packed bytes 0..1 are seen, bytes
  2..7 never (k=64, 8-byte row).
- dequant cross-check on the same bytes: row0 decodes bit-exactly at
  k=64 (and k=512), so the bit order and byte layout of the test packer
  are right; the defect is specific to the qmv/wide read path.
- shader + encoder audit: the qmv shader's byte walk, byte_at() select,
  group scale/bias indexing, and the encoder's binding order and push
  constants all read correct; the dequant shader differs mainly in
  mapping one ROW per workgroup. Suspect space: the qmv dispatch/params
  path or glslc codegen for this shader on the G13G driver — needs a
  kernel-lane pass with the preserved probes.
