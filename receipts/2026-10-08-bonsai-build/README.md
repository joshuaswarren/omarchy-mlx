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

## Task B leg 1 — CPU build + ctest on jwm1 (done 2026-10-07); wheel blocked on bundle

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

**Wheel: blocked, by design.** `DEV_RELEASE=1 scripts/build-wheel.sh` on
jwm1 refused at the bundle gate:

```
[bundle] runtime pin declares parakeet-encoder-whole but MLX_OMARCHY_WHOLE_BUNDLE_DIR is unset; refusing to build a wheel that would silently fall back
```

The pinned `parakeet-encoder-whole` bundle (458 MB, manifest + program-0.anec)
was searched for and is staged on **none** of: jwm1, jw16 (`/var/tmp` and
`~`), jw14m2 (`/var/tmp` and `~`), dev box (known absent since FamBonsai's
2026-10-04 attempt). The op-level pytest suites (`tests/test_bonsai_*.py`)
and the GPU tickets need that wheel; either the bundle's durable location is
named and staged, or the pin changes. No fallback was attempted — the repo
refuses silently-degraded wheels by design and so do these tickets.
