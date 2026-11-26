# Upstream drift #21: 26 commits already inside pin 9c3d35571a — nothing missing

Lane: UpstreamDrift. Issue: #21 (bot drift report, old pin `59d600b5` -> upstream main
`9295197533`, 26 commits). Verdict: **the pin bump (2e6a39d6a, merged 88890c73c) already
contains all 26 commits; the series applies clean at the pin; prepare + configure + full
compile pass.** One test-only compile break on main tip was found and fixed by this lane
(see "Fix" below) — it was not upstream-induced but blocked the `MLX_BUILD_TESTS=ON`
deep gate.

## Provenance

- Source: worktree at main `07729f401` (contains the pin merge `88890c73c`);
  `mlx.lock` pin `9c3d35571a`, archive SHA-256
  `3e564bf7a0d5e5bac3a56fc26668cdcedd4a9db9fbfbf2bb6d762632ceb6f65a` verified by
  `scripts/prepare-mlx.sh`'s `sha256sum --check` at prepare time.
- Host class: x86-64 development box, kernel 6.17.2, gcc 12.2.0, cmake 3.25.1,
  ninja 1.13, glslangValidator 16.5 (no `glslc` here — compile evidence only; no M1
  shader-acceptance claim, no runtime/numerical gates run on this box).
- Upstream history read through a bare blob-filtered clone under `.work/` (gitignored);
  this repo's history untouched.

## Ancestry (why the issue is stale)

`git merge-base --is-ancestor 9295197533 9c3d35571a` -> true. Every commit in the
bot's `59d600b5..9295197533` window is an ancestor of the pin. (`v0.32.3..9295197533`
is 7 commits; the issue's 26 are counted from the old pin.)

## The 26 commits

Dispositions: every row is "in pin". "entry" = Omarchy-backend action the commit
requires beyond inheriting it.

| commit | change | class | in pin | omarchy entry |
|---|---|---|---|---|
| `77e1cfb2` | #4552 fence deadlock (`Fence::wait`/`update` contract) | shared contract | yes | overlay `fence.cpp` implemented (waits cover every published update) |
| `04282417` | #4519 complex64 NaN ordering | CPU | yes | inherited |
| `84ea22f3` | #4505 SDPA D256 metal memory | Metal-only | yes | n/a |
| `073d2252` | #4402 `mx.matrix_transpose` | composed op | yes | lowers to existing `Transpose`; no new primitive |
| `c0418f6e` | #4487 SDPA D512 metal memory | Metal-only | yes | n/a |
| `465b7a5d` | #4510 shapeless scan compile | shared header | yes | adds only `DEFINE_INPUT_OUTPUT_SHAPE()` to `Scan` (macro; no new virtual) — verified diff |
| `a2a09fd5` | #4567 gather mm metal | Metal-only | yes | n/a |
| `02ce1fb6` | #4520 fused `fast.cross_entropy` Metal kernels | Metal-only | yes | composed path stays reference/tested |
| `351e59a0` | #4499 SDPA D96/V64 | Metal-only | yes | n/a |
| `a32f0c3d` | #4484 IndexError for bad axes | common error path | yes | inherited |
| `0a8e18e0` | #4515 integer floor_divide | common | yes | inherited |
| `4fe4d1e2` | #4517 CPU binary ops, large data | CPU | yes | inherited |
| `9eb3e3ab` | #4514 CUDA dtors / implicit default streams | CUDA + common | yes | inherited (`compile.cpp`/`stream.h` compile clean) |
| `90089440` | #4572 gather_qmm metal | Metal-only | yes | n/a |
| `0780da5e` | #4511 compiled-kernel fp constant precision | backend/common | yes | covered by `omarchy_compiled_tape_tests` |
| `42987b66` | #4557 `ReduceScatter::eval_cpu` assert | CPU/distributed | yes | **absorbed** our `mlx-distributed-reduce-scatter-assert.patch` (patch deleted at the bump) |
| `64ea011c` | #4576 thread-local stream leak at exit | per-backend device | yes | inherited |
| `83b976ea` | #4563 SDPA VJP | shared fast contract | yes | overlay `ScaledDotProductAttentionVJP::use_fallback` + `has_sinks` param present; grad test in `test_fast_ops.cpp` |
| `e5b4021c` | #4555 Python 3.15/3.15t | build (top CMakeLists) | yes | `mlx-build.patch` applies clean over it |
| `92db340a` | #4501 fixed-size `unique` | composed op | yes | built from existing sort/slice; no new primitive |
| `9b3df2cf` | #4446 single-element negative-stride slice | common | yes | inherited |
| `7916d8b0` | #4582 version bump 0.32.4 | version.h | yes | dev-line version handled by `mlx-version-time.patch` |
| `92951975` | #4565 GDN VJP (`GatedDeltaUpdateVJP`) | shared fast contract | yes | overlay fallback + fused-decode VJP regression test (`test_fast_ops.cpp`, 2e-2) |
| `b3848e20` | #4488 `array.__pos__` | python binding | yes | inherited |
| `71816203` | #4492 tri/tril/triu/gather_mm defaults | python binding | yes | inherited |
| `09e67c68` | #4500 upsample linear perf | python nn | yes | inherited |

New upstream surface in the window that a mirrored backend must implement: **none**.
No new `Primitives` class; the only new `fast` class (`GatedDeltaUpdateVJP`) has an
overlay fallback plus a regression test; `Fence` and SDPA-VJP signature changes are
already in the overlay.

## The bot's two "conflict" files

- `mlx/backend/cpu/distributed.cpp` — the conflict was our
  `mlx-distributed-reduce-scatter-assert.patch`, which upstream `42987b66` (inside this
  very window) made redundant; the patch was deleted at the bump. No current patch
  touches that file. Clean by construction.
- `CMakeLists.txt` — the conflict was `e5b4021c` (Python 3.15 CMake work) against the
  pre-bump `mlx-build.patch`; the bump rebased the series onto the pin, which contains
  `e5b4021c`. Probe confirms clean apply.

The bot's other conflict labels (`mlx/fast.cpp`, `mlx/fast_primitives.h`, `mlx/ops.cpp`,
`mlx/transforms.cpp`, `python/src/fast.cpp`, `mlx/backend/no_gpu/primitives.cpp`) were
anchors of the pre-bump series; all apply at the pin.

## Patch series evidence

`tools/upstream-compat-check.sh probe` (same order + fuzz as `scripts/prepare-mlx.sh`):

- at the pin `9c3d35571a`: **19/19 patches apply clean**, exit 0;
- at the bot tip `9295197533`: 18/19 — `mlx-python-buffer.patch` hunk #3 misses because
  it anchors on the pin's `python/src/buffer.h`, which includes in-pin upstream #4523
  (`d63f0f93`) that postdates the bot tip. Expected: the tip is strictly older than the
  pin; no action.

`scripts/prepare-mlx.sh` from scratch at the pin: exit 0 (hash check + overlay copy +
full series).

## Compile evidence

Configure: `MLX_BUILD_OMARCHY=ON MLX_BUILD_CPU=ON MLX_BUILD_METAL=OFF
MLX_BUILD_CUDA=OFF MLX_BUILD_TESTS=ON EXAMPLES/BENCHMARKS=OFF` — exit 0.

Clean full-graph build (`ninja -k 0`, fresh build dir): **1174/1174 steps ran; the only
failing step is the link of `tests/omarchy/omarchy_ane_runtime_tests`**, which is a
documented configuration boundary, not a defect: `ane/runtime.cpp` is compiled into
`mlx` only under `MLX_OMARCHY_ANE_SOURCE_DIR`
(`overlay/mlx/backend/omarchy/CMakeLists.txt:682`; requires a qualified omarchy-ane
checkout + `BUILD_SHARED_LIBS=ON` + libdrm), and the suite "requires the private ANE
checkout / a second rank" per the pin-bump receipt. The M1/jw16 batteries configure with
that checkout. Everything else — `libmlx.a` (whole omarchy backend), all other omarchy
suites, upstream tests, tools — compiles and links. Note for the ANE lane: on a plain
box, the unconditional `add_executable(omarchy_ane_runtime_tests ...)` in
`overlay/tests/omarchy/CMakeLists.txt` makes a default `MLX_BUILD_TESTS=ON` build fail;
gating that one suite on `MLX_OMARCHY_ANE_SOURCE_DIR` would fix the default build.

`scripts/mlx_provenance.py` on this box: `verified: no-mlx` (no wheel installed;
compile-only run, no runtime measurement). Source identity is the HEAD + pin archive
SHA-256 above.

## Fix (test-only, found by this build)

`8db7abb73` added `overlay/tests/omarchy/test_conv_gemm_decomp.cpp` after the last full
battery, and its `conv1d` calls passed a `Stream` in the `dilation` slot — the file never
compiled under `MLX_BUILD_TESTS=ON` anywhere. Fixed by passing `/*dilation=*/1,
/*groups=*/1` explicitly at all four call sites (signature:
`conv1d(input, weight, stride, padding, dilation, groups, stream)`). Compile verified
before/after; the suite now builds and links on this box (running it needs a qualifying
Vulkan device, which this box does not provide).

## Closing summary

Upstream drift window `59d600b5..9295197533` (26 commits) is fully inside pin
`9c3d35571a` (merge `88890c73c`): series 19/19 clean at the pin, fresh prepare +
configure + full compile green (single by-design exception:
`omarchy_ane_runtime_tests` needs the private ANE checkout), no backend entry missing;
only fix needed was the test-only `conv1d` signature repair in `test_conv_gemm_decomp.cpp`.
