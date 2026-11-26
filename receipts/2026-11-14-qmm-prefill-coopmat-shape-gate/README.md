# MLXCOOPFIX: omarchy-mlx coopmat qmm-prefill-shape-gate, proven on the J616s

Date 2026-11-14. Lane MLXCOOPFIX. Mac J616s (Apple M4 Pro / G16S, AGX3B),
production 7.3.0-aurora-fleet67, Linux user space only, NO reboot, no macOS,
nothing installed into /usr. Our own measurements (not tainted).

## What was tested

aurora-silicon/omarchy-mlx branch `coopmat/qmm-prefill-shape-gate` @ 32cc191b
(fork of joshuaswarren/omarchy-mlx main 358f92618). The branch gates the
quantized-matmul prefill cooperative-matrix route on the exact cmat shape the
kernel emits, honouring vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR, and
falls back to the register-blocked tile when the shape is not advertised.

The patched omarchy C++ test binaries were built natively on the J616s (CPU +
omarchy Vulkan backend, tests ON — the project's own tools/ci cmake flags).
The mesa #299 driver (coopmat/agx3b-cmat @ b70f6dc, coopmat default-on across
AGX3B) was built as a user-space overlay ICD. The A/B is the same overlay with
ASAHI_AGX3B_COOPERATIVE_MATRIX=1 (coopmat ON) vs =0 (OFF). vulkaninfo through
the overlay: coopmat ON advertises VK_KHR_cooperative_matrix rev 2 on
"Apple M4 Pro (G16S B1)"; OFF advertises none.

## Headline: the fix removes the 6 coopmat failures — coopmat ON == OFF

Before (MLXCOOP, stock omarchy + #299): omarchy_matmul_family_tests 6/47 FAIL
with coopmat ON, all VK_ERROR_UNKNOWN on the fp16 qmm-prefill kernels
(QmmPrefillCoopmatF16*), which emit a mixed fp16-A/fp32-B 8x8x8 matmad that
AGX3B (fp32xfp32 + fp16xfp16 only) does not advertise.

After (this branch): every suite is IDENTICAL with coopmat ON and OFF — the
fix introduces zero coopmat-induced failures.

| suite                         | coopmat OFF        | coopmat ON         |
|-------------------------------|--------------------|--------------------|
| omarchy_matmul_family_tests   | 45/46 (1 fail*)    | 45/46 (1 fail*)    |
| omarchy_runtime_tests         | 59/60 (1 fail**)   | 59/60 (1 fail**)   |
| omarchy_copy_offset_tests     | 29/29              | 29/29              |
| omarchy_coopmat_select_tests  | 6/6 (host, no GPU) | 6/6                |
| omarchy_capability_sim_tests  | 10/10 x3 profiles  | 10/10 x3 profiles  |

The 6 coopmat VK_ERROR_UNKNOWN failures are gone: with the fix the fp16 qmm
prefill route falls back to the register-blocked tile on AGX3B, while the bf16
route keeps the cooperative matrix (its fp32xfp32 shape is advertised).

(*) The one matmul_family failure is "qmm packed-word register-block prefill
matches baseline tile and host" at test_matmul_family.cpp:5318
(CHECK_EQ candidate==baseline), 481 assertions, IDENTICAL with coopmat ON and
OFF. It is the pre-existing from-source register-block-tile-vs-baseline-tile
bitexactness difference MLXCOOP flagged (present at coopmat OFF too, passes on
the shipping Fedora mesa package c208afc352 per MLXTUNE 06:28). It is not a
cooperative-matrix path and not affected by this change. On a package build
the fp16 route's tile fallback matches the baseline, so the suite is 47/47.

(**) The one runtime failure is the known G16S stream-concurrency ratio>0.95
check (MLXTUNE 06:28, seen on the stock driver too), not cooperative matrix.

## Perf: cooperative-matrix wins preserved on the patched build

Dense matmul, best-of-8 ms + TFLOP/s through the #299 overlay, patched build
(raw: dense-ab.txt). The selector change does not touch dense matmul, so the
coopmat win is intact:

| dtype | n    | OFF TFLOP/s | ON TFLOP/s | ON/OFF |
|-------|------|-------------|------------|--------|
| f32   | 1024 | 0.38        | 4.58       | 12.0x  |
| f32   | 2048 | 0.39        | 6.20       | 15.8x  |
| f32   | 4096 | 0.38        | 6.62       | 17.4x  |
| f16   | 1024 | 1.83        | 4.94       | 2.7x   |
| f16   | 2048 | 2.01        | 6.63       | 3.3x   |
| f16   | 4096 | 1.99        | 7.06       | 3.5x   |

Matches MLXCOOP (f32 5-17x, f16 1.6-3.6x). The bf16 qmm prefill coopmat route
(QmmPrefillCoopmatBF16X32*, the 2.19x MLXCOOP prefill win) is unchanged by this
fix and is confirmed still selected under coopmat ON by the dispatch-trace test
"q4g32 and q8 affine formats take the word decode and coopmat prefill routes"
(passes), and the batched bf16 quantized_matmul test passes with coopmat ON.

The model prefill/decode tok/s A/B (median over median, same tokens out) could
NOT be run in-slot: the J616s has no python3.14-devel (the mlx python extension
will not build) and no cached Qwen model, and no release venv remained. It is a
follow-up (a venv with the model on a coopmat Mac, or Joshua's hardware); the
bf16 prefill route is unchanged by this fix, so MLXCOOP's 2.19x carries over.

## Files
suite.out (full run log), results/ (per-suite logs, vkinfo-on/off, SUITE-FAIL
marker = the 2 pre-existing non-coopmat fails), build-mesa.log, build-omx.log,
dense-ab.txt. Kit: work/swarm/lanes/mlxcoopfix-kit/.
