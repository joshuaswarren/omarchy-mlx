# Long-SDPA flash prefill route (hd128) — H3 joint attention on Honeykrisp

Date: 2026-10-05. Worker: LongSdpa2. The branch is rebased onto current main commit `979f1707` (latest rebase; the recorded wheel and tests were completed on `771285b9` before the next M16 selector commit landed).; selector, test, and documentation are built into a fresh wheel. The hd128 flash-prefill kernel passed historical Linux-host numeric/determinism and corrected live-context gates. The selector defaults to flash only when the f32 score matrix exceeds 2^30 elements or 25% of device heap; the former L>=4096 default was removed after composed attention measured faster at H=56, L=4096. Current-base focused routing and native suite results are recorded below.

## Route and scope

TensorFold's MiniMax-H3 DiT uses one joint attention over text, audio, and video with Q/K/V `[1,56,13365,128]` bf16, scale `128**-0.5`, and no mask. The composed route materializes `[1,56,13365,13365]` f32 scores (about 10.0 G elements) and fails for large lengths: M2 at L=8192 asserts `addr_range.range <= UINT32_MAX`; at L>=10923 the checked index limit rejects the request.

`SdpaPrefillFlashBF16Hd128` streams K/V through 32-key shared tiles with f32 online softmax and no materialized score matrix. A 256-lane workgroup maps 32 query rows; one q tile is dispatched at a time. The route accepts arbitrary batch/head/row strides with innermost stride 1. It requires bf16 Q/K/V/output, head/value dimension 128, and no mask/causal/sinks. By default it replaces composition only when the f32 score matrix exceeds 2^30 elements or 25% of the device heap, avoiding the slower flash route at L=4096 where composed attention measured faster. `MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L` remains an explicit A/B force threshold; `MLX_OMARCHY_SDPA_PREFILL_FLASH=0` disables flash.`

## Build and verification

The main build fix is b226895dc. A full Linux validation-host build of the fix alone from ff46b5a5a passed and produced mlx_omarchy-0.32.4.dev202610051752+ff46b5a5-cp314-cp314-linux_aarch64.whl (416,555,043 bytes, SHA-256 acbda42ddc9edfe792ec3742aaad1aa7ce915f2313074b134b16fc52f6ea74a02). That wheel did not include the LongSdpa kernel.

The rebased LongSdpa branch build from 92e48da54 passed fresh prepare, the binding-declaration guard, pinned whole-bundle staging, runtime-asset verification, and wheel build on the Linux validation host. Wheel: /var/tmp/longsdpa-jw16-diagnostic/dist/mlx_omarchy-0.32.4.dev202610051759+92e48da5-cp314-cp314-linux_aarch64.whl (416,561,186 bytes, SHA-256 ac49fee540d9b3824184010b1fe4018b96ca8a61d62a999aa0eaa09bfc4fd7ab). It is installed in /var/tmp/longsdpa-jw16-venv; import reports version 0.32.4.dev202610051759+92e48da5.
Another fresh Linux validation-host branch build from 5f86785a (based on main 75f4e430d before later fast-ops test-only commits) passed prepare, whole-encoder bundle staging, pinned runtime-asset verification, and wheel build. It produced /var/tmp/longsdpa-jw16-currentmain/dist/mlx_omarchy-0.32.4.dev202610051919+5f86785a-cp314-cp314-linux_aarch64.whl (416,561,191 bytes; SHA-256 3b6b1d850e3c6cd78125cfe8c216b207574293a1c5745583a0700b8d2280a013). Bundle manifest SHA-256 08769793f8ee3299381f499bf90537d620e23635000a6dbc54b9b5c6a55a54ab; program SHA-256 13c744231524d440b0a774155343df9ade0bbcbc37edc4b1ccf9698e580d5453.

Before origin/main advanced to 53dce73d8, a golden-cache build at tip `99a5c17df` made wheel `mlx_omarchy-0.32.4.dev202610050436+99a5c17-cp314-cp314-linux_aarch64.whl` (SHA-256 `7a19e104bcd2197dcd9b5632f3ba339f0fbb0f336ab886e272b58fa4f40465c5`) and installed it in `/var/tmp/longsdpa-venv`. This tree replayed `overlay/` but kept stale staged `mlx/fast.cpp`; `golden_rebuild.sh` does not reapply patches. Main required a fresh `scripts/build-wheel.sh` run. That fresh prepare/build started at `/var/tmp/longsdpa-patchfix` and was interrupted by the M2 reboot.

The first wheel (pre latest-main rebase) was `/var/tmp/longsdpa-wheel/dist/mlx_omarchy-0.32.4.dev202610050436+02890c7-cp314-cp314-linux_aarch64.whl`, SHA-256 `eb42c5dff16d47ef97d7ed690e534cfad65e659a3f608c93bbb3157a38c2f255`. The latest-base wheel is `/var/tmp/longsdpa-wheel/dist/mlx_omarchy-0.32.4.dev202610050436+ccb104e-cp314-cp314-linux_aarch64.whl`, SHA-256 `426213a8ec138f2f2055076e75269a5aaee81695c5a191f34aa6732ccf7b2359`, installed into `/var/tmp/longsdpa-venv` (version `0.32.4.dev202610050436+ccb104e`).

M2 numeric/performance results from the first wheel:

- L=256/512/2048 flash rel-L2 vs fp64: 0.001650 / 0.001661 / 0.001667. Composed error was the same; flash/composed max absolute difference was 0.000977.
- L=13365 fp64 reference on 16 sampled query rows: rel-L2 0.001671; three outputs were bit-identical.
- The first M2 wheel measured 9,928.0 ms (0.52 TFLOP/s; 418 q-tile dispatches averaged 23.8 ms). The G13C Linux validation-host wheel measured 10,408.6 ms for 5,121,485,107,200 FLOPs (0.49 TFLOP/s; 418 q-tile dispatches averaged 24.9 ms). Both are below the ~40 ms firmware timer; validation-host dispatches are 2.1–5.0x the 5–12 ms register-promotion expectation, so these timings do not establish whether private arrays spill.
- M2 composed L=2048 59.0 ms; L=4096 245.6 ms; bf16 8192-square matmul 3.25 TFLOP/s. Linux validation-host composed L=2048 71.3 ms; L=4096 287.3 ms; bf16 8192-square matmul 2.67 TFLOP/s.
- At H=56, L=4096, composed SDPA measured 0.287 s versus 0.987 s for forced scalar flash (3.4× slower). This is why the default no longer routes on sequence length alone.
- The scalar kernel misses the original 0.5–0.7 s/call target by 15–21x on the Linux validation host. A 4x cooperative-matrix speedup would still be about 2.60 s/call, above target; the QKᵀ/PV cooperative-matrix upgrade needs a larger gain.
- For 48 layers × 8 denoise steps (384 attention calls), current Linux validation-host scalar latency projects to 3,997 s (66.6 min) for attention alone. The 0.5–0.7 s/call target projects to 192–269 s (3.2–4.5 min); the 2.23 s scalar-FMA ceiling projects to 856 s (14.3 min). The measured scalar route is 15–21x slower than target; a cooperative-matrix QKᵀ/PV implementation is required to approach the target. Estimates exclude projections and all other model work.

The corrected live-context harness casts normalized Q/K back to bf16 to match the stated SDPA contract. At L=2048, attention maxdiff was 0.000488 and final maxdiff 0.015625. At L=13365, two live full forwards took 20.53/20.60 s and were bit-identical. Both passed. This was a harness correction, not a kernel fallback.

Rebased-tree battery on the stale golden-cache build: regression passed (2/2, 16 assertions), runtime passed (42/42, 22,731 assertions), and fast-ops failed (42/43, 1,383,891/1,383,920 assertions passed). Its remaining non-SDPA failure was `rope_rms_norm vjp matches the composed chain and host differences`, throwing `[RoPE::vjp] vjp through the fused rms-norm rope is not supported`; the 29 SDPA VJP assertion failures were marked Allowed to fail. The current main patch delegates this VJP through `Custom::vjp`; the staged source lacked it. A fresh prepare staged `Custom::vjp` at `mlx/fast.cpp:970`, then the M2 reboot interrupted the wheel build before suite rerun. On jw16-linux, a clean build clone at `/var/tmp/longsdpa-jw16-fresh` staged the current patch. The first build stopped at the pinned whole-bundle gate; the bundle-enabled retry exited 1 after 27 seconds while another native build was active and produced no wheel. The fast-ops test file is unchanged by the LongSdpa branch.

The current-base fresh wheel from source `f90387545` built successfully on jw16: `/var/tmp/longsdpa-jw16-currentmain/dist/mlx_omarchy-0.32.4.dev202610052024+f9038754-cp314-cp314-linux_aarch64.whl`, 416,576,726 bytes, SHA-256 `6ac939aa6b0b8484b464c9d7b20f6ce077ff9890494e7471ea1be1dfb8ff680a`. It was installed into `/var/tmp/longsdpa-jw16-venv`; import reported `0.32.4.dev202610052024+f9038754` and `Device(gpu, 0)`. The current-base native routing test passed (6 cases, 28 assertions). The sequential native battery exited 0: `omarchy_fast_ops_tests` (43 cases; 1,306,975/1,307,001 assertions passed, with 26 SDPA VJP mismatches explicitly marked `Allowed to fail`); `omarchy_fast_regression_tests` (2 cases, 16 assertions); `omarchy_runtime_tests` (47 cases, 22,890 assertions). The runtime suite emitted queue-watchdog child diagnostics before reporting `Status: SUCCESS!`.

## Status

- [x] Rebased worker branch onto latest main `979f1707`.
- [x] Historical numeric, determinism, and corrected live-context gates on the Linux validation host; receipts and logs are listed above.
- [x] Compare forced scalar flash with composed SDPA at H=56, L=4096; composed is 3.4x faster.
- [x] Built the fresh wheel on `771285b9`; latest main `979f1707` landed afterward and needs one more build.
- [x] Ran routing and all native suites on `771285b9`; latest main `979f1707` needs one more rerun.
- [ ] Build/test the rebased `979f1707` tree after the jw16 quiet window, then land scalar kernel on main.
- [ ] Coopmat QK/PV upgrade: preregistered in private `LongSdpa2` lab entry; test-first implementation and Linux-host numerical/live/performance gates remain with LongSdpaCoop.
- [ ] TensorFoldPort end-to-end H3 comparison against macOS reference latents. Keep latest wheel and venv until TensorFoldPort confirms its forward is complete.

Private raw logs and notebook entry are under the LongSdpa2 Apple Silicon lab artifacts. Main commits worker notebook directories.
