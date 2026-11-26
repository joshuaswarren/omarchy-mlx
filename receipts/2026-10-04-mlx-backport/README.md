# Upstream MLX backport triage

Baseline: MLX 0.32.3, commit `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8` (`mlx.lock`). Upstream merge commits below were checked against that pinned commit. Backports are applied by `scripts/prepare-mlx.sh` after the Omarchy patches, in upstream first-parent order.

## Candidates

| PR | Upstream merge commit | Disposition |
|---|---|---|
| #4556 | `f5cffe9456b7a0accede0a17d812493a1fc6f770` | Backported as `mlx-eval-cleanup-deadlock.patch`; finalizes open GPU streams before synchronization. |
| #4563 | `83b976ea438466418dea20c89709601201851fe1` | Already in pinned baseline. Upstream implementation is Metal fused-SDPA VJP; it does not fix Omarchy's custom fused-VJP value defects. Keep the existing OOB fix and documented may-fail cases (qL=1 maskless dk/dv zero; B=1, kL=5 spots); no evidence supports removing them. |
| #4565 | `9295197533d586dc26d6814b92899fa088320ab3` | Already in pinned baseline. Upstream GDN VJP work is Metal-specific; it does not change Omarchy's custom-kernel defects. |
| #4575 | `831164aca3ba51522141b0d9dc2a8fd0f931de01` | Backported as `mlx-as-strided-contiguity.patch`; generic view/shared-buffer contiguity calculation. |
| #4585 | `9c5d56902c677f44d3a632d3afbe95e48f2828b2` | Backported as `mlx-view-offset.patch`; guards GPU view evaluation for misaligned offsets. |
| #4586 | `a76fbddbbf226d04d786b6b27649eac8dfbf4d5e` | Already in pinned baseline. |
| #4587 | `f56ab23defaeb8752bc1de9b29e150481a8e02df` | Backported as `mlx-view-last-axis-stride.patch`. |
| #4588 | `6a1b7f778fa720e4502f56449c3adcf1438a1e86` | Backported as `mlx-view-contiguity-flags.patch`. |
| #4589 | `aa4096a34f399138c20d7275c4c719c9182b9a65` | Backported as `mlx-shared-buffer-reshape-contiguity.patch`. |
| #4322 | `25532871329fe4cafdada9fa0941a1cdbfb3a9f2` | Backported as `mlx-vmap-scatter-axis.patch`; generic vmap scatter batch-axis fix. |
| #4549 | `0840c42a7aab95fcd4a39e8bc56aa6887470c043` | Not backported: diff is CPU and Metal reduction code; no Vulkan/shared backend change. |
| #4546 | `49028fb217cd5e639002c6d1fdab2809e75a138f` | Not backported: Metal-kernel-only slice change. |
| #4529 | `ad00ea2ddaf146c7d658fc64ca0187c8882ad623` | Already in pinned baseline. |
| #4590 | `174130d025d2fc1ed036d7978a4eea0435ca0b73` | Already in pinned baseline. |
| #4540 | `4079bec5b1e5bdcbb330816edad335532109efee` | Already in pinned baseline. |
| #4513 | `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8` | The candidate is the pinned baseline commit. |
| #4532 | `ac8be42c1c3668d8adc80931d7eda94e4a033def` | Already in pinned baseline. |
| #4570 | `10f116177182b9eafb783ac59a4edde3b11c9b57` | Backported as `mlx-multioptimizer-empty-group.patch`; Python optimizer empty-group handling. |
| #4592 | `2ae29a98aace140d5cdc1482b3926563247bdf04` | Backported as `mlx-aligned-array-pointer.patch`; pointer alignment check in Python converter. |
| #4523 | `d63f0f939a1822cb95fe031694ed5d135a495958` | Already in pinned baseline. |

## Validation and limits

- `scripts/prepare-mlx.sh` applied the nine new patches to a clean staged source tree with `--fuzz=0`; the source tree was rebuilt from the pinned archive.
- CPU-only Linux dev-box build compiled `libmlx.a` and the targeted Omarchy test binaries. The default aggregate build reaches the unrelated `omarchy_ane_runtime_tests` link failure because the private ANE runtime implementation is unavailable in this checkout. A targeted build of `omarchy_capability_sim_tests`, `omarchy_runtime_tests`, `omarchy_primitive_tests`, `omarchy_fast_ops_tests`, and other selected targets succeeded.
- Capability simulation profiles: all six profile invocations passed (7/7 test cases each).
- `omarchy_runtime_tests`, `omarchy_fast_ops_tests`, and `omarchy_primitive_tests` returned success. This workstation has no qualifying Vulkan device; GPU cases were skipped, so these are not GPU-behavior proof.
- Full standing M1 GPU battery and 2B/4B/9B output-digest comparison were first run on the M2 (jw14m2-linux) at the orchestrator's assignment because jw16 was being reinstalled; see the M2 sections below. v0.7.27 published at 2026-10-04T18:54:25Z; the freeze is over.
- Two test-only `conv1d` calls in `overlay/tests/omarchy/test_conv_gemm_decomp.cpp` were corrected to pass explicit dilation/groups because the pinned source signature otherwise treated Stream as dilation and prevented compiling the suite.

## M2 hardware verification (jw14m2-linux, Apple M2 Max, 2026-10-04)

Host: aarch64 Apple Silicon (M2 Max, Asahi Linux), 12 cores, glslc present. Lane-held GPU windows through `gpu-turn` FIFO tickets (≤25 min each); no reboot, no module reload. jw16 was unavailable (reinstall).

**Wheels (distinct stamps asserted per the A/B stamp rule):**

| side | wheel | sha256 (prefix) | stamp |
|---|---|---|---|
| REF (published v0.7.27) | `mlx_omarchy-0.32.4.dev202610041653+6edd258-cp314-cp314-linux_aarch64.whl` | verified against release `SHA256SUMS` | `+6edd258` |
| CAND (this backport) | `mlx_omarchy-0.32.4.dev202610041920+bc0703c-cp314-cp314-linux_aarch64.whl` | 416,342,801 bytes | `+bc0703c` (branch tip) |

Both built with `scripts/build-wheel.sh` (`DEV_RELEASE=1`, `CMAKE_BUILD_PARALLEL_LEVEL=8`, niced, whole-encoder bundle staged with manifest `08769793…` / program `13c74423…` matching the runtime pin). REF venv = release wheel + `mlx-lm==0.31.3`; CAND venv = identical stack with only the mlx-omarchy wheel swapped (same `mlx-lm` both sides).

**Greedy digests (1-pass protocol, `tools/dfuse/pins_window.sh` adapted with 9B cells; `ordered_records_sha256`, first 16 hex):**

| model | arm | depth | REF (v0.7.27) | CAND (backport) | equal | receipt pin |
|---|---|---|---|---|---|---|
| 2B | tiled/pf/perrow | 64 | `cb3e87705c65497c` | same | yes | `cb3e8770` |
| 2B | tiled/pf/perrow | 128 | `a9a7eef85227ed13` | same | yes | — |
| 4B | default | 64 | `e2c919be0fe285c6` | same | yes | `e2c919be` |
| 4B | default | 512 | `fff6d03be28a2ef9` | same | yes | `fff6d03b` |
| 9B | default | 64 | `26d569c86af27c5b` | same | yes | `26d569c8` |
| 9B | default | 512 | `b899ccced74a77ea` | same | yes | `0315217f` (jw16) |

All 20 REF/CAND cells returned rc=0 with byte-equal digests. REF digests reproduce the jw16 receipt pins on 2B d64, 4B d64/d512 and 9B d64; the 9B d512 digest differs from the jw16-era `0315217f` on BOTH sides equally — a host/driver numeric difference on the longer decode horizon, not a backport delta (the gate is REF==CAND on the same host, which holds everywhere).

**Standing battery:** results appended below after the chained gpu-turn windows.

## M2 battery results

Standing battery, backport wheel build (`bc0703c`), real Vulkan device (Apple M2 Max, G14X-class driver), via chained gpu-turn windows:

- 25/26 suites green: runtime 42/42, primitive 104/104, matmul_family 23/23, kv_ops 16/16, indexing 57/57, reduce 34/34, shape 25/25, linalg 30/30, copy_offset 26/26, distributed 9/9, compiled_tape 13/13, fft_ops 19/19, fft_general 14/14, eig 9/9, take_fill 8/8, conv 13/13, complex 34/34, select_layout 13/13, fast_regression 2/2, scatter_determinism 21/21, eq_math 7/7, fused_chain 36/36, error_contract 3/3, ane_bundle 48/48.
- `omarchy_capability_sim_tests`: all six profiles 7/7 (m1-honeykrisp-fork, m1-stock-no-coopmat, subgroup-size-64, small-shared-memory, no-cooperative-matrix, m1-g13-legacy).
- `omarchy_fast_ops_tests`: 42 cases, 40 passed, **2 failed** — triaged below; **not caused by the backport** (identical failure on the origin/main base tree, same host, same driver; see A/B).

## Failure triage: omarchy_fast_ops_tests (A/B vs origin/main base)

Selected the three implicated cases (`-tc`) and ran them on BOTH trees on the same M2 host:

| tree | result |
|---|---|
| origin/main `33ff979ed` (no backport patches), base build | rc=1 — 3 cases selected: 1 passed, 2 failed (39 skipped of the full suite) |
| backport `bc0703c`-lineage build | rc=1 — same 3 cases: 1 passed (the allowed-to-fail SDPA case), 2 failed (identical rope cases) |

1. **`fused rope offset sweep validates every tolerance band`** and **`fused rope refuses beyond the trig argument limit by name`**: pre-existing test/message drift on main, hardware-independent. `de34407c1` ("trig: one honest contract…") rewrote the gate's refusal text to "exceeds the **trig reduction** limit" (`overlay/mlx/backend/omarchy/primitives.cpp`, `rope_trig_gate`), but both test cases still grep the old string "exceeds the **built-in accuracy** limit" (0 occurrences of the old text in the gate on main; 3 in the tests). The gate itself fires correctly. Out of scope for this backport; the test strings need a follow-up fix on main.
2. **`sdpa vjp dk dv match finite differences at rep=1 (path per gate)`**: the documented OPEN fused-VJP value defects (`docs/known-defects.md`, "SDPA backward fused VJP value defects at small rep=1 shapes") — doctest counts it allowed-to-fail. This M2 run is the first hardware probe of the `B=1, kL=5` signature (previously llvmpipe-only); the failure signatures match the pinned defects.

Conclusion: no backport patch breaks any battery suite; nothing to drop. Zero-CPU evidence in this block: the release-path wheel compiles the CPU primitive fallback refusal in (`mlx-no-silent-cpu-fallback.patch`) and `omarchy_error_contract_tests` passed with it active; the full ecosystem workflow trace (Zero-CPU counter walk) was not re-run in this window and remains a release-gate item.

## Limits

- jw16 (M1 Max) was unavailable (reinstall); the standing battery ran on the M2 per the orchestrator's assignment. The 2B/4B digest REF side reproduces the jw16 receipt pins exactly, which cross-checks the M2 numerics against the jw16 pins where they overlap.
- 9B d512 digest differs from the jw16-era `0315217f` pin on BOTH sides equally (host numerics); REF==CAND holds on every cell.
- The two pre-existing rope test/string mismatches and the open SDPA VJP value defects remain open on main, unchanged by this landing.
