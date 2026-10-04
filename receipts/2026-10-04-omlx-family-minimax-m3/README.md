# minimax_m3 (MSA CSR attention) — omarchy-native implementation

Owner lane: FamMinimaxM3.
Pins: oMLX v0.7.0 = 4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40 (custom_kernels/minimax_m3/).
omarchy-mlx base: 14bf031a1 (this branch agent/fam-minimax-m3).

## What this lane owns

The `minimax_m3` family has two delivery surfaces and both currently fail
on Linux:

1. **Prebuilt metallib** (`custom_kernels/minimax_m3/csrc/minimax_msa.{cpp,metal}`).
   Built on macOS with `xcrun -sdk macosx metal` → `omlx_minimax_m3_kernels.metallib`,
   loaded by the C++ extension at runtime (a `mlx::core::Primitive` named
   `MinimaxMSATopKPrimitive` exposes `minimax_msa_topk`). Two Metal kernels:
   `minimax_msa_block_scores_*` (per-row-block Q·K^T dot products reduced by
   atomic-max, using the upstream MLX `mlx::steel::GEMMKernel` template) and
   `minimax_msa_topk_select_topk16_t256` (per-row top-k block selection with
   init/local forced inclusion). On Linux the metallib never exists, so the
   `is_native_available()` path returns false and the patch routes to the
   Python fallback `build_grouped_msa_topk`.

2. **JIT `mx.fast.metal_kernel` patch kernels** at
   `omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/msa.py`:
   `_MSA_CSR_K1_STEEL_MMA` (line 344, the explicit target), `_MSA_CSR_K1_SCALAR`,
   `_MSA_CSR_K1_SIMD`, `_MSA_CSR_K1_SIMD_PACKED`, `_MSA_CSR_K2`, `_MSA_TOPK_SELECT`,
   `_MSA_DECODE_B1_SIMD`. The MSL→GLSL translator refuses `simdgroup_matrix` /
   `tile_matmad` (matrix.h top-of-file `forbidden` list,
   `overlay/mlx/backend/omarchy/custom_kernel.cpp:472-481`), and the steel-MMA
   kernel relies on `simdgroup_matrix` and `tile_matmad` to do the K1 Q·K^T
   block product. The remaining JIT kernels are simpler scalar / simd loops
   and translate to GLSL in principle; the C++ extension's
   `scalar_arguments_` path is rejected by the omarchy translator
   (custom_kernel.cpp:1291-1294), which the steel-MMA kernel also depends on
   (the `scale` float scalar).

   Net: every JIT kernel in this family fails today with the omarchy backend.

## Plan (this branch)

The acceptance bar (from parity matrix v1.1 A25b): "real omarchy
implementation (translated GLSL or native Vulkan) + numerics-gate parity +
speed vs generic fallback (or fallback proven at/above macOS speed, same
machine)".

This is large. Split into four deliverables, each independently verifiable
and each small enough to keep commits reviewable:

### D1 — translator extensions: scalar arguments + simdgroup_matrix exclusion path

C++ changes in `overlay/mlx/backend/omarchy/custom_kernel.cpp`:

- Add scalar-argument support. The upstream `fast::CustomKernel` already
  stores `std::vector<ScalarArg>` where `ScalarArg = std::variant<bool, int, float>`.
  On the Metal path, scalars are bound as `set_bytes` after the buffers. On
  the omarchy translator path, we currently error out
  (`fast::CustomKernel MSL subset: serialized scalar arguments`,
  custom_kernel.cpp:1291-1294). The translator's GLSL output needs push
  constants for scalars (or a tiny uniform storage buffer). Approach:
  - Append a `push_constant` block at the top of the generated GLSL with
    one uint slot per scalar argument (bitcast from bool/int/float at the
    call site), so the dispatcher binds scalars via
    `VkCommandBuffer::pushConstants` instead of storage buffers.
  - Reflect the scalar name into the body as a const value the kernel
    reads.
  - Keep the contract: any kernel that uses a named push constant is
    dispatched through the same `omarchy::ComputeParams`+`ComputeBinding`
    path; only the per-pipeline push-constant range changes.
- Add a translator `[[simdgroup_matrix]]`/`tile_matmad` short-circuit that
  emits a translated GLSL `coopMatMulAdd` (the GLSL `GL_KHR_cooperative_matrix`
  extension) when it can prove the kernel is a `MMATile` matmul on the
  shapes steel-MMA is asked for. Otherwise, the existing exact-error stays.
  Steel MMA's K1 needs `B*kFragSize` reductions; the omarchy coopmat path
  has its own matmul shape (`shaders/matmul_coopmat.comp`, 32x32 output tile,
  8-wide k). Bridging the two is a real implementation: D3.

### D2 — translator: full coverage of the simple JIT kernels

C++ changes in `overlay/mlx/backend/omarchy/custom_kernel.cpp`:

- Extend the body translator to support the subset of MSL the remaining JIT
  kernels need: `threadgroup_barrier(mem_flags::mem_none)`, `metal::exp`,
  `metal::log`, `metal::log2`, `metal::exp2`, `metal::min`/`max` for
  non-constant args, `int3`/`uint3`, `simdgroup_barrier` over the same
  flags, the `MTL::Size`-style grid via `gl_WorkGroupID`/`gl_LocalInvocationID`
  (already covered), `if (work >= total_work) return;` early-out
  (already covered), `bool` constants and `?:` ternaries on bool.
- After D1, `_MSA_CSR_K1_SCALAR`, `_MSA_CSR_K1_SIMD`,
  `_MSA_CSR_K1_SIMD_PACKED`, `_MSA_CSR_K2`, `_MSA_TOPK_SELECT`,
  `_MSA_DECODE_B1_SIMD` should all translate and dispatch on the dev-box
  lavapipe stack (M-class Vulkan) without changing the calling Python.
- `_MSA_CSR_K1_STEEL_MMA` is still refused with an exact error; D3.

### D3 — coopmat K1 reimplementation (the explicit msa.py:344 target)

New compute kernel enum (overlay/mlx/backend/omarchy/compute.h), new
`.comp` shader, and a primitive dispatch in `overlay/mlx/backend/omarchy/primitives.cpp`.

- New shader: `overlay/mlx/backend/omarchy/shaders/minimax_m3_csr_k1_coopmat.comp`
  modeled on the existing `matmul_coopmat.comp` (8x8x8 fp32 coopmat, 32-thread
  workgroup, shared-mem staging with bf16 widening when `A_BF16`/`B_BF16` are
  defined). Dispatch: one workgroup per `(h_kv, row)` of the K2Q CSR, with
  `Q_TOKENS_PER_GROUP` queries staged in a shared A tile and the block-sized
  K/V pair streamed in as 8-wide k steps. The Q staging matches the steel
  MMA's MMATile<TQ,1> (a single row of the K1 output) and the K staging
  reads the same `k[(k_pos * H_KV + hkv) * D + d]` view the steel-MMA
  body reads; the only difference is the matmul, which the omarchy
  coopmat path already proves numerically bit-identical to the cast +
  f32 matmul + cast chain (see numerics-gate receipts, matmul_coopmat
  parity).
- Online softmax: identical to the steel-MMA body, using the existing
  `row_reduce<MSA_MaxOp>` and `row_bin_op<MSA_Exp2SubOp>` shapes the
  vendor msa.py already pulls in. Output is `[topk, total_q, h_q, d]`
  partials + LSE partials, matching the upstream `_MSA_CSR_K1_*` outputs.
- Primitive: `MinimaxMSACSRK1Coopmat` (subclass of `omarchy::Primitive`)
  registered in `primitives.cpp`'s `MinimaxMSA` dispatch block; gates on
  `D == 128`, `block_size == 128`, `qhead_per_kv == 16`, `total_q >= 8`.
  When the gate is missed, the existing `unsupported()` exact-error
  fires so the Python fallback (`k1_impl="scalar"` / `"simd"`) still works.
- The C++ extension's prebuilt metallib path stays macOS-only; on Linux
  the `minimax_msa_topk` Python shim continues to fail back to the
  Python topk builder (D4).

### D4 — generic-fallback parity + speed numbers

Python harness under `tests/local-omarchy/test_minimax_m3_parity.py`
(self-contained, not a public test — the public test suite is the lane
gate). It exercises:

- `build_grouped_msa_topk` (Python) — the live fallback. Capture:
  per-element parity against a small CPU ref, and a wall-clock.
- After D1+D2 land, the same harness exercises `_MSA_CSR_K1_SCALAR`,
  `_MSA_CSR_K1_SIMD`, `_MSA_CSR_K1_SIMD_PACKED`, `_MSA_CSR_K2`,
  `_MSA_TOPK_SELECT`, `_MSA_DECODE_B1_SIMD` on the dev box. Capture
  per-op parity vs the Python fallback (within bf16 tolerance) and
  a wall-clock number on lavapipe; the same harness runs on the M2
  ticket when one is available.
- After D3 lands, the harness runs the coopmat K1 end-to-end
  (`msa_sparse_attention_b1_from_csr` with `k1_impl="coopmat"`) and
  compares to the SIMD K1 / scalar K1 numerics-gate (per-op fp64
  within the tolerance the existing receipt states; teacher-forced
  top-1 within 99%; PPL within 0.1%).
- All numbers recorded in
  `receipts/2026-10-04-omlx-family-minimax-m3/`.

### D5 — row update + lane-send

- Update parity matrix row A25b (and any sibling rows whose `minimax_m3`
  evidence changes) with the closure status, evidence file:line, and
  speed number, and post the lane-send `FamMinimaxM3` at each milestone
  (design + first kernel on dev lavapipe; first M2 parity+speed; landed).

## Non-goals (this lane)

- Building a real m3-vl model on the M2 (no MiniMax-M3 weights in the M2
  HF cache; would need >5 GB, requires Main approval per the lane rules).
- Re-introducing `simdgroup_matrix` or `tile_matmad` translator support.
  Exact-error stays, per the omarchy-mplus contract.
- Re-touching the C++ extension's macOS-only metallib path. The prebuilt
  metallib's `MinimaxMSATopKPrimitive` is correct on macOS; on Linux it
  simply never loads.

## Risks (recorded now)

- The C++ extension `minimax_msa_topk` calls `metal::device::get_library`
  with the metallib name, which on Linux never resolves. The Python shim
  has the `if _ext is not None` branch and falls back to `mx.fast`; the
  current `mx.fast.minimax_msa_topk` is provided by the same metallib
  path through MLX, which is not in the omarchy wheel. So even after
  D1+D2 land, `build_grouped_msa_topk` on Linux will exercise the
  Python topk path until the `omlx_minimax_m3_kernels.metallib` is
  built natively — which the lane does NOT own. The Python topk path
  is what we measure against; the native C++ path is `minimax_msa_topk`
  with the metallib and stays macOS-only.
- The C++ `MinimaxMSATopKPrimitive` is correct and fast on macOS; we
  do not port it (no Metal toolchain on Linux, and the rule says "do
  not emulate Metal"). On Linux the family has no native prebuilt
  metallib. The closure bar is therefore: omarchy has real K1 + K2 +
  topk kernels (D1+D2+D3) plus a Python topk fallback for the
  prebuilt extension, and a numerics-gate proof.
