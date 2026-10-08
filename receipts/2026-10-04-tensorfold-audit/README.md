# TensorFold / MiniMax-H3 on omarchy-mlx — compatibility audit (2026-10-04)

Scope pin (orchestrator addendum): the TensorFold LLM audit cites ashhart/
TensorFold **v0.6.5** (tag commit `609ca419abecebdc5a059498a613680bd3aa847f`;
main at audit time == v0.6.5). The H3 engine is the drowzeys fork at
`ea9b6372` (git-describe `v0.5.0-217-gea9b637`; the wrapper pins this as
"TensorFold 0.6.5 + H3 family").

License: TensorFold is **Apache-2.0** (`pyproject.toml:11`, relicensed from MIT
at v0.6.0; the fork carries the same Apache-2.0 LICENSE). Any code ported from
TensorFold into omarchy-mlx (MIT) or into wrapper patches keeps its notices,
license headers, and the Apache-2.0 patent grant; this receipt records that
obligation. The translator extensions and int8 ops below are original
implementations (no TensorFold code copied).

Audit of github ashhart/TensorFold (LLM decode) and the MiniMax-H3 video path
(github drowzeys/keys-Mac-TensorFold-MiniMax-H3-MLX wrapping drowzeys/TensorFold
fork @ `ea9b6372` + mrbizarro/minimax-h3-mlx @ `7919020`) against omarchy-mlx
(branch `agent/tensorfold-port` from main `d24e68e00`; MLX pin per `mlx.lock`,
omarchy wheel mlx 0.32.4-base, mlx-lm 0.31.3).

All paths `tf/` = ashhart/TensorFold clone; `tfd/` = drowzeys TensorFold fork @
ea9b6372; `h3w/` = keys-Mac-TensorFold-MiniMax-H3-MLX; `ref/` = minimax-h3-mlx;
`om/` = this repo's prepared tree `.work/mlx` + `overlay/`. Line numbers from
the audited clones (2026-10-04).

## 1. TensorFold LLM kernels (priority 1 per orchestrator)

Five active Metal kernel packages (`tf/src/tensorfold/kernels/README.md:3-9`):
`glm/flash/v1`, `nemotron/lightning/v1`, `qwen/dense/v1`, `qwen/flash_next/v1`,
`gemma/v1`. 30 `mx.fast.metal_kernel(...)` call sites (script-counted; index in
§1.4). CUDA families (`*/cuda/`) are out of scope for Omarchy.

### 1.1 API inventory -> omarchy status

| API (TensorFold use) | Sites (examples) | omarchy status | Evidence |
|---|---|---|---|
| `mx.fast.metal_kernel` (custom MSL) | 30 call sites, e.g. tf qwen/dense/v1/simd_qmm.py:339, glm/flash/v1/kernels.py:358 | Implemented via MSL->GLSL translator + runtime SPIR-V compile (glslc or glslangValidator) | om overlay/mlx/backend/omarchy/custom_kernel.cpp (translator; forbidden list :425-432; compiler :722-759); docs/custom-kernel-inventory.md:9-16 |
| — simd_sum/max/min/broadcast/shuffle* | nemotron/lightning/v1/kernels.py:67 region, glm/flash/v1/hc.py:178-182 | Translated to Vulkan subgroup ops | om custom_kernel.cpp:488-499 |
| — threadgroup memory + barriers | 141 `threadgroup_barrier` uses; e.g. qwen/flash_next/v1/rows.py:55-99 | Supported (bounded static threadgroup arrays) | om custom_kernel.cpp:487; docs/custom-kernel-inventory.md:13-16 |
| — `simdgroup_matrix` / `simdgroup_multiply*` | 24 + 7 uses; qwen/dense/v1/simd_qmm.py (_MMA body), qwen/flash_next/v1/experts.py:322, stream_experts.py:68-100 | **Rejected**: "unsupported MSL feature `simdgroup_matrix`" — exact compatibility error | om custom_kernel.cpp:425-432; docs/custom-kernel-inventory.md:79-82 (quantized_verifier precedent; hand-port analogue `matmul_coopmat` exists) |
| — atomics | 7 `atomic*` uses | Supported except float atomic add return values | om custom_kernel.cpp:347-389 |
| — textures / dynamic threadgroup mem / serialized scalars | none used by TensorFold | Named errors if hit | om custom_kernel.cpp:425-432; docs/custom-kernel-inventory.md:16 |
| `mx.fast.rms_norm` | 36 | Implemented (omarchy shader + composed arms) | om shaders/fast_norm*.comp; docs/compatibility.md fast-op rows |
| `mx.fast.scaled_dot_product_attention` | 14 | Implemented incl. fused decode arm (bf16 bit-identical arm; f16 arm tolerance 0.01) | docs/compatibility.md:167,184-187; shaders/fast_trio.comp |
| `mx.fast.rope` | 6 | Implemented through composed fallback on Vulkan | docs/compatibility.md:405 |
| `mx.quantized_matmul` | 14 (tf deepseek/v4/rows.py:178, nemotron/lightning/v1/rows.py:416, qwen/flash_next/v1/prefill_mm.py:275,370) | bits=4 and bits=8 supported; mlx-lm Linear shape gated; other mode/bits/group combos raise exact named errors | om primitives.cpp:2901,3003,6271; docs/compatibility.md:188-198 |
| `mx.gather_qmm` | 6 (tf glm/flash/v1/kernels.py:443, qwen/flash_next/v1/prefill_mm.py:280,454) | Implemented (shaders/gather_qmm.comp) | om shaders/gather_qmm.comp |
| `mx.dequantize` | 9 | Implemented (dequant word-read semantics documented) | om primitives.cpp:3003 region; docs/compatibility.md:203-206 |
| `mx.compile` | 17 | Compiled bf16 tapes run by default | docs/compatibility.md "Compiled tape bfloat16 - fixed and re-enabled" |
| `mx.async_eval` / `mx.depends` | 41 / 4 (simd_qmm.py `_compiled(dep=True)`) | Supported (event graph); custom-kernel `dep` buffers handled by translator header path | om custom_kernel.cpp (metadata args); async tests in om tests |
| `mx.metal.is_available()` | 9 gates: tf kernels/device.py:19, glm/flash/v1/{kda.py:285, fused.py:43, kernels.py:351, sparse_attention.py:82}, qwen/dense/v1/lane_gdn.py:192, threads.py:39, prefill_mm.py:215, gpu_sampling.py | **Returns False on Omarchy** (no Metal backend; `no_metal.cpp:11`). All TensorFold custom-kernel paths are therefore gated OFF today; mlx-lm needed repo-local gate patches (`patches/mlx-lm-gated-delta-fast-route.patch:7`) | om .work/mlx/mlx/CMakeLists.txt:99-102; patches/mlx-omarchy-metal-kernel.patch:11-14 (metal_kernel availability separately ORs `omarchy::is_available()`) |
| `mx.metal.device_info()` / `mx.device_info()` | tf kernels/device.py:21, threads.py:39, prefill_mm.py:215 (chip detection) | `gpu::device_info` provided (memory, architecture strings) | om overlay/mlx/backend/omarchy/device_info.cpp:18-40 |
| `mx.get_active_memory/get_peak_memory/reset_peak_memory` | 10+ | Provided via allocator | om allocator.cpp |
| `mx.set_wired_limit` / `mx.set_cache_limit` | 4 / 4 | Provided | om allocator.cpp (wired/cache limit plumbing) |
| `mx.random.{normal,key,split,randint}`, `mx.random.bits` semantics | sampling paths | Preserved incl. `mx.random.bits`/split-key shapes | docs/compatibility.md:306-307 |
| `mx.conv3d` (H3 video+audio VAE) | tfd families/h3/vae_video.py:215; ref minimax_h3_mlx/video_vae.py:171 | Implemented: general direct conv 1D/2D/3D incl. groups/depthwise/stride/pad/dilation | om .work/mlx/mlx/backend/omarchy/primitives.cpp:4225-4245 |
| `mx.load` / `mx.save_safetensors` | tf engine weight loading | Implemented (io device patch) | om patches/mlx-io-device.patch |
| `mx.distributed.all_sum` | tf drafters/vendor/z_lab_dflash/model_mlx.py:605 (optional DFlash drafter) | Not on the critical path for single-host decode; sharding group is optional drafting | tf file cited; om distributed tests exist (tests/omarchy/distributed) |
| `mx.stream` / custom streams | tf z_lab_dflash generation_stream | Supported | om stream.cpp |
| fft / vmap / grad | none found in TensorFold src (grep) | n/a | grep receipt in audit notes |

### 1.2 The load-bearing gate: `mx.metal.is_available()`

Every TensorFold family gates custom kernels on
`mx.default_device() == mx.gpu and mx.metal.is_available()`
(tf glm/flash/v1/kernels.py:351 pattern). On omarchy that conjunction is False:
the Metal namespace is the `no_metal.cpp` stub while `mx.fast.metal_kernel`
itself works through the translator. Consequences per family need one runtime
probe each (§1.4 plan); statically, families whose kernels are optional
fall back to stock paths (slower, still exact-to-themselves); families whose
row-exact QMM is load-bearing (qwen/dense `simd_qmm`) fall back to
`mx.quantized_matmul`-shaped code paths only where the family provides them.

Port requirement: a platform-gate patch (upstream draft PR) replacing
`mx.metal.is_available()` in TensorFold's gates with a capability probe that
is True when the custom-kernel path works on the current backend
(`mx.fast` availability), so the same code enables kernels on both macOS
and Omarchy.

### 1.3 Exactness contract

- Speculative verification is self-consistency: drafted rows are verified
  against the target model's own sample on the same backend
  (tf engine/lane_engine.py:1 "Verify each draft against the target sample and
  roll caches back to accepted rows so shared rounds match serial decoding";
  drafter proposal limited by matching context, lane_engine.py:15).
- Kernel invariant: "A multi-row kernel must match its own serial row before it
  can verify drafts" (tf kernels/README.md:17-18). simd_qmm checks
  scalar/MMA equivalence at install time (tf qwen/dense/v1/simd_qmm.py:1).
- Therefore the Omarchy bar: same-backend self-consistency (speculative output
  == plain decode, token for token) + per-op error vs fp64 reference. Bit-level
  equality with macOS outputs is NOT required by TensorFold's own contract and
  is not the gate here.

### 1.4 Per-kernel classification (static; runtime probe pending)

Translator allows: buffers/scalars, templates, shape/stride/dim metadata,
header helpers, thread/grid attributes, threadgroup memory+barriers, subgroup
ops, atomics, multiple outputs, MLX math modes (docs/custom-kernel-inventory.md:13-16).

- Likely translatable (no simdgroup_matrix): qwen/dense row_* and lane_* glue,
  stream_* , simd_qmm scalar body (`_SCALAR` uses threadgroup staging + per-lane
  chains, tf qwen/dense/v1/simd_qmm.py:37-60), nemotron rows.py:232, gemma
  base/decode/glue unless they use simdgroup types, glm kda/fused/sparse kernels
  using `simd_` reductions only (tf glm/flash/v1/kernels.py:92-96, hc.py:178-182).
- Rejected by translator (simdgroup_matrix family): qwen/dense simd_qmm `_MMA`
  (+`_fragment_source` mmaf), qwen/flash_next experts.py:322,
  stream_experts.py:68-100, any lane_attention MMA tiles
  (qwen/dense/v1/lane_attention.py:193-198 uses simdgroups — verify variant).
- Runtime probe plan: after the wheel lands, call every kernel-building entry
  with tiny shapes on lavapipe (`MLX_OMARCHY_ALLOW_NON_APPLE=1`) and record the
  exact outcome (runs / named error). Table to be appended below with observed
  results.

## 2. MiniMax-H3 int8 path (priority 2)

### 2.1 What the int8 path does exactly (tfd @ ea9b6372)

- Files: tfd src/tensorfold/kernels/minimax/h3/v1/mlp_int8.py (273 ln),
  qkv_int8.py (114 ln). Family wiring: tfd src/tensorfold/families/h3/weights.py
  (`int8_mlp` :73-90, `int8_attention` :92-115; attention softmax itself stays
  bfloat16).
- Scheme (docstring tfd mlp_int8.py:1-8): follows antirez/h3.c — **W8A8**:
  - Weights: int8 (N,K) + one fp32 scale per output channel
    (`quantize_weight`, mlp_int8.py `"""(N, K) float weight -> int8 weight and
    one float32 scale per output channel"""`).
  - Activations quantized at run time: one scale per row (per row AND per
    1,024 channels for the fc2 input: `FC2_GROUP = 1024`, mlp_int8.py:16;
    otherwise per-row `quantize_rows`, mlp_int8.py).
  - Compute: 128x128x128 int8 tiles through **Metal 4 tensor operations** —
    `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`,
    `mpp::tensor_ops`, `tensor<device int8_t, dextents<int32_t,2>, tensor_inline>`,
    `matmul2d_descriptor(T,T,T,false,true,true, multiply_accumulate)`,
    `execution_simdgroups<8>`, cooperative destination tensor with **int32
    accumulation** (mlp_int8.py:20-60). Rescale `xscale[row,group]*wscale[n]`
    in fp32, optional fp32 bias, output bf16 (`int8_linear`, `int8_swiglu`).
  - Layers: every block's SwiGLU MLP (fc1 fused with SiLU: W packed (2N,K),
    gate rows first) via `Int8MLP`; QKV and attention-output projections via
    `Int8Linear`/`Int8QKV` (qkv_int8.py: fused q/k norm + rotation inside the
    QKV kernel, head-major outputs). Zero-point: none (symmetric int8, QMAX=127).
- M5-only upstream: `available()` compiles+runs the kernel on the host GPU and
  returns False on failure (mlp_int8.py:158-167); non-M5 Macs -> `int8_mlp`
  returns 0 and the model stays bfloat16 (weights.py:84-86). The wrapper
  oneshot-setup.sh prints the same gate (h3w oneshot-setup.sh:37-38: "M5: int8
  kernels on the tensor units ... without them the engine runs bfloat16").
- Wrapper platform gates: h3w oneshot-setup.sh:33 (Darwin+arm64 die), :36
  (RAM >= 128 GB die — comment: "float32 adapter merge peaks at 103 GB
  (measured on 256 GB only)"), :37-38 (M5 chip case). Engine payload: venv
  install of the pinned fork + minimax-h3-mlx + Turbo LoRA adapter
  (oneshot-setup.sh:19-24); pins h3w requirements.lock:23,25 — **mlx==0.32.3,
  mlx-lm==0.32.0** vs our wheel mlx 0.32.4-base / mlx-lm 0.31.3.

### 2.2 Reference repo (minimax-h3-mlx @ 7919020)

Stock MLX only: `mx.conv3d` (video VAE ref minimax_h3_mlx/video_vae.py:171;
diffusers autoencoder_kl_minimax_h3.py:65), common elementwise/reduce ops,
`mx.quantized_matmul` in scripts/bench_gemm.py:63 only, **no custom Metal
kernels, no int8** — the reference is the bf16 baseline
(`--parity` compares TensorFold vs reference: h3w h3_generate.py:94-127
reports max abs diff / relative diff).

### 2.3 Port design (per ticket)

1. Backend op surface (omarchy-mlx): `mx.fast.int8_matmul` (linear + bias) and
   a fused `int8_matmul_swiglu` — int8 x int8, exact int32 accumulation
   (K-chunked so int32 cannot overflow: 127*127*4096 << 2^31), fp32 group
   rescale, bf16 out; matches reference int8 math bit-for-bit on the integer
   path and beats/exacts the fp32 rescale in fp64 comparison. Vulkan compute:
   unpack int8 (u32 words) + i32 MACs, or `VK_KHR_shader_integer_dot_product` /
   coop-matrix int8 where the device exposes it (feature query on the M1 Max /
   M2 Max hosts before choosing). Same Python entry points TensorFold calls;
   the fork's `mlp_int8.py` gets an Omarchy branch behind a platform gate
   (upstream draft PR; exact compatibility error stays for MPP-MSL sources).
2. `mlp_int8.available()` on Omarchy: True once the native ops are present.
3. Wrapper gates: Darwin/arm64 die -> warning; 128 GB die -> warning with
   measured-need number (upstream draft PR).

## 3. mlx-lm 0.32 assessment

TensorFold imports mlx_lm.models.{gemma4_text (logit_softcap, geglu),
gated_delta, qwen3_next, qwen3_5, switch_layers (_gather_sort/_scatter_unsort),
cache.KVCache} (tf greps in §audit notes). mlx-lm 0.31.3 (our wheel) predates
several of these module names; assessment: install mlx-lm==0.32.0 into a
throwaway venv with `--no-deps` over the omarchy wheel and import-check each
module TensorFold needs; record which imports fail. (Pending run; mlx 0.32.3 vs
0.32.4-base C-API drift to be probed by the same import check.)

## 4. Runtime verification (dev box, lavapipe + glslangValidator, wheel
`0.32.4.dev202610041734+d24e68e00`, `MLX_OMARCHY_ALLOW_NON_APPLE=1`)

Observed 2026-10-04 (probe scripts + raw outputs archived in the private
artifacts store; run receipts below are quoted from those outputs):

- `mx.metal.is_available()` -> **False** at runtime on Omarchy (matches
  docs/custom-kernel-inventory.md:10). Every TensorFold custom-kernel gate is
  therefore closed by default.
- `mx.fast.metal_kernel` works on the dev stack (float32/bfloat16/float16/uint32
  buffers, implicit bf16<->f32 promotion, `simd_sum` -> subgroupAdd, constexpr +
  threadgroup arrays + barriers, fma, `<name>_shape` metadata all pass).
- TensorFold `simd_qmm kind="mma"` -> exact compatibility error, verbatim:
  `[omarchy] fast::CustomKernel MSL subset: unsupported MSL feature
  'simdgroup_matrix' is not implemented for the Omarchy Vulkan backend
  (dtype=bfloat16, shape=[8,1536]). No GPU kernel exists for it; no silent CPU
  fallback occurs.` (product contract behavior confirmed).
- TensorFold `simd_qmm kind="scalar"` -> reaches the translator and fails on
  body constructs, not on the signature: `unsupported MSL syntax remains after
  translation`.
- Construct-level battery (body dialect, MLX fast.metal_kernel conventions):

| Construct | Result | GLSL evidence |
|---|---|---|
| plain math, bf16/f32/f16/u32 buffers | OK | |
| explicit `bfloat(expr)` cast in body | FAIL | `'bfloat' : no matching overloaded function found` |
| `as_type<float>(u)` bitcast | FAIL | `'as_type' : undeclared identifier` |
| `thread_index_in_simdgroup` / `simdgroup_index_in_threadgroup` | FAIL | `gl_SubgroupInvocationID: required extension not requested: GL_KHR_shader_subgroup_basic` (translator emits the builtin but no `#extension`) |
| `(const device uint4*)ptr` vector load | FAIL | GLSL syntax error on emitted uvec4 load |
| `uint2(scalar)` construct | FAIL | GLSL compile error |
| `const device T* arr[N]` pointer array | FAIL | GLSL compile error |
| `_Pragma("clang loop unroll(full)")` | FAIL | GLSL compile error |
| `atomic_fetch_add_explicit((device atomic_uint*)in, ...)` body-level | FAIL | translation leftover (declared atomic OUTPUT params are the supported form) |
| `simd_sum` | OK | subgroupAdd emitted with extension |
| constexpr + `threadgroup float[]` + `threadgroup_barrier` | OK | |
| `fma`, `<in>_shape[0]` metadata | OK | |

Implication: TensorFold's MSL bodies routinely use `bfloat(...)` casts,
`as_type` bitcasts, vector pointer loads and `_Pragma` — none translatable
today. Extending the translator with these general mappings (bitcast helper,
bf16 pack helper, vector pointer-load helper, pragma strip, subgroup-basic
extension request, scalar->vector construct) is the smallest path to running
TensorFold's scalar kernels unmodified; the simdgroup_matrix MMA kernels stay
on the exact-error path and need the wrapper's omarchy-native route
(qmm scalar path or a native qmm/int8 op).

## 5. C++ suite baseline on the dev stack

`ctest` over the built test tree on lavapipe: 69% passed (83 failed of 270,
most "Not Run" unbuilt targets; named failures: fused rope_rms_norm bit-exact,
rope offset sweep, fast_ops_tests, two_rank_harness needs two devices). This is
the dev-box environment baseline BEFORE any of this branch's changes; rope
trig failures are the known llvmpipe trig-precision surface (g15-trig gate runs
on real hardware). Re-run the same set after implementation changes and compare
against this baseline, not against a claim of zero failures.

## 6. Runtime verification plan (remaining)


1. Dev box (lavapipe, `MLX_OMARCHY_ALLOW_NON_APPLE=1`): wheel import probe;
   per-kernel translator outcomes; `mx.metal.is_available()` value; doctest
   runs for the new int8 ops vs fp64.
2. M1 Max host (gpuwin windows, glslc): shader build + numeric parity of the
   int8 ops and translatable TensorFold kernels on real Honeykrisp.
3. M1 Ultra macOS host: stock H3 wrapper reference run (no adapter merge first,
   65 GB peak), memory_pressure gates, bf16 path expected (non-M5), outputs
   kept with sha256.
4. M2 Max host (gpu-turn window): gates patched to warnings, int8 ON, PSNR/SSIM
   + audio metrics vs the macOS reference.

## 7. Weight component sizes (macstudio FL2VA, measured 2026-10-04)

transformer (DiT) 62 GiB bf16, text_encoder 63 GiB, video_vae 10 GiB,
audio_vae 1 GiB. M2 (~48 GiB free) cannot hold any bf16 combination; the
feasible Omarchy configuration is a quantized DiT (the reference repo's own
4-bit core + 8-bit AdaLN recipe) plus pre-encoded prompt conditioning
(skipping the 63 GiB encoder), ~30 GiB resident — the pre-encoding needs a
small upstream flag (draft-PR list). The int8 W8A8 swap path still works from
a quantized-DiT-free bf16 checkpoint: note the swap quantizes from whatever
weights the projections hold, so the DiT must stay bf16/8-bit for the int8
path; a 4-bit core checkpoint serves bf16-comparison runs only.

## 8. M1 Ultra reference run (2026-10-04, bf16, no adapter merge)

Stock wrapper main on macOS 26.6.2, M1 Ultra 128 GB, weights on the main
drive (134 GiB FL2VA; components: DiT 62, text encoder 63, video VAE 10,
audio VAE 1 GiB). Config 768x448, 124 frames, points 4, seed 0, no lora,
no int8 flags. Result: load 19.2 s (weights resident, no per-step streaming;
peak 61.8 GiB), denoise 269.8 s over 3 forwards (86-90 s each), decode
106.2 s (video VAE 99.1), mux 2.75 s; memory gate min 53.4 GB free+inactive
(20 GB abort line never approached). Output mp4 sha256
be52043ee7df15989f28ad883bb349030b9567f025c022165f5c78f21a17f5f9; frames
0/60/123 and all logs archived in the private artifacts store with
SHA256SUMS. Setup-verify on this NON-M5 host printed "int8 tensor-unit
kernels available": the MPP int8 mma compiles on M1 Ultra, so "M5-only" is a
performance claim, not an availability claim; an int8-flags run is a cheap
follow-up.

## 9. M1 Ultra int8-flags run (in progress at entry time)

Config identical to the bf16 reference plus --int8-mlp --int8-qkv --int8-out.
Log confirms the INT8 PATH TAKEN on M1 Ultra: "int8 MLP in 50 blocks",
"int8 attention projections: 100" — the MPP int8 mma compiles and runs on
this NON-M5 host. Per-step cost so far: 106.1/104.7/104.7 s vs the bf16
reference's 90.0/87.4/86.1 s — int8 is ~17% SLOWER per forward on M1 Ultra,
consistent with "M5-only" being a tensor-unit performance claim. Memory
notably lower during denoise (68-71 GB free+inactive vs 53-57 GB bf16 — the
int8 resident set is smaller). Comparison (PSNR/rel-L2 vs bf16 frames
0/60/123) lands with the output hash in the artifacts store.

## 10. Heap-corruption debug state (C-cast scanner, quarantined)

Reproduction (dev box): TensorFold H3 `quantize_rows` (the `_QUANTIZE`
custom kernel) through the current branch build aborts with
`free(): invalid next size (fast)` x2 during eval. ASAN follow-up: an
asan-flagged wheel was built and run under `LD_PRELOAD=libasan.so`, but the
asan interceptor CHECK (`real___cxa_throw == 0`) fires first because the
interpreter is not asan-built; the throw under investigation is
`compile_glsl`'s glslang failure path. Proper harness (next session):
asan-built CPython, or a small asan-built C++ driver that exercises
`fast::CustomKernel::eval_gpu` directly. The corrupting site is inside the
cast scanner's replace/span arithmetic (commit f64c97681 reverted it from
the build; the experimental scanner lives in later WIP commits).

## 11. M2 run staging state (2026-10-04 late)

Leg 1 of the shipment complete and manifest-verified on the dev relay
(int8-dit 7 shards 25.8 GiB + text rows 5.3 MB + int8 latents 5.1 MB +
sidecar; 10/10 sha256 OK). Leg 2 (dev -> M2, resumable) waits for the M2 to
return from the Thunderbolt macOS window. The M2 ticket script is staged
(int8_matmul numeric validation vs fp64; VAE decode deferred until VAEs
ship). Wheel coordination: the shared M2 wheel must be built from main
793546208+ (int8_matmul landed in 33ff979ed); flagged to Main — an older
base silently misses the op.

## 12. M2 H3 ticket (2026-10-05 ~00:2xZ): PASS

Shared wheel b8af62c (main + distributed branch; int8_matmul present). Under
gpu-turn (nice -n 19): `mx.fast.int8_matmul` numeric validation on M2
Honeykrisp REAL GPU — max|err| 0.415 vs one-bf16-round bound 0.560 (fp64
reference) = PASS. First real-GPU validation of the landed op outside
lavapipe. Ticket log ~/tfport/m2-h3-ticket.log on the M2. VAE decode deferred
until the 11 GiB VAE weights ship to the M2 (Lead's disk call).

## Depth bisect (2026-10-08): the Linux Vulkan int8 path is the defect
Pre-registered rule: first K where rel-L2(linux_native_K, mac_ideal_K) > 1e-2 x the per-K
bf16 round-trip floor (rel-L2(mac_int8_K, mac_bf16_K)). Result (files in the lab artifacts
dir, DEPTH-BISECT.md): native diverges from K=1 (ratio 5.745; 2.0-2.6 at K>=2); Linux
plain-ops == mac ideal at every K (ratios 0.007-0.014). Full depth: native vs plainops
1.1497 video; plainops vs bf16p2 0.0431. VERDICT: (a) the Vulkan int8 kernel/graph path
is the defect; the quantization design and the other Linux ops are sound; the version
difference is not implicated (Linux plain-ops == mac plain-ops ideal).
