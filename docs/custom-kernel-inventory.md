# Custom-kernel inventory: does `fast::CustomKernel` need a compiler?

Date: 2026-09-09. mlx-omarchy commit `8eb24c4` (MLX 0.32.2, pin `1f8e74e`).
Upstream sources scanned: mlx-lm `2184db2`, mlx-examples `796f5b5`,
mlx-vlm `8f5dc3d`, mlx-audio `17001a6` (all 2026-09-07..09).

## Current implementation

As of 2026-09-10, mlx-omarchy routes `mx.fast.metal_kernel(...)` calls to
the Omarchy GPU backend while keeping `mx.metal.is_available()` false. The
backend translates the MLX-generated signature and a bounded MSL kernel subset
to GLSL, compiles it to SPIR-V with the installed shader compiler, caches the
Vulkan pipeline, and dispatches it without a CPU fallback.

The supported subset covers the signatures and language features exercised by
the targeted qualification suite: array and scalar buffers, templates, shape,
stride and dimension metadata, header helpers, thread and grid attributes,
threadgroup memory and barriers, subgroup operations, atomics, multiple
outputs, and MLX math modes. Textures, precompiled libraries, dynamically sized
threadgroup memory, and serialized scalar inputs stop with named errors.

## Inventory baseline

Before this path existed, `mx.fast.metal_kernel(...)` failed at call time in
`resolve_metal_kernel_stream` because `metal::is_available()` was false.
The inventory below records the user impact and kernel corpus measured on
2026-09-09; its compiler recommendation is superseded by the implementation.

## Inventory method

`git grep -n -E 'metal_kernel|CustomKernel'` in each repo (receipt: private
`receipts/customkernel-spike/01-git-grep-four-repos.txt`), then read every
kernel source body, its call-site gate, and its fallback. GitHub code search
for `mx.fast.metal_kernel` / `fast.metal_kernel` (201 repos, lower bound;
receipt `02-github-code-search.json`). Hugging Face: 822 unique `mlx`-tagged
repos (500 most-downloaded + 500 most-liked), listed `.py` siblings for each,
fetched and grepped every one (receipt `03-hf-scan.json`).

105 call sites in the four repos (35 files). Deduping identical copies shared
across repos (bitlinear, ssm, rwkv7, gated_delta, encodec lstm) gives ~95
unique kernels.

## Classification

### (a) Covered: gate exists, pure-MLX fallback runs today (perf gap only)

| Kernels | Where | Hot path? |
|---|---|---|
| `gated_delta_step{,_vec,_mask,_vec_mask,_xtree,_packed_btree}` (6 fwd paths, 3 call sites) | mlx-lm `models/gated_delta.py`, mlx-vlm `models/gated_delta.py`, `qwen3_5/gated_delta.py` (+2 `with_states`) | Yes: Qwen3-Next / Qwen3.5 / Kimi Linear decode+prefill. Fallback is a per-token Python loop (mlx-lm) or `gated_delta_chunked` parallel scan (qwen3_5, ~20x faster than the loop). |
| `ssm_kernel`, `ssm_initial_kernel` | mlx-lm/vlm/audio `ssm.py` | Decode only (seq_len==1); prefill already uses chunked `ssm_attn`. Fallback correct. |
| `wkv7_kernel` | mlx-lm/vlm `rwkv7` | Yes for RWKV-7. Per-token loop fallback. |
| `attnres_mix`, `k3_short_conv_step` | mlx-lm/vlm `kimi_k3` | Kimi K3 decode. Ops fallback exists. |
| `kl_forward/backward`, `js_forward/backward` | mlx-lm `tuner/losses.py` | Training only (quant calibration). `nn.losses` fallback. |
| `lstm` (encodec) | mlx-examples + mlx-audio `encodec.py` | Audio codec inference. Composed LSTM math. |
| `relu_squared`, `fused_multiply_add` | mlx-audio mossformer2 | ReLU2 attention; `maximum`/multiply cover it. |
| `nearest/bicubic/grid_sample/separable_interpolate` | mlx-vlm `models/kernels.py` | Vision preprocess. In-file pure-MLX fallbacks. |
| `mrope_apply_*`, `rotary_apply_*` | mlx-vlm `models/rope_utils.py` | Every VLM. Pure-MLX fallback + custom_function VJP. |
| `hc_sinkhorn_collapse`, `short_block_hc_{norm,normalized_norm,expand}`, `short_block_affine{4,5}_{switch_gate_up,moe_down}` (6) | mlx-vlm `deepseek_v4/hyper_connection.py`, `fast_ops.py` | DeepSeek-V4 + switch layers. Return-None-to-fallback pattern. |
| `exact_speculative_verify_{gemv,switch_gemv}` | mlx-vlm | Speculative verify fast path. Optional. |
| `glm_moe_dsa_target_verify_qmv_*` | mlx-vlm | Speculative verify. Optional. |
| `nemotron_h_target_{verify,replay}_mamba_*` (2) | mlx-vlm | Nemotron-H verify. `_mamba_update_timewise` fallback. |
| quantized_verifier family (13: fused/moe/qmv/qargmax/token-tiled/streamed/nvfp4) | mlx-vlm `quantized_verifier.py` | Speculative decode of quantized models. Native-quantized fallback. Note: uses `simdgroup_matrix` + `atomic<>` in source; hand-port analogue is `VK_EXT_cooperative_matrix`, which the overlay already uses (`matmul_coopmat`). |
| `qwen3_5_ragged_sdpa_{1p,2p1,2p2}` (3) | mlx-vlm | Qwen3.5 prefill. Per-pad-group SDPA fallback. |
| `qwen4_exp_qsa_sparse_attention_*` | mlx-vlm | Sparse prefill. Dense fallback. |
| `minimax_m3_sparse_prefill_1p_*` | mlx-vlm | Sparse prefill. Dense fallback. |
| `indexed_sparse_attention_*` | mlx-vlm | Sparse attention. Fallback. |
| `indexer_epilogue_h*` | mlx-vlm longcat | Indexer epilogue. Gated. |
| `mlx_vlm_affine_1bit_qmv/qmm_*` (2) | mlx-vlm `quantization/one_bit.py` | 1-bit inference. `dequantize_one_bit` + matmul fallback. |
| turboquant family (27: score/pack/quantize/decode/2-pass/split) | mlx-vlm `turboquant.py` | Opt-in KV quant. Every maker returns None without Metal; `@mx.compile` fallbacks exist. |
| `nemotron_small_row_{gemv,swiglu}` (2) | mlx-vlm nemotron_labs_diffusion | Gated on `_HAS_METAL`. Small GEMV/SwiGLU. |

### (b) Bounded hand-port set: hard-fails today, ports are small, real users blocked

| # | Kernel | Repo / file | Computes | Effort | Why it matters |
|---|---|---|---|---|---|
| 1 | `bitlinear_matmul` | mlx-lm + mlx-vlm `bitlinear*.py` (same source twice) | Ternary 2-bit packed GEMV + scale; only MSL feature is `simd_sum` | S | **BitNet models cannot run at all** — no gate, no fallback. |
| 2 | `flux2_fused_double_norm_rope_*` | mlx-vlm bonsai `klein_fast/blocks.py` | Fused RMSNorm + RoPE for double-stream Flux2 blocks | M | Bonsai image gen has no slow path; both norm and rope exist as primitives/shaders already (`fast_norm`, `fast_rope`). |
| 3 | `flux2_fused_single_norm_rope_*` | same | Same, single-stream | M | Same. |
| 4 | `inkling_banded_mask` | mlx-vlm `inkling/language.py` | Banded relative-position mask (index math) | S | Gate checks `default_device()==gpu` only; on omarchy it takes the kernel path and raises. Upstream one-line fix (`and mx.metal.is_available()`) routes to the existing `rel@proj` fallback. |
| 5 | `inkling_banded_mask_v2` | same | Same, v2 layout | S | Same gate fix. |
| 6 | `inkling_sconv_decode` | same | K-1 state short convolution decode | S | Same gate fix; conv fallback exists below. |
| 7 | `inkling_moe_route` | same | Top-k routing + weights | S | Same gate fix; argpartition fallback exists. |
| 8 | `inkling_moe_down_combine` | same | 4-bit dequant rows + weighted combine; gated on `bits==4, group==64, input_dims==2048` | M | Same gate fix for correctness; the overlay's `dequant`+`qmm_vec` shaders cover the math for a fast port. |
| 9 | `mlx_vlm_llguidance_mask` | mlx-vlm `structured.py` | Bit-test mask over logits (`as_type<uint>`, `-inf`) | XS | Guided/structured generation hard-fails; `mx.where` + bit ops cover it exactly. |
| 10 | `custom_depthwise_conv1d` | mlx-audio mossformer2 | Stride-1 symmetric-pad depthwise conv1d | XS | No Metal gate; `mx.conv1d` fallback one branch away. Trivial port or gate fix. |
| 11 | `qk_relu_squared` | mlx-audio mossformer2 `flash_attention_kernels.py` | Grouped Q@K^T + ReLU2 (uses `q_shape`/`k_shape` attrs) | S | Used in the fused attention path; matmul+maximum+square cover it. |
| 12 | `mlx_audio_phonon_unpack_base5_v1` | mlx-audio `stt/models/phonon/packed.py` | Load-time base-5 → 2-bit-plane unpack (bit twiddling, no float math) | XS | Phonon-1 decoder cannot load; a numpy port runs once at load, no GPU kernel needed. |
| 13 | CBQ family: `cbq_gather_mm{,_v2,_v3,_v4,_v3_situ,_v4_situ,_glu}` + `kda_glue_{pre,post}` + `moe_route_fused` + `situ_fused` + `cbq_grad_{d,x}` (15, incl. 2 training-only `atomic_outputs` kernels) | HF `avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw` (`k3_cbq.py`, `k3_fuse.py`) — the **only** HF repo of 822 shipping custom kernels | Custom 1.5625bpw codebook gather-MM for a third-party K3 quant; `dequant_ref` (pure-MLX vectorized) exists as parity reference | L (as one group) | Third-party quant (3.2k downloads); no runtime fallback. Self-contained: hand-port against `dequant_ref`, not a compiler. |

### (c) Genuinely needs arbitrary MSL compilation: 1 kernel

`nemotron_bm32_steel_linear_nt` (mlx-vlm `nemotron_labs_diffusion/language.py`)
builds its `header=` by inlining `mlx/backend/metal/kernels/steel/gemm/gemm.h`
— Apple's internal Metal headers, `using namespace mlx::steel`, simdgroup
matrix GEMM throughout. It is gated (`_HAS_METAL`, native fallback), so nobody
is broken; but a hand port means reimplementing steel GEMM, which is a project,
not a kernel. This is the shape of the long tail a translator would serve.

## MSL-to-SPIR-V survey (honest, with links)

- **Metal Shader Converter (Apple) is the wrong direction.** It converts
  DXIL → Metal IR/metallib for porting Windows games to Apple platforms
  (https://developer.apple.com/metal/shader-converter/). It takes no SPIR-V
  input and emits nothing usable on Vulkan. Not applicable.
- **SPIRV-Cross is SPIR-V → MSL only.** Khronos's tool reflects and
  cross-compiles SPIR-V *out* to MSL/GLSL/HLSL; there is no MSL frontend and
  none planned (https://github.com/KhronosGroup/SPIRV-Cross,
  issue #650). Not applicable.
- **Naga has no MSL frontend.** Gfx-rs Naga parses WGSL, GLSL, and SPIR-V and
  *emits* MSL; it cannot parse MSL (https://github.com/gfx-rs/naga). Not
  applicable.
- **No open MSL frontend exists.** MSL is a C++ dialect whose only compilers
  are Apple's; parsing arbitrary user kernels means a Clang-scale C++
  frontend, and the ecosystem demand runs the other way (write HLSL/GLSL →
  SPIR-V → MSL). No project was found that parses MSL to SPIR-V.
- **Apple's `metal` toolchain cannot run on Linux.** The compiler ships with
  Xcode/macOS SDKs, depends on the macOS runtime/ABI, and the Xcode and Apple
  SDKs Agreement restricts the tools to Apple-branded hardware. Even where it
  physically runs (Darling experiments), the license forbids it. The
  constraint file already records: never commit Apple SDK contents anywhere.

## Implementation decision

Use the bounded runtime translator for MLX custom kernels. Keep its refusal
boundary explicit rather than importing a general C++ or MSL frontend. Port a
kernel only when it falls outside that boundary and has a maintained reference
implementation.

## Vulkan translation status (final GPU recheck, 2026-10-08)

The bounded MSL subset (see "Current implementation" above) was rechecked
against all 26 kernels of section (b) on the Vulkan GPU. Each kernel ran in
its own process against an fp64 or integer-exact NumPy reference, with a
fresh translation cache per run and NaN/inf/checksum validation of every
output. The table shows one row per kernel from the final full run on an
M2 Max (G14C), wheel `0.32.4.dev202610081112+f81a560`; the harness under
`tools/kernel_recheck` was byte-identical to main `987988a` (md5-verified).
Float kernels allow at most 2 bf16 ulp at magnitude; integer and bit kernels
compare exactly (docs/numerics-gate.md rule 1).

| Kernel | Status | Detail | Wheel | Chip | `uname -r` | Date |
|---|---|---|---|---|---|---|
| `bitlinear_matmul` | PASS | maxdiff 0.000477148 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `fused_double_norm_rope` | PASS | maxdiff 0.00789396 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `fused_single_norm_rope` | WRONG | 288816/294912 outputs bad, maxdiff 164691 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `inkling_banded_mask` | PASS | maxdiff 0.00374421 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `inkling_banded_mask_v2` | COMPILE-FAIL | glslc: GL_NV_explicit_typecast, nested C-casts survive | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `inkling_sconv_decode` | COMPILE-FAIL | glslc: GL_NV_explicit_typecast, nested C-casts survive | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `inkling_moe_route` | PASS | maxdiff 0.00214481 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `inkling_moe_down_combine` | COMPILE-FAIL | glslc: GL_NV_explicit_typecast, nested C-casts survive | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `mlx_vlm_llguidance_mask` | PASS | maxdiff 1e+308 (sentinel, equal ±inf values) | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `custom_depthwise_conv1d` | PASS | maxdiff 7.13119e-05 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `qk_relu_squared` | PASS | maxdiff 1e+308 (sentinel, equal ±inf values) | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `mlx_audio_phonon_unpack_base5_v1` | PASS | exact, maxdiff 0 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_gather_mm` | REFUSED | `device pointer arithmetic` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_gather_mm_v2` | REFUSED | `device pointer arithmetic` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_gather_mm_v3` | REFUSED | `as_type<char4>` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_gather_mm_v4` | REFUSED | `as_type<bfloat4>` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_gather_mm_v3_situ` | REFUSED | `as_type<char4>` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_gather_mm_v4_situ` | REFUSED | `as_type<bfloat4>` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_gather_mm_glu` | REFUSED | `as_type<char4>` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_grad_d` | REFUSED | `device pointer arithmetic` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `cbq_grad_x` | REFUSED | `device pointer arithmetic` | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `kda_glue_pre` | COMPILE-FAIL | glslc: GL_NV_explicit_typecast, nested C-casts survive | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `kda_glue_post` | COMPILE-FAIL | glslc: GL_NV_explicit_typecast, nested C-casts survive | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `moe_route_fused` | PASS | exact, maxdiff 0 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `situ_fused` | PASS | exact, maxdiff 0 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |
| `situ_pair_fused` | PASS | exact, maxdiff 0 | `f81a560` | M2 Max (G14C) | `7.1.12-2-12.3-sep-ARCH` | 2026-10-08 |

**Counts: 11 PASS / 1 WRONG / 5 COMPILE-FAIL / 9 REFUSED = 26**

`maxdiff` is the runner's worst relative error against the reference. `0` is
an exact integer or bit match. `1e+308` is the runner's sentinel for equal
±inf values, which make the difference undefined. The five COMPILE-FAIL
kernels need the cast scanner rewritten with a real parser, or MSL
pre-normalization before the scanner; three regex-based rewrites each failed
differently, and the failed forms are recorded in
`receipts/2026-10-08-kernel-recheck/README.md`. No kernel in the final run
produced a silent wrong value inside tolerance: a wrong value fails the run
(exit 3). Translated GLSL is cached under `~/.cache/mlx-omarchy/spirv`,
keyed by the kernel source and the translator hash; the runner uses a fresh
cache directory per run.
