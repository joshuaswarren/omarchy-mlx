# Fused Vulkan backward kernels for SDPA and gated delta nets (upstream #4563 + #4565)

Merged: origin/main 6e3069427 → 9395cecd5 (fast-forward).
Branch: agent/vjp-kernels-fused (3 commits: c8a99e128 kernels+eval_gpu,
bcbe2970e GQA doctest, 9395cecd5 probe script).

## What shipped

The Omarchy backend now runs the gradient of `fast.scaled_dot_product_attention`
(upstream #4563) and `fast.gated_delta_update` (upstream #4565) as fused
Vulkan kernels instead of the composed fallback, so LoRA fine-tuning of
attention and DeltaNet (Qwen3-Next / Qwen3.5 GDN class) models runs on the
GPU with fused backward kernels.

New shaders (overlay/mlx/backend/omarchy/shaders/):

- `sdpa_vjp_odo.comp` (f32/f16/bf16 blobs): delta = rowsum(o · cot_o),
  one workgroup per row, shared-memory tree reduce (device-independent;
  no subgroup ops).
- `sdpa_vjp_lse.comp`: row logsumexp of the forward's float32 score
  buffer with the causal limit; runs alongside softmax so the backward's
  P = exp(scale·S − lse) sees the same score words the forward used.
- `sdpa_vjp_ds.comp`: P and dS = P·(dP − delta)·scale per score element;
  dS lands in S's buffer, P in its own. Dead-row guard keeps
  exp(−inf − −inf) from producing NaN.
- `sdpa_vjp_reduce.comp` (f32/f16/bf16 out): GQA repeat-group sum into
  the KV-head gradient tiles.
- `gated_delta_vjp.comp` (save + backward blobs): state-recall scan with
  the fused-forward step math snapshotting every 16 tokens (h0 included),
  then the backward walk matching upstream Metal's seq_gated_delta_vjp
  geometry (32 lanes per row, 4 state columns per lane, per-segment
  register replay). dq/dk/dg/db accumulate as compare-exchange float adds
  (the ScatterFCAS twin; Honeykrisp does not expose
  VK_EXT_shader_atomic_float), one workgroup partial per 32-row block.
  dv is written directly bf16; dh is float32.

Routing: `ScaledDotProductAttention::use_fallback` sends training
(output_logsumexp) to the fused VJP when causal / no array mask / no
sinks / float dtypes / GQA valid / grid fits; everything else keeps the
composed graph. `GatedDeltaUpdateVJP::use_fallback` serves bf16
activations + scalar gates (bf16 or f32) + f32 state, Dk = Dv = 128, any
Hv that is a multiple of Hk. `MLX_OMARCHY_NO_FUSED_VJP=1` restores the
composed path everywhere.

## Evidence

All runs on the merged branch (9395cecd5). Zero CPU tensor primitive
dispatch held on every leg (the suite's counters assert it).

### M2 Max (G14C), gpu-turn tickets

| suite | result |
|---|---|
| omarchy_fast_ops_tests | 37/37 (includes the new GQA GDN VJP doctest) |
| omarchy_primitive_tests | 104/104 |
| omarchy_runtime_tests | 41/41 |
| omarchy_error_contract_tests | 3/3 |
| omarchy_indexing_ops_tests | 57/57 |
| omarchy_sdpa_decode_fused_tests | 6/6 |
| omarchy_compiled_tape_tests | 13/13 |
| omarchy_fused_chain_tests | 36/36 |

Build: cmake Ninja Release, MLX_BUILD_OMARCHY=ON, glslc 2026.3.
Wheel: mlx_omarchy-0.32.4.dev202610011956+a03989c (416 MB,
sha256 5dfdf784e5d88d95ff88725a3a7c4030fc6076adb3cedfd7eb7c47cf05ae7d83).

### M1 Max (G13C), gpuwin slice (pre-merge branch 4d659d874; the VJP
kernels and gates are identical through the rebase)

| suite | result |
|---|---|
| omarchy_fast_ops_tests | 36/36 |
| omarchy_primitive_tests | 104/104 |
| omarchy_gdn_fast_route_repeat_tests | 1/1 |
| omarchy_gdn_prefill_profile_tests | 3/3 |

### Development box (llvmpipe software Vulkan)

37/37 fast_ops including the new doctest, via the staged rehearsal ICD
prefix + MLX_OMARCHY_ALLOW_NON_APPLE=1 (development only).

## LoRA one-step before/after: named blocker

The planned measurement (Qwen3.8-2B-4Bit, batch 1, seq 512, LoRA on
attention + GDN projections, alternating MLX_OMARCHY_NO_FUSED_VJP=1 vs
unset) cannot run on this backend today, on either chip, for a reason
independent of the VJP kernels:

> [omarchy] QuantizedMatmul bf16 non-transposed tile is not implemented
> for the Omarchy Vulkan backend (dtype=bfloat16, shape=[1,512,2048]).

The identical error fires with the composed path forced
(MLX_OMARCHY_NO_FUSED_VJP=1), so it is the Q4 matmul kernel the model's
4-bit weights need, not the VJP routing. Qwen3.5-9B-4bit additionally
exceeds the M2 Max's GPU memory for one autograd step. Follow-up lane
work: the 4-bit group-64 bf16/f16 non-transposed qmm kernel path, then
the measurement (the probe script is committed at
artifacts-vjp-lora-probe.py).

## Known scope edges

- Sinks never reach the fused VJP (the forward gate routes them to the
  composed graph, matching upstream Metal's NYI).
- Array masks keep the composed graph; causal rides the fused kernels
  (prefix-query causal, kL < qL, stays composed).
- Per-channel (Dk-vector) decay keeps the GDN composed path; scalar g
  rides the fused kernels.
- The SDPA GQA-shape doctest compares per-KV-head slices against a
  reshape-free composed reference (a follow-up commit adds it).

## Notebook

Private working record:
~/.local/share/apple-silicon-lab/entries/VjpKernels/
20261001T0535Z-omp-studio-local-vjp-fused-kernels.md with raw logs and
SHA256SUMS under artifacts/VjpKernels/.
