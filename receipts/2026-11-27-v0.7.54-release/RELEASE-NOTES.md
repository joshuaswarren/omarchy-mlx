# omarchy-mlx v0.7.54

Measured decode defaults for the Apple M1 Max and Apple M1 that beat or
match macOS, the G14C (Apple M2 Max) GDN/Mamba decode stack, a
one-dispatch online-softmax decode SDPA, cooperative-matrix and flash
prefill kernels for discrete GPUs, 39 new `aten` lowerings in
torch-mlx-vulkan, and a serving/assistant hardening batch. 298 non-merge commits
since v0.7.53.

## Decode and kernel speedups

- Apple M1 Max (G13C): measured Qwen3.5-9B decode defaults enabled,
  +12.1% to +13.5% over the previous defaults and ahead of the matched
  macOS baseline (#1117). The stack behind it: SwiGLU upfront
  scale/bias plus ResNormVec mid-loop scale/bias (+1.35% on the 4B
  model, #1077), a zero-shared-memory single-barrier uvec4 ResNormVec
  (+2.29% on shipped Mesa, #1107), the 3-weight FIFO-upfront QKV Multi
  variant (+0.90% m15 / +0.80% shipped Mesa, #1111), fast-tile-only
  R2S8W2 Q4 decode variants (+1.42%, #1063), a 128-bit uvec4
  coherent-L2 ResNorm epilogue (+1.14% 4B, #1049), and batch_work
  80000 with the chunked vocab tail (+2.30% 4B, #1015).
- Apple M1 (G13G): measured Wave 4/5 decode defaults enabled (+2.16%
  2B ctx2048, +1.16% 4B, #1104; +8.2% 2B decode, #1067).
- Apple M2 Max (G14C): the GDN decode stack — coalesced-state GDN
  conv+norm and conv+delta+norm Sg32 kernels (+11.34% 2B, #1038;
  +5.12% 9B, #1039), GdnConvDeltaUpdate f32 A_log with a GQA 3-in-1
  Sg32 fusion (+3.9% 9B to 60.4 tok/s, #1043), folded in_proj_a/b
  (#1053), a low-register SwiGLU R1S8W2 with a 512-thread coalesced
  uvec4 ResNorm (+5% 2B, +3% 9B, #1040), a balanced-pair SwiGLU with a
  3-dispatch greedy word head (+4.72% 2B, #1024), restored upfront
  8-load Q4_ROWS (+3.02% 2B, +5.03% 9B, #1062), fast-tile-only
  upfront Q4 variants (+1.50% 2B, +0.60% 9B, #1068), ResNormVec4
  Pack32/hoist variants (#1071, #1081, #1101), the G14C parity bundle
  (GDN pow2 trim, 3-member Multi, Small-N Word, SDPA HD256 fast,
  prologue norm, #1080), and `AGX_HOIST_LOADS=1` defaulted on the M2
  Max (+4.12% 2B, #1066). Batch-1 word/lm_head reaches 215.03 tok/s
  (#1101).
- One-dispatch online-softmax BF16 decode SDPA: on G13C for k<=2304
  (#997), on G14C and HD=256 (#1016), with fused Q/K RMSNorm + RoPE +
  k_cache write on G13C (#1013), a fused output-gate epilogue and
  HD=256 partial-RoPE online decode (#1036), a parallel subgroupAdd
  epilogue with the low-register Q4_ROWS fast tile (+2.32%, #1050),
  and the G13G default for k<=2304 (#1012).
- Mamba-2 `ssm_kernel` decode step ported to Vulkan; mlx-lm
  `ssm_update` routes through it (#1035).
- A `<=128`-thread fused rope-norm variant for D<=128 rows (#958).

## Other GPU backends

- Intel ANV Xe2 cooperative-matrix BF16 dense matmul and QMM prefill
  (#963).
- 16x16x16 BF16 cooperative-matrix QMM prefill and causal flash
  prefill kernels for RADV-class GPUs (#880, #883).
- Device-local weights default on discrete GPUs, with a patch
  hunk-count fix (#904).
- RTX 5060 Ti compiled-tape A/B measured a null: batch_work 40000 owns
  the gain (#1109).

## torch-mlx-vulkan and omarchy_torch

- 39 new `aten` lowerings: mode, kthvalue, cosine_similarity,
  softplus, mish, xlogy, special_xlog1py, expm1, logaddexp,
  searchsorted, bucketize, fft_fft, fft_ifft, grouped conv1d,
  native_layer_norm_backward in the wheel backend, the piecewise
  activation family, relu/relu_/gelu-tanh boolean dispatch, and
  slice.Tensor optional-argument defaults (#812, #1073-#1102 series).
- Parity tests: aten.mv + addmv (#1073), where.self + masked_fill
  (#873), comparison .default dispatch-miss fix (#1078), PAR-41 census
  fixes for squeeze_, conv1d unbatched, LN-backward broadcast and
  clamp bounds (#1098), native_batch_norm_backward contract and
  accumulation dtype (#993).
- Token-parity pins per shape across the kernel toggle (#999).

## Serving, assistant and storage hardening

- serve: bounded reservations, snapshot-index, config.json and
  mlx-lm stamp/RECORD/metadata reads; symlink and FIFO refusals that
  stop target truncation and hangs; uclamp_min range refusal; clean
  refusal of an inaccessible local target path (#770, #777, #981,
  #1010, #1031, #1047, #1103, #1110).
- assistant: symlinked/FIFO model-path refusal, capped VAD and
  wake-word downloads, hardened managed log writes with bounded tail
  reads and spawn cleanup, atomic runtime-file publish, and a bounded
  transfer export (#1041, #1042, #1060, #1061).
- bonsai2: bounded config.json and safetensors header reads, a
  --managed registry home that releases only what it reserved, and
  refusal of packs whose files change between admission and load
  (#1029, #1054, #1106).
- laya and pairs: atomic safetensors writes, hashed-snapshot copies,
  bounded manifest and encoder-config reads, and fail-closed
  --managed and lock handling (#1005, #1007, #1009, #1032, #1034,
  #1037, #1044, #1048).
- synthesis: refuses FIFO, oversized or deeply nested state files
  (#1057).
- coreml: a <=128-thread 1-row device-chained TDT path for Apple M6
  (G18G) and the s_pj OOB/carrier sizing fix (#1022, #1033).
- mlx-lm: the MLX_API_KEY default import fix for the 0.31 series
  (#1025).

## Packaging

- `packaging/qualified-mesa-commit` moves to `53ad9af196f5`
  (mesa-1 #15-#20) (#1113).

## Cut

Cut commit and assets recorded in the release receipt at tag time.

## Note on Apple M1 Max PMP memory DVFS

The 16-inch M1 Max parity rows referenced for this release ran with the
Apple PMP (`apple_pmp`) memory DVFS driver active, which the shipped
Linux kernel does not yet turn on for M1 Max (`t6001`).
[`aurora-silicon/linux#3401`](https://github.com/aurora-silicon/linux/pull/3401)
turns it on. Matched decode measurements on the same 16-inch M1 Max
with Mesa `53ad9af196f5` (`MLX_OMARCHY_DEFER_TOKENS=0`) without PMP vs
with PMP:

- `Qwen3-4B-Instruct-2507-4bit` (`50d42775`) `ctx=512`: `98.48 tok/s`
  without PMP vs `107.87 tok/s` with PMP (`+9.5%` / `−8.7%` without).
- `Qwen3.5-9B-MLX-4bit` (`938d8919`) `ctx=512`: `58.56 tok/s` without
  PMP vs `64.09 tok/s` with PMP (`+9.4%` / `−8.6%` without).
- `Qwen3.5-9B-MLX-4bit` (`938d8919`) `ctx=2048`: `57.74 tok/s` without
  PMP vs `63.49 tok/s` steady with PMP (`+10.0%` steady / `−9.1%`
  without; second consecutive alias leg `60.44 tok/s` under fan ramp).

