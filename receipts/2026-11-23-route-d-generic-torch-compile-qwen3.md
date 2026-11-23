# Route D milestone 4a: generic torch.compile(backend="mlx_vulkan") at 23.7 tok/s

Date: 2026-11-23. Host: jwm1 (M1 13-inch, Asahi Linux, kernel 7.3.0-2-20261110.1-lab-ARCH).
Wheel: omarchy-mlx 0.7.50rc2-1 (mlx 0.32.4.dev202610292155+42b30a2d), torch 2.15.0+cpu, transformers 5.22.1.

## Result

| path | decode tok/s | greedy ids vs CPU |
|---|---|---|
| torch CPU eager fp32 (DynamicCache, sdpa) | 3.3-3.5 | ref |
| torch.compile mlx_vulkan, M2 per-node | ~1.76 | 32/32 |
| torch.compile mlx_vulkan, compiled-graph | **23.72** (42.2 ms/tok; isolated steady-state 46.15 ms/tok) | **32/32** |
| hand-lowered qwen3_mlx.py (M3) | 22.9 | 32/32 |
| native mlx-lm (M3 baseline) | 38.24 | — |

Qwen3-0.6B, prompt "The capital of France is", 32 greedy tokens. Ref/MLX texts identical:
" Paris. The capital of Italy is Rome. ..." Full id list in the run log.
Generic path now ~1.61x of native mlx-lm, faster than the M3 hand-lowered path.

## What changed in tools/torch-mlx-vulkan/mlx_vulkan.py

The M2 executor replayed per-node Python dispatch each call (~570 ms/tok). New
`make_compiled_executor` lowers the whole dynamo FX graph into one lazy mx
computation wrapped in `mx.compile`, called through a guard that falls back to
the eager executor on failure.

- **Static KV cache**: `transformers.cache_utils.StaticCache` + `attn_implementation="sdpa"`.
  Cache mutations (`index_copy_` x2/layer, `add_` on cumulative_length, 84 state
  placeholders total) are modeled functionally: `index_copy_` ->
  `where(inrange, take(src, arange-pos0), K)`; env-mutation makes post-write
  reads see updated values; updated K/V/cumulative become extra graph outputs
  fed back as inputs next call. HF's buffers only seed state once.
- **RMSNorm fusion**: `to->pow->mean->add(eps)->rsqrt->mul->to->mul` chains ->
  `mx.fast.rms_norm`; 113 chains fused (28 block norms x2 + 28 qk norms x2 + 1).
- **SDPA**: the single FX `scaled_dot_product_attention` node maps to
  `mx.fast.scaled_dot_product_attention` with the graph's additive mask.
  HF's GQA `expand+reshape` of KV to q-heads is bypassed (pre-expand 5-D tensor
  squeezed); mx sdpa broadcasts kv heads natively.
- **dtype policy**: bf16 at GEMM/sdpa boundaries and cache storage, fp32
  residual/elementwise spine -> greedy ids identical to fp32 CPU.
- Driver `gen_qwen3.py`: eager prefill populates the StaticCache, then the
  compiled decode loop. First compiled call (dynamo trace + mx.compile) is a
  warmup step outside the timed window.

## Gotchas found

- dynamo emits `operator.getitem` as `<built-in function getitem>`; name-keyed
  lookups need `__name__` normalization everywhere (dispatch + fusion pass).
- torch 2.15 retrace bug: if compiled `step` returns a different rank (HF emits
  `logits` [B,V] not [B,T,V] under compile) and the caller feeds rank-shifted
  ids back, the retrace hits a fake-tensor broadcast failure inside
  `apply_rotary_pos_emb`. Fix: normalize rank at the call site, keep
  `dynamic=False`.
- The fused matcher must check `n in fused` BEFORE resolving node args, and
  skip only chain members not shared with other chains' inputs.
- `MLX_OMARCHY_TRACE_DISPATCH` logs nothing inside `mx.compile`d calls; dispatch
  accounting for this path is node-level, not kernel-level.

## Caveats

- Only the decode step is compiled; prefill runs eager (one-time cost, also
  avoids a second mx.compile signature for T=S scatter writes - the take+where
  lowering handles it but it is untested at T>1).
- QKV fusion (3 linears -> 1 GEMM) and RoPE fusion are future levers; mx.compile
  already fuses the elementwise rope chain.
- jwm1 is shared; CPU baseline numbers vary with contention. The 46.15 ms/tok
  steady-state figure is the controlled measurement (flock on /tmp/m1-gpu.lock
  unavailable; run was uncontended).
