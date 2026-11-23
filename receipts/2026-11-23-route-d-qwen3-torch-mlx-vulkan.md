# Route D milestone 3: fused Qwen3-0.6B decode through torch-mlx-vulkan

Date: 2026-11-23. Lane: w7Y.

## Environment

- Host: 13-inch M1 (T8103) running Omarchy Linux (placeholder host id)
- Kernel: 7.3.0-2-20261110.1-lab-ARCH
- Mesa: 1:26.2.4-1; Vulkan ICD `libvulkan_asahi.so` (API 1.4.362)
- Wheel: omarchy-mlx 0.7.50rc2-1, `mlx-omarchy 0.32.4.dev202610292155+42b30a2d`
- Provenance: `scripts/mlx_provenance.py` -> `verified: match`,
  `mlx/core.cpython-314-aarch64-linux-gnu.so` sha256 `7542a0bcff502c00…`
- torch 2.15.0+cpu, transformers 5.22.1, mlx-lm 0.31.3
- Model: `Qwen/Qwen3-0.6B`, `model.safetensors` sha256 `f47f71177f32bcd101b7573ec9171e6a…` (bf16 weights on disk)
- Source: this commit adds `tools/torch-mlx-vulkan/`

## Commands

```bash
# fused path + CPU reference, same prompt "The capital of France is" (S=5), 32 greedy tokens
python tools/torch-mlx-vulkan/gen_qwen3_fast.py --model <snapshot> --dtype bf16
# native baseline, same model and host
mlx_lm.generate --model <snapshot> --prompt "The capital of France is" --max-tokens 32 --temp 0
# dispatch trace of one decode step
MLX_OMARCHY_TRACE_DISPATCH=1 python <one-step harness>
```

`/tmp/m1-gpu.lock` held for the GPU runs.

## Correctness

Fused `Qwen3Fast` bf16 (fp32 residual stream), 32/32 token ids identical to
torch CPU fp32 eager:

```
[12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 17689,
 374, 24081, 13, 576, 6722, 315, 5616, 374, 26549, 13, 576, 6722, 315,
 6323, 374, 26194, 13, 576, 6722]
-> " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. ..."
```

The same graph at fp32 and fp16 also matches 32/32. Pure-bf16 activations
(no fp32 residual stream) diverge at generated token 5 on a near-tied
argmax — documented; the fp32-stream configuration is the default.

## Speed (decode tok/s, same host, same prompt, 32 tokens)

| path | dtype | decode tok/s |
|---|---|---|
| torch CPU eager (reference) | fp32 | 1.7–2.3 idle host; 0.28–0.38 under CPU-side load |
| `mlx_vulkan` per-node lowering (milestone 2) | fp32 | ~1.76 |
| `Qwen3Fast` fused + `mx.compile` | fp32 | 10.6 |
| `Qwen3Fast` fused + `mx.compile` | fp16 | 11.0 |
| `Qwen3Fast` fused + `mx.compile` | bf16 (+fp32 stream) | **22.9 idle; 17.8 under CPU-side load** |
| `Qwen3Fast` fused + `mx.compile` | pure bf16 | 25.0 (token-id divergence at step 5) |
| mlx-lm native (`mlx_lm.generate`) | bf16 | **38.24** generation, 84.30 prefill |

Per-step bf16 tok/s (idle run): median ~23.5, range 16.9–25.6 across the 31
decode steps; prefill 0.088–0.17 s. bf16 fused is **within 1.67x of native
mlx-lm** (22.9 / 38.24) and ~10–13x the idle CPU reference.

Note on measurement conditions: the box also runs other lanes' CPU-side jobs
(wheel builds, probes). CPU eager is far more sensitive to that contention
than the GPU path; both conditions are reported rather than dropped.

## Dispatch trace

`MLX_OMARCHY_TRACE_DISPATCH=1`, one decode step at position 5:
**1,074 kernel dispatches in 29 submits** (vs ~1,785 dispatches/step across
far more submits in the per-node-lowered milestone-2 path — rtmod evidence).
The compiled decode graph keeps GEMMs/attention as named dispatches with
elementwise chains fused by `mx.compile`.
