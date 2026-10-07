# MoE gather_qmm decode: subgroup kernel + z-chunked dispatch (2026-10-06/07)

Landed on main: `gather_qmm_sub.comp` (one workgroup per output, 256 lanes, subgroupAdd reduction)
and the z-chunked dispatch (x = min(count, 65535), z = ceil(count / 65535)).
Selector: bf16, 4-bit, group 64, affine, transposed weights, m == 1. Kill switch:
`MLX_OMARCHY_GATHER_QMM_SUB=0` selects the scalar kernel.

## Why

Routed-MoE decode spent 71-80% of GPU time in `GatherQmmBF16`: one thread per output, serial K loop,
33-48 workgroups per dispatch (DeepSeek-Coder-V2-Lite, M2, 4.17 ms per dispatch, 78 dispatches
per token). On GLM-4.5-Air the old Sub selector also fell back to the scalar kernel as soon as
`batch x topk x N > 65535` (down_proj: B=2 gives 65,536), so batch decode scaled negatively.

## Measurements

Host: jw16 (Apple M1 Max, G13C), packaged Honeykrisp e7631595df, cooperative matrix on.
Script: w7G's `w7g-batchstep.py` (last 4 layers, median of 12 steps, ms per layer).

GLM-4.5-Air-4bit (glm4_moe, 128 experts, top-8, moe_intermediate 1408):

| arm | B=1 | B=2 | B=4 |
|---|---:|---:|---:|
| scalar kernel (control) | 82.07 | 146.63 | 268.40 |
| Sub, count-gated (earlier) | 22.75 | 60.6 | 111.2 |
| Sub, z-chunked (landed) | 9.71 | 18.02 | 28.17 |
| speedup vs scalar | 8.5x | 8.1x | 9.5x |

Aggregate tokens per second versus serial: 1.08x at B=2, 1.38x at B=4 (the count-gated kernel gave 0.75-0.85x).
Dispatch census at B=2: `GatherQmmSubBF16` x24 (4 layers x 3 calls x 2 chunks), zero scalar dispatches.
w7G's pre-Sub baseline was 81.00 ms per layer at B=1.

DeepSeek-Coder-V2-Lite decode, 16-token greedy: 70.9 ms per token with Sub, 449.6 with the scalar
kernel (6.3x); both outputs are coherent text.

## Numerics

Per `docs/numerics-gate.md` (independent reference). `omarchy_matmul_family_tests`, case
"gather qmm subgroup kernel matches scalar at decode shapes", per-case relative L2 against the fp64
host reference (scalar | Sub):

| k | experts | index_count | n | scalar | Sub |
|---:|---:|---:|---:|---:|---:|
| 128 | 3 | 2 | 64 | 0.00272 | 0.00272 |
| 192 | 3 | 2 | 64 | 0.00249 | 0.00249 |
| 128 | 3 | 8 | 1408 | 0.00251 | 0.00251 |
| 2048 | 3 | 8 | 1408 | 0.00251 | 0.00251 |
| 128 | 48 | 48 | 1408 | 0.00245 | 0.00245 |

The 67,584-workgroup z-chunk case: tail relative L2 0.00240.

DeepSeek-Lite against a Metal reference on the same quantized weights, 527-token teacher-forced text:
scalar 96.02% top-1 agreement (PPL +0.245%), Sub 97.15% (+0.416%).

Decode logits differ between the count-gated and z-chunked builds by 1-4 bf16 ULP from the first
decode step. The prefill logits are bit-identical and all 16 greedy token ids match. No argmax
flips (minimum top-2 gap 0.25). Cause: the recompiled shader changed accumulation order.

## Suites (jw16, tree = main 436633335 + the four commits; head cc45eab4c rebuilt afterwards)

matmul_family 27/27 (82,942,518 assertions), runtime 49/49, take_fill 9/9, take_bool 5/5,
fast_ops 43/43 (26 assertions inside the known `may_fail` SDPA-VJP case), kv_ops 16/16,
capability_sim 6/6. Earlier stack (ade1656c3) was green on jwm1 (G13G): matmul_family 26/26,
runtime 49/49.

## Test fixture history (so nobody repeats it)

From b2 to b10 the new test failed with relative L2 0.91-1.40 on both arms. Cause: the fixture
passed an all-zero `lhs_indices` array while the reference contracted each batch's own x row.
Neither kernel was wrong. An earlier claim of a broken pack loop in `host_affine_quantize` was
checked against the source and is false.

## Not measured / open

- jwm1 (G13G) and M2 (G14C) runs of this exact stack: the M2 and jw16 are in the GLM cluster window.
- Models with f16 scales, group sizes 32/128, 8-bit, or mxfp4/nvfp4 stay on the scalar kernel.
- Prefill (m > 1): Laguna prefill hangs between 482 and 2016 tokens and runs at ~7 tok/s; the
  evidence chain (diag wheel profile) is queued.
- The ~390 staging copies per token (~20 ms per token on DeepSeek-Lite) are next; the design is on
  `agent/MoeLayer2` and waits for the allocator fence fix (landed as 2f0dbb2a7).
