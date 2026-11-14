# M4 Pro (G16S): default routes and cooperative matrix, 2026-11-14

MacBook Pro 16-inch, M4 Pro (J616s, T6040), 48 GB, Linux 7.3.0 aurora, mlx-lm 0.31.3.
`models-before-after.txt` is the raw log (two rounds per arm, the table gives medians).
Prompts of 297-318 tokens (pf300) and 2016-2038 tokens (pf2k), warm, then greedy decode.

Arms:

- `before`: wheel b445c03 (previous head of this branch), system Mesa `05ddb00f5c`.
- `after-sys`: wheel fed73df (this head), system Mesa. Only the two decode defaults apply.
- `after-275`: fed73df, Mesa aurora-silicon/mesa#275 a7b8244d, no variable set: the backend sets
  `ASAHI_AGX3B_COOPERATIVE_MATRIX=1` on a T6040 when the variable is unset.
- `after-fix`: fed73df, Mesa aurora-wip 25ba19af with aurora-silicon/mesa#319, no variable set
  (that driver turns the extension on by itself).
- `fix-nocoopmat`: as `after-fix` with `MLX_OMARCHY_COOPMAT=0`.

| Model | Arm | pf300 tok/s | pf2k tok/s | decode tok/s | text sha |
|---|---|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct-4bit | before(sys,b445c03) | 648.0 | 652.4 | 133.9 (133.1-134.8) | 6fa09084770d |
| Qwen2.5-1.5B-Instruct-4bit | after-sys(fed73df) | 649.6 | 651.9 | 135.7 (133.5-137.8) | 6fa09084770d |
| Qwen2.5-1.5B-Instruct-4bit | after-275(fed73df) | 1515.0 | 1537.2 | 137.2 (137.1-137.3) | 6fa09084770d |
| Qwen2.5-1.5B-Instruct-4bit | after-fix(fed73df,55c06da6) | 1504.2 | 1541.1 | 137.6 (137.4-137.7) | 6fa09084770d |
| Qwen2.5-1.5B-Instruct-4bit | fix-nocoopmat | 649.0 | 646.1 | 135.0 (134.7-135.3) | 6fa09084770d |
| Qwen3-8B-4bit | before(sys,b445c03) | 133.4 | 93.7 | 41.7 (41.6-41.7) | 9ee73549c37f |
| Qwen3-8B-4bit | after-sys(fed73df) | 134.3 | 93.5 | 41.9 (41.8-42.0) | 9ee73549c37f |
| Qwen3-8B-4bit | after-275(fed73df) | 304.4 | 315.1 | 41.8 (41.7-41.9) | 9ee73549c37f |
| Qwen3-8B-4bit | after-fix(fed73df,55c06da6) | 304.1 | 314.8 | 41.4 (40.8-41.9) | 9ee73549c37f |
| Qwen3-8B-4bit | fix-nocoopmat | 134.7 | 93.5 | 41.8 (41.8-41.9) | 9ee73549c37f |
| Qwen3.5-4B-MLX-4bit | before(sys,b445c03) | 188.1 | 166.6 | 54.1 (52.6-55.6) | 8d04f5ffd6c2 |
| Qwen3.5-4B-MLX-4bit | after-sys(fed73df) | 188.9 | 167.3 | 64.6 (64.4-64.8) | 8d04f5ffd6c2 |
| Qwen3.5-4B-MLX-4bit | after-275(fed73df) | 491.2 | 567.5 | 64.8 (64.7-64.9) | 8d04f5ffd6c2 |
| Qwen3.5-4B-MLX-4bit | after-fix(fed73df,55c06da6) | 491.3 | 568.1 | 64.8 (64.7-64.8) | 8d04f5ffd6c2 |
| Qwen3.5-4B-MLX-4bit | fix-nocoopmat | 188.6 | 166.8 | 64.8 (64.8-64.9) | 8d04f5ffd6c2 |
| granite-4.0-1b-4bit | before(sys,b445c03) | 530.6 | 292.8 | 109.4 (109.2-109.5) | 288995c87e1f |
| granite-4.0-1b-4bit | after-sys(fed73df) | 543.1 | 292.4 | 120.5 (120.1-120.9) | 288995c87e1f |
| granite-4.0-1b-4bit | after-275(fed73df) | 1121.2 | 1137.0 | 119.9 (119.0-120.8) | 288995c87e1f |
| granite-4.0-1b-4bit | after-fix(fed73df,55c06da6) | 1165.2 | 1129.3 | 119.9 (119.3-120.5) | 288995c87e1f |
| granite-4.0-1b-4bit | fix-nocoopmat | 543.8 | 292.9 | 119.1 (119.1-119.1) | 288995c87e1f |

The generated text has the same hash in every arm of a model.

## Decode defaults on G16S

| Change | A/B (Qwen3.5-4B unless named, same tokens) | Log |
|---|---|---|
| GDN decode: native GQA kernel (`gdn_pf_gqa_default_part` gains G16S) | 54.9 -> 60.5 tok/s | `ab-gdn.txt` |
| Qmm decode geometry 2x16 (`qmm_geometry`) | 60.6 -> 64.8; granite-4.0-1b 105.1 -> 120.0; Qwen2.5-1.5B and Qwen3-8B within spread | `ab-knobs4b.txt`, `ab-geom-gran.txt`, `ab-geom-small.txt`, `ab-knobs8b.txt` |

Isolated GEMV shapes are 2-10 % slower with 2x16; the gain is end to end, so the switch is made
from the model runs. `MLX_OMARCHY_QMM_GEOM=2x4` and `MLX_OMARCHY_GDN_PF_GQA=0` restore the
previous routes. Vocab chunking, `NORM_APPLE_2560`, `GDN_CONV_DELTA`, `GDN_PF_WIDE` and token-multi
stayed within run-to-run spread and are unchanged.

## Where decode time goes

- Bandwidth roof: a coalesced 512 MiB read runs at 255-260 GB/s (95 % of the 273 GB/s
  nominal); copy (read plus write counted) 207-220 GB/s. Memory clock and DRAM are not the limit.
- Decode reads per token: Qwen3.5-4B 2.37 GB (64.8 tok/s = 154 GB/s, 60 % of the read roof),
  Qwen3-8B 4.26 GB (41.8 tok/s = 178 GB/s, 69 %).
- A standalone Q4 GEMV reaches 200-225 GB/s on the large projections and 241-245 GB/s on the
  lm_head; small N=1024 projections 115-145 GB/s.
- Fixed costs: an in-command-buffer dependent dispatch about 3-5 us, a semaphore hop between
  submissions about 21 us, a host fence join about 118 us. The rest of the gap to the roof is the
  small projections, norms and the per-layer dependency chain.

## Registers

`shader-registers.txt` (664 compute shaders, `AGX_MESA_DEBUG=shaderdb`). The decode shaders use
43-127 of the 192 16-bit registers per thread and do not spill. Eight shaders reach the 192 cap and
spill, none on the decode path; the GDN prefill coopmat batch kernel (47:31) is the only one on a
hot prefill path.

## Pipeline creation

`pipeline-sweep-mesa-275.txt` (24 refused) and `pipeline-sweep-mesa-fixes.txt` (7 refused, all bf16
cooperative-matrix shaders that need `VK_KHR_shader_bfloat16` and are never created on this GPU):
all 671 compute shaders of this build created as pipelines.

## Tests

81 units, 1138 cases on the system Mesa: all pass. With aurora-wip plus #319: 80 of 81 in one run,
the failure being `fused trio override plans the rope pair with the gemv route off` (one f16 ulp,
random inputs); that case fails 1-2 times in 20 on every Mesa build tried, the system one included,
and with either Qmm geometry, so it predates this change.
