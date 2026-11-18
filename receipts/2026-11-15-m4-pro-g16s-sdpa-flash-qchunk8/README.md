# M4 Pro (G16S): default Q_CHUNK=8 causal coopmat flash SDPA prefill

Date 2026-11-15. Device: Apple M4 Pro (G16S B1). ICD: aurora-wip + mesa #353 + #343 + #350 + #355 (Mesa 26.3.0-devel `git-f1a664ebd6`). `python -I`, `MESA_SHADER_CACHE_DISABLE=true`. Model `mlx-community/Qwen3-8B-4bit`. 32 greedy decode tokens after each prefill.

## Change

The landed causal flash kernel keeps all 16 eight-wide Q panels in registers for the whole key walk. The `Q_CHUNK=8` twin loads eight panels per D-pass and reuses them across the four key-blocks of a 32-key tile. The matmad add order is the same as `s = 0 .. DT-1`, so isolated dumps are bit-identical to the 16-panel kernel.

On G16S this twin is the new default when the existing causal-flash route is selected. `MLX_OMARCHY_SDPA_FLASH_QCHUNK=0` restores the 16-panel kernel. Other parts keep the 16-panel kernel.

## Isolated flash (32 Q heads, 8 KV heads, hd 128, warmup + 7 reps)

| L | landed median ms | Q_CHUNK=8 median ms | change |
| --- | --- | --- | --- |
| 2048 | 13.573 | 12.485 | −8.0% |
| 8192 | 223.672 | 204.429 | −8.6%, ranges disjoint |
| 16384 | 934.079 | 841.126 | −10.0%, ranges disjoint |

Bit-identical at L=128, 256, 512, 1024 and 2048 (`max_abs=0`, `rel_l2=0`).

## End-to-end (5 interleaved runs)

Token sha identical in every run: `1dd5a67f8afc` (pf2060 and pf4006), `cdb4ee2aea69` (pf8010), `c07a9aa35af0` (pf16004). Decode was flat.

Session 1 (pf2060 / pf4006 / pf8010):

| length | landed median tok/s (range) | Q_CHUNK=8 median tok/s (range) | change |
| --- | --- | --- | --- |
| pf2060 | 361.75 (359.59–362.39) | 362.45 (358.02–362.91) | +0.19%, overlap |
| pf4006 | 316.07 (315.66–317.94) | 316.41 (308.10–316.71) | +0.11%, overlap |
| pf8010 | 265.63 (264.56–265.95) | 268.28 (262.54–268.79) | +1.00%, overlap (qchunk8 run 1 = 262.54) |

Session 2 (pf8010 / pf16004):

| length | landed median tok/s (range) | Q_CHUNK=8 median tok/s (range) | change |
| --- | --- | --- | --- |
| pf8010 | 265.16 (254.80–265.90) | 268.47 (267.95–268.95) | **+1.25%, ranges disjoint** |
| pf16004 | 206.42 (206.17–206.69) | 212.90 (212.19–213.25) | **+3.14%, ranges disjoint** |

Raw rows: `e2e.txt`.

Short context is flat, not a regression. The disjoint win shows at 8k–16k, where flash is a larger share of prefill.
