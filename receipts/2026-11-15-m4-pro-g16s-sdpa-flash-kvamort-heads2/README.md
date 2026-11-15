# M4 Pro (G16S): env-gated Q_CHUNK=8 heads-2 K/V amortization

Date 2026-11-15. Device: Apple M4 Pro (G16S B1). ICD: aurora-wip + mesa #353 + #343 + #350 + #355 (Mesa 26.3.0-devel `git-f1a664ebd6`). `python -I`, `MESA_SHADER_CACHE_DISABLE=true`. Model `mlx-community/Qwen3-8B-4bit`. 32 greedy decode tokens after each prefill.

## Change

`MLX_OMARCHY_SDPA_FLASH_KV=heads2` selects a `Q_CHUNK=8` twin compiled with `KV_HEADS=2`. A workgroup owns two GQA query heads that share one KV head: each 32-key K/V tile is loaded once and applied to both heads. `params.lhs_gap` carries the true query-head count; grid Y is `heads/2`. The dispatch is used only when the existing causal-flash route is selected, `Q_CHUNK=8` is on, `repeats >= 2`, and both `repeats` and `heads` are even.

The default on G16S stays the `Q_CHUNK=8` kernel. Other chips are unchanged. Unset the env var (or set any other value) to keep that default.

## Isolated flash (32 Q heads, 8 KV heads, hd 128, warmup + 7 reps)

| L | Q_CHUNK=8 median ms (range) | heads2 median ms (range) | change |
| --- | --- | --- | --- |
| 2048 | 12.470 (12.135–12.973) | 11.432 (11.146–11.892) | −8.3%, ranges disjoint |
| 8192 | 201.389 (199.540–211.074) | 185.794 (183.864–191.746) | −7.7%, ranges disjoint |
| 16384 | 820.341 (817.576–855.834) | 772.053 (744.948–813.204) | −5.9%, ranges disjoint (tight: max heads2 813.204 < min Q_CHUNK=8 817.576) |

Bit-identical to `Q_CHUNK=8` at L=128, 256, 512, 1024, 2048 and at flash-valid tails L=2000 and 2056 (`max_abs=0`, `rel_l2=0`).

## End-to-end (5 interleaved runs)

Token sha identical in every run: `cdb4ee2aea69` (pf8010), `c07a9aa35af0` (pf16004). Decode was flat.

| length | Q_CHUNK=8 median tok/s (range) | heads2 median tok/s (range) | change |
| --- | --- | --- | --- |
| pf8010 | 256.31 (255.61–256.42) | 259.54 (258.29–259.67) | **+1.26%, ranges disjoint** |
| pf16004 | 204.38 (204.18–204.62) | 210.07 (209.88–210.23) | **+2.78%, ranges disjoint** |

Raw rows: `e2e.txt`.

The two lengths were measured in separate sessions. Each session interleaves the two kernels on the same prompt, so the comparison is paired. Short-context identity holds; the e2e win is at 8k–16k, where flash is a larger share of prefill.
