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

## Re-running

All commands run from this directory with the patched wheel installed, the cooperative-matrix ICD selected (`VK_DRIVER_FILES`), and `MESA_SHADER_CACHE_DISABLE=true MLX_DISABLE_COMPILE=1`. The twin is the G16S default; `MLX_OMARCHY_SDPA_FLASH_QCHUNK=0` selects the 16-panel kernel. (The rows above were taken before the switch got its final name: the twin arm was selected with a pre-merge variable on the same build, and the landed arm was the default. Same two kernels.)

Isolated flash rows (32 Q heads, 8 KV heads, hd 128, warmup + 7 reps; prints `ATTR <tag> L=... med_ms=... range=...`):

```
MLX_OMARCHY_SDPA_FLASH_QCHUNK=0 FLASH_KILL_TAG=landed python -I flash_time.py
FLASH_KILL_TAG=qchunk8 python -I flash_time.py
```

`FLASH_KILL_LS` (default `2048,8192,16384`), `FLASH_KILL_REPS` (default 7), `FLASH_KILL_HQ` / `FLASH_KILL_HKV` change the shape.

Bit-identity dumps (prints `IDENT` for run-to-run identity and `CMP ... max_abs=... rel_l2=...` against the dump):

```
MLX_OMARCHY_SDPA_FLASH_QCHUNK=0 FLASH_GQA_TAG=landed python -I flash_numeric.py --ls 128,256,512,1024,2048 --dump landed.npz
FLASH_GQA_TAG=qchunk8 python -I flash_numeric.py --ls 128,256,512,1024,2048 --check landed.npz
```

End-to-end rows (`e2e.txt`), one arm per process, arms interleaved:

```
MLX_OMARCHY_SDPA_FLASH_QCHUNK=0 python -I run_prefill.py --model mlx-community/Qwen3-8B-4bit --targets 2048,4000,8000 --max-tokens 32 --tag landed1
python -I run_prefill.py --model mlx-community/Qwen3-8B-4bit --targets 2048,4000,8000 --max-tokens 32 --tag qchunk8_1
```

(session 2 used `--targets 8000,16000`.)

## Parity test

`overlay/tests/omarchy/test_sdpa_prefill_flash.cpp`, "causal flash Q_CHUNK=8 twin matches the 16-panel kernel exactly": on G16S it runs the default (twin) and `MLX_OMARCHY_SDPA_FLASH_QCHUNK=0` (16-panel) on the same inputs, checks which kernel each run dispatched, and requires the bf16 outputs to be byte-identical. 32 Q / 8 KV heads, hd 128, causal:

| q_len | k_len | what it covers | kernels (default / `=0`) | differing elements | max_abs |
| --- | --- | --- | --- | --- | --- |
| 2065 | 2072 | 2048+17 rows: last q tile clamped (`q_len % 32 = 17`), key offset 7 | twin / 16-panel | 0 of 8,458,240 | 0 |
| 8192 | 8192 | long context, square | twin / 16-panel | 0 of 33,554,432 | 0 |
| 1024 | 3072 | `k_len > q_len`, key offset 2048 | twin / 16-panel | 0 of 4,194,304 | 0 |
| 2056 | 2056 | `q_len % 32 = 8`, square | twin / 16-panel | 0 of 8,421,376 | 0 |

Run on the G16S (same ICD as above): 1 test case, 20 assertions, all passed. Raw output: `parity.txt`.

The route needs `k_len % 8 == 0`, so the 2048+17 row case pairs `q_len` 2065 with `k_len` 2072; it covers the clamped last q tile and a nonzero key offset at once.

```
omarchy_sdpa_prefill_flash_tests -tc='causal flash Q_CHUNK=8 twin*' -s
```

## Compile coverage

The CI syntax gate configures with Ubuntu's glslang, which lacks `GL_EXT_bfloat16`, so `OMARCHY_BF16_DIRECT` is off there: neither causal flash shader (16-panel or `-DQ_CHUNK=8`) nor the dispatch block under `MLX_OMARCHY_BF16_DIRECT` is compiled by CI. The only compile evidence for the twin is the G16S build used for these runs and for the parity test.
