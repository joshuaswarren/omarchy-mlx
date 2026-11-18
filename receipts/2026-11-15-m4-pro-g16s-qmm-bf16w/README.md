# M4 Pro (G16S): two-pass bf16 4-bit prefill (dequantize to bf16, dense bf16 matmul)

Backs `MLX_OMARCHY_QMM_BF16W` and `omarchy::qmm_bf16w_default_part` / `qmm_bf16w_rows_ok` in `device.h`.

## Host

Apple M4 Pro (G16S B1), kernel 7.3.0, Mesa 26.3.0-devel Borealis with the cooperative-matrix extension (aurora-wip + Mesa #350, #353, #343, #355; the system Borealis build has no cooperative matrix on G16S, and there the route is not taken). Stock clocks. One build from this branch; the two arms differ only in `MLX_OMARCHY_QMM_BF16W` (`0` = the single-kernel route of `main`, unset = the two-pass route). Model `mlx-community/Qwen3-8B-4bit` (bf16 activations, 4-bit group 64).

## What changes

The single-kernel cooperative-matrix qmm route dequantizes each weight tile to float32 inside the matmul. The two-pass route runs the affine Dequant kernel once per call to write the weight as bf16 (an encoder temporary) and contracts with `MatmulBF16Coopmat`. Rounding the weight to the activation type before the product is what the Metal backend does (`QuantizedBlockLoader` writes `threadgroup T` tiles that `BlockMMA<T>` consumes), so the two-pass route has the Metal backend's numerics. Its output equals `matmul(x, transpose(dequantize(w)))` bit for bit (`omarchy_matmul_family_tests`).

## Row rule (`isolated/rows-sweep.txt`, `tools/sweep.py`)

Milliseconds per call, best of three runs of six calls, single-kernel against two-pass:

| N x K | 256 rows | 512 rows | 1024 rows | 2048 rows |
|---|---|---|---|---|
| 1024x4096 | 0.63 / 1.12 | 1.05 / 1.02 | 1.89 / 1.58 | 3.58 / 2.88 |
| 4096x4096 | 1.76 / 3.03 | 3.34 / 3.06 | 6.42 / 5.61 | 12.65 / 10.66 |
| 12288x4096 | 4.84 / 8.14 | 9.42 / 8.60 | 18.57 / 16.15 | 36.84 / 31.33 |
| 4096x12288 | 5.14 / 8.86 | 9.86 / 8.89 | 19.38 / 16.60 | 38.11 / 31.58 |
| 151936x4096 | 56.5 / 137.5 | 113.7 / 148.6 | 257.9 / 250.4 | 532.1 / 469.8 |

The dequantize pass is a fixed cost per call, so the route needs enough rows: from 1024 rows on any weight, from 512 rows when the weight has at most 2^26 elements.

## Prefill (`e2e/prefill.txt`, `tools/run_prefill.py`)

32 greedy tokens, interleaved arms, cool-down before each arm. Runs stopped by the 95 C guard are listed and not counted. Generated text hashes are identical in every run of both arms (`1dd5a67f8afc` at 2060 tokens, `cdb4ee2aea69` at 8010).

| prompt tokens | single-kernel tok/s, median (range, n) | two-pass tok/s, median (range, n) | change |
|---|---|---|---|
| 2060 | 338.68 (336.74-339.45, 7) | 392.10 (388.77-392.91, 8) | +15.8%, ranges disjoint |
| 8010 | 253.27 (252.85-253.56, 6) | 281.96 (280.81-282.20, 5) | +11.3%, ranges disjoint |

## Accuracy (`accuracy/`, `tools/acc_cpp.py`, `tools/acc_yardstick.py`)

wikitext-2 test, 8 windows of 2048 tokens (16376 predicted tokens), one process, the flag switched per forward.

| comparison | perplexity | argmax agreement | mean KL | max abs logit diff |
|---|---|---|---|---|
| two-pass against single-kernel | 8.7422 against 8.7437 | 98.70% | 9.64e-4 | 9.97 |
| causal flash prefill off against on (accepted G16S default), both single-kernel | 8.7408 against 8.7437 | 98.78% | 9.35e-4 | 23.66 |

The two-pass route moves the logits by the same amount as the accepted flash prefill default and does not raise perplexity.

## Tests (`tests/`)

Every test binary of this branch on the same driver, one unit at a time (the flash prefill suite one case at a time, the capability simulator once per profile): 79 units, 75 pass, among them `omarchy_matmul_family_tests` 48 of 48 with the two-pass cases (`tests/battery.txt`). The four failing units were rerun with `MLX_OMARCHY_QMM_BF16W=0` and with the default (`tests/recheck-failures-route-off-on.txt`):

- `omarchy_fused_chain_tests` "fused trio override plans the rope pair with the gemv route off": a one-ulp mismatch on time-seeded random input; passes in both reruns.
- `omarchy_runtime_tests` "independent streams measure against the serialized sum", `omarchy_sdpa_prefill_flash_tests` "big causal bf16 keeps the named storage-binding refusal" and "chunked composed causal prefill is bit-identical to the whole pass": fail the same way with the route off, so they do not come from this change.
