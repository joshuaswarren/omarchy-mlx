# jw16 2026-09-29: mlx-lm last-logits prefill — quantized head on the final prompt position only

Change: `patches/mlx-lm-last-logits.patch` (vendored mlx-lm 0.31.3 series, applied last by
`scripts/apply-mlx-lm-patches.sh`; `install.sh` derives its download list from that script).
`TextModel.__call__` in `mlx_lm/models/qwen3_5.py` slices the post-norm hidden states to
`[:, -1:, :]` when a cache is passed (incremental inference). Whole-sequence calls
(`cache=None`: scoring/training) and `MLX_OMARCHY_FULL_LOGITS=1` keep full logits. Decode
steps never route through this call (the greedy fast path applies the head to the last
hidden via `head.body` = the inner transformer), so decode arithmetic is unchanged.

Why: the pf cell — and every mlx_lm generate prefill — ran the tied q4/g64 head
(N=248320) over ALL prompt positions. Generation consumes one row. At T=2048 the head
cost 448 ms of the measured prefill (bd2048 `lm_head_full_T` 446.7 ms). Roofline: the
full-T head is compute-bound (2·2048·2048·248320 = 2.08 TFLOP; 446.7 ms = 4.66 TFLOP/s
effective vs macOS 365 ms = 5.71 TFLOP/s — a 1.22x gap consistent with the jwm1 raw-FMA
1.19x bound, so kernel tuning alone could not recover more than ~75 ms). The M=1 decode
head is memory-bound (~300 MB in 1.37 ms = 219 GB/s vs the ~300 GB/s read ceiling).

## Numeric contract (kernel-flags.md class)

The M=1 head GEMM rounds differently from the M=T GEMM (different qmm reduction order):
the last row is NOT bitwise identical (T=512 full 76925c45cd769ea6 vs m1 ea81fdac66343dc0;
T=2048 90919d4471a3a702 vs e9796761582f0cf1). Measured contract (`tools/pg3_lmhead_check.py`,
subprocess per arm, this receipt's `cand/lmhead_check.json`): max |full−m1| = 0.0625 (T=512,
exactly 2.0 bf16 quanta at the reference argmax precision) and 0.015625 (T=2048, 1.0 quantum);
argmax equal at every gate T (6043, 264); both rows finite; per-host deterministic
(two LAST-arm runs identical); top1−top2 margin ≫ max delta (no flip possible).
Same contract class as the GDN chunk-form kernel: determinism + ≤2-quanta rule.

## Paired cells (same window, n=5 fresh processes per cell, serving wheel +5ab5133,
## vec2 ICD 9d949d4-vec2, boot cacd7b45, llm-inference restored health 200 + finish=length)

| Cell | base (serving) | cand (+patch) | Δ | macOS window-4 | ratio base → cand |
|---|---|---|---|---|---|
| pf512 | 894.42 [891.30-895.27] | 1089.55 [1083.31-1094.53] | +21.8% | 1327 | 0.674 → 0.821 |
| pf1024 | 988.93 [985.90-990.30] | 1234.30 [1227.26-1236.85] | +24.8% | 1356 | 0.729 → 0.910 |
| pf2048 | 1032.61 [1032.01-1033.62] | 1307.31 [1304.74-1308.52] | +26.6% | 1380 | 0.748 → 0.947 |

Candidate min-max above base min-max at every length. `ordered_records_sha256`
100a61b62470 in all 30 pf runs of both arms (generation token-identical).
d64 records c84b3e7af640 both arms. Full-logit gate digests (the gate runs under
`MLX_OMARCHY_FULL_LOGITS=1`, so its all-position coverage is unchanged):
f771c4265f88 / ce24f3b4ce42 / b8c4e14f8f8a, finite — identical to the pins.

Kernel-level: `tools/pg3_lmhead_check.py` (acceptance: contract stats above); the wheel
is unchanged, so kernel_bits hashes are the deployed wheel's hashes by construction.

## Files

- `base/`, `cand/` — the paired cell JSONs (pf512/1024/2048 ×5, d64, bd2048), `env.txt`
  provenance (both arms stamp +5ab5133; vulkaninfo names Apple M1 Max G13C).
- `cand/ld-full-logits.json` — the three full-logit gate digests (finite).
- `cand/lmhead_check.json` — the kernel-level contract check (shapes, hashes, quanta).
- Deployed serving-venv state after adoption: `mlx_lm/models/qwen3_5.py` patched
  (sha256 recorded at deploy time in the notebook entry; rollback = revert the patch file
  from the venv, the wheel itself is untouched).
