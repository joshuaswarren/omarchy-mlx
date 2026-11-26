# 2026-09-29 — GDN coopmat prefill correctness fix + bit-exact SDPA prefill composition speedup

Host: M1 Max (T6001) Omarchy test host, Linux 7.1.6-1-1-ARCH, Vulkan
Honeykrisp (Mesa 26.3.0-devel, mesa-1 `9d949d4`, `AGX_SIMDMAT` default on),
device "Apple M1 Max (G13C C0)". Model SiddhJagani/Qwen3.8-2B-mlx-4Bit snapshot
`0867d98bfb174b042d88461c0e7c97b86b34b381`, mlx-lm 0.31.3 with the shipped
patches (production venv copy). Wheels (release, stamped, provenance verified
against the wheel RECORD before every arm):

| arm | source | wheel stamp |
|---|---|---|
| base | origin/main `af951914a` | `0.32.3.dev202609291300+af95191` |
| c2 | this branch: `49fa3681e` (SDPA) + `61f2cac98` (GDN fix) | `0.32.3.dev202609291337+61f2cac` |

Every GPU run under the llm-inference stop / `flock /tmp/m1-gpu.lock` /
restore discipline; every restore: health 200 + real completion probe
(`finish_reason=length`). Thermal/timing procedure: the run-linux-cells
protocol (5 processes per prefill cell, warmup 1), both arms in one
back-to-back window.

## 1. Defect: GDN coopmat prefill returned NaN logits on real text

`gated_delta_prefill_coopmat.comp` (default for every GatedDeltaUpdate with
T >= 64 since 2026-09-23) built the chunk inverse
`Tinv = (I - N)(I + N^2 + N^4 + N^6)` with its third term as `N^4 . N^4 = N^8 = 0`
(both operands reloaded from `N^4`), so `Tinv` lacked the `N^6 - N^7` terms.
Random near-orthogonal keys keep `N = beta.KK^T` tiny and hid it; real text
puts similar keys in one chunk.

Evidence (base wheel):
- Teacher-forced logits over the 100-prompt corpus
  (`prefill_logits_digest.py`, full-T logits, fresh prompt cache): T=512,
  1024, 2048 non-finite at every position; finite with
  `MLX_OMARCHY_NO_COOPMAT_GDN=1`. Deterministic per host (two runs identical).
- Per-layer replay of captured in-model `gated_delta_update` inputs (T=512)
  vs mlx-lm `use_kernel=False`: layers 0-5 and 7-13 max|diff| 6e-5..0.038;
  layer 6 4.1e29, 14 7.8e20, 15 1.3e7, 16 6.3e20, 17 non-finite from token 456.
- Gate decay was not involved: largest aligned 8-token chunk log-decay -30.8
  (no exp overflow).
- The pf-cell records digest never saw it: it covers 32 tokens after a
  12-token prompt (scan route); the prefill leg is timing only.

Fix (`61f2cac98`): third product is `N^2 . N^4`; the chunk also folds the decay
into `N` (`Gamma = exp(gamma_r - gamma_c)`, `c < r`) and computes
`delta = Tinv . (diag(beta) V - (diag(gamma beta) K) . S)`, so every exponent
is <= 0 (the old `diag(gamma) Tinv diag(exp(-gamma))` form overflowed past a
chunk log-decay of -88.7) and two MACs plus one round trip per chunk go away.

Gates (c2), numeric contract of this kernel (per-host determinism, <= 2 bf16
quanta vs the scan route):

| gate | result |
|---|---|
| full logits T=512/1024/2048, 100-prompt corpus | finite at every position; two runs identical (`f771c4265f88`, `ce24f3b4ce42`, `b8c4e14f8f8a`) |
| vs scan route (`NO_COOPMAT_GDN=1`), argmax per position | 510/512, 1020/1024, 2040/2048 agree; every flip at a reference top-1/top-2 margin <= 1.0 quantum (three exact ties at T=2048); max |delta| at the reference argmax 1.00 quantum |
| per-layer replay, all 18 layers | finite; max|diff| vs `use_kernel=False` <= 2.4e-4 (layers 6/14-17: 1.2e-4, 2.4e-4, 2.4e-4, 6.1e-5, 3.1e-5) |
| `scripts/gdn-coopmat-check.py` correlated-keys case | c2 PASS (p999 0.0 quanta vs scan, oracle error 1.0x scan's); base FAIL (p999 118-121, 4.6-6.8x) |

The correlated case's pre-set bound (max quanta <= 8) failed on both wheels:
the max-quanta figure is dominated by near-zero outputs (54 quanta on the
unchanged uncorrelated 512 case for both wheels). It was replaced by the p999
and oracle-ratio criterion above, which the defective kernel fails and the
fixed one passes; the change is recorded, not silent.

The harness gate (ane-linux-experiments `6ac5360e`): every pf cell of
`run-linux-cells.sh` now runs this full-logit finite check at the cell's T and
prints `LOGITS GATE FAILED` otherwise. On this run: base FAILED at pf512,
pf1024, pf2048; c2 passed all three.

Numerics note: T >= 64 prefill numerics change against every wheel built from
2026-09-23 to `af951914a`; those wheels were wrong there. All coopmat-GDN
prefill timings recorded in that window were measured on the defective kernel
(its instruction count is about the same, so the timings are indicative, but
parity claims need the paired rerun below).

## 2. SDPA prefill composition: bf16 operands staged exactly + causal tile skip (bit-exact)

`49fa3681e`: the bf16 f32-score composition no longer casts q/k/v, scales q in
a separate pass, or casts the result: `matmul_coopmat.comp` attention builds
widen bf16 exactly at staging, apply the scale as the same single f32 product,
and round the accumulator to bf16 with the cast pass's RNE store; causal flags
skip fully masked score tiles and stop the probs k walk at the last reachable
key.

- Kernel hashes (11 SDPA cases: causal/non-causal, GQA 8x2 / 8x8 / 4x1, hd
  64/128/256, T 13..2048, Tk > Tq, batch 2, large logits): identical base vs
  candidate.
- In-model full logits with `NO_COOPMAT_GDN=1` (so both arms are finite):
  identical at T=512 (`468a120fd1b6`) and T=2048 (`c0c039d53eb7`).
- SDPA causal hd256 GQA 8x2 (ms, median of 7): T=2048 21.79 -> 11.78,
  T=1024 6.38 -> 3.48, T=512 2.10 -> 1.30 (macOS same script: 5.50 / 1.58 / 0.61).

## 3. Paired prefill cells (same window, n=5 processes each)

| cell | base tok/s [min-max] | c2 tok/s [min-max] | delta | c2 Linux/macOS |
|---|---|---|---|---|
| pf512 | 800.9 [797.3-806.0] | 807.6 [804.4-810.7] | +0.8% | 0.609 (macOS 1327) |
| pf1024 | 896.8 [896.2-898.4] | 908.8 [904.7-910.1] | +1.3% | 0.670 (macOS 1356) |
| pf2048 | 929.5 [927.4-929.8] | 949.7 [930.9-953.0] | +2.2% | 0.688 (macOS 1380) |

Records digest `100a61b62470` in all 30 runs. Module breakdown at T=2048
(ms/layer, base -> c2): attn.mixer 36.46 -> 26.54, gdn.mixer 32.20 -> 32.43,
mlp 35.4-35.7 both, lm_head(all T) 461.6 -> 465.6; sub-ops: sdpa 21.60 ->
11.68, gated_delta_update 9.50 -> 9.48 (macOS 3.55).

## 4. Refuted on the way (not landed)

One-subgroup GDN workgroups + subgroup-scope round trips + register-pipelined
chunk loads (branch `agent/prefill-parity-l2-refuted`, `45af75344`):
bit-identical but 2.2x slower (T=2048 9.09 -> 20.48 ms); AGX shaderdb shows
6812 instructions, 255 GPRs and 44 scratch spills (base kernel 1420 / 143 / 0).
