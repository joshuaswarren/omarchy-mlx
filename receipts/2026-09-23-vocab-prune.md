# 2026-09-23 — exact greedy vocabulary pruning for the Qwen3.8-2B lm_head (VocabPrune)

Branch `agent/vocab-prune` off `agent/parakeet-final` 51c0f68b9. Target: the
tied 4-bit/group-64 head (V = 248320, K = 2048) that costs 5.465 ms per decode
token on jwm1 (lm_head-shaped GEMV microbench, installed wheel ab51769,
52.3 GB/s; 19% of the 28.2 ms token).

**Status: measured on jwm1 the same day — +3.1% decode outside noise, outputs
bit-identical, wheel installed into the control venv.** The A/B, digest
identity, on-device pruning fraction, and qualification are in section 6.

## 1. The bounds from the brief prune nothing on this model

Numbers come from 310 real decode hidden states (10 corpus prompts x 31 steps,
mlx-lm 0.31.3 on CPU, `receipts-work/vocab-prune/collect.py`), with exact
logits from the dequantized head (`analyze*.py`):

| certificate | rows pruned (median token) |
| --- | ---: |
| Cauchy-Schwarz, ‖w‖‖x‖ < m | 0% (the argmax row's own ‖w‖‖x‖ is 4.46 m) |
| 1-bit code sketch (top bit), exact threshold m | 0% (radius 90.3 vs m − center 24.2) |
| 2-bit code sketch, exact threshold m | 0.01% (radius 38.7) |
| 3-bit code sketch, threshold from the sketch alone | 26% |
| **3-bit code sketch, threshold = exact logit of the best candidates** | **99.68% mean, 184 survivors median, p90 1385** |

The hidden state is dense (L1 5349, L2 156, near-Gaussian L1/L2 ratio) and not
low-rank (the top 128 PCA directions leave 52% of ‖x‖ on held-out tokens). A
worst-case bound on k unknown code bits therefore scales with Σ|s|·‖x_g‖₁.
Three known bits per code is the least that separates the bulk from the max.
The lever's byte ceiling on this head is 896 of 1152 bytes per row: a 22%
reduction of the lm_head read, about 1.2 ms per token at the same bandwidth,
or about +4% decode at best.

## 2. What was built

- `mx.fast.greedy_quantized_argmax(x, w, scales, biases, high_bits,
  group_size=64, bits=4)`: a new Custom primitive
  (`patches/mlx-fast-greedy-argmax.patch`). The composed fallback is exactly
  `argmax(logits - logsumexp(logits, -1, keepdims=True), -1)`. It returns
  `(token, stats[3])`, where stats is [rows evaluated exactly, 1 if the full
  path decided, and the float bits of the max logit, or of the logsumexp on
  the full path].
- The GPU route is `QmmVecGreedyBF16`, a new `QMM_VEC_GREEDY` mode of
  `shaders/qmm_vec.comp`, with 8 dispatches:
  - 0: prologue.
  - 1: bounds from the 3-bit sketch, plus one best-center candidate per
    128 rows.
  - 2: exact logits of the candidates, giving LB*.
  - 3: compaction.
  - 4: exact survivors.
  - 5: select, plus the merge check.
  - 6–7: the full path. Each workgroup exits after one state read unless
    stage 5 flagged the token.

  Every other `qmm_vec` variant compiles to byte-identical SPIR-V before and
  after the change (all 23 CMake define sets checked with glslc).
- The eligibility gate in `eval_gpu` matches QuantizedMatmul's own pick of
  `QmmVecQ4WordSubgroupBF16`: subgroup size 32 with arithmetic ops, the
  `MLX_OMARCHY_QMM_VEC_Q4_WORD` default, bf16 storage. Anywhere else the
  composed fallback runs.
- `patches/mlx-lm-greedy-prune.patch` (mlx-lm 0.31.3 `generate_step`):
  - It applies when the sampler is the default, there are no logits
    processors, the input is one token, and the head is a tied 4-bit/g64
    Qwen3.5 head. The step then runs the body, the greedy op, and a *lazy*
    `logprobs` graph. The full projection runs only if a consumer evaluates
    `logprobs`, and `async_eval` drops `logprobs`.
  - It builds the 3-bit sketch (190.7 MB) once per process. It refuses any
    non-finite scale or bias, and any at or above 2^20.
  - Kill switch: `MLX_OMARCHY_NO_GREEDY_PRUNE=1` restores the upstream step.
    Documented in `docs/kernel-flags.md`, along with
    `MLX_OMARCHY_GREEDY_PRUNE_TEST=full|keep` for qualification.

## 3. Proof that the token cannot change

Notation: u = 2^-24 and γ_n = nu/(1−nu). The composed path computes these
quantities:

- y_j = RNE_bf16(S_j), with S_j the fp32 output of the
  `QmmVecQ4WordSubgroupBF16` column.
- L = RNE_bf16(LSE_32(y)), with LSE_32 the `logsumexp_suffix.comp` arithmetic.
- p_j = RNE_bf16(fl(y_j − L)).
- token = min{j : p_j = max p}. `argreduce_suffix.comp` breaks ties toward
  the smaller index.

Let m = max y, i* = min{j : y_j = m}, L1_g = Σ_{i∈g}|x_i|, X_g = Σ_{i∈g} x_i,
and A_j = Σ_g (16|s_g| + |b_g|)·L1_g.

**L1 (full-kernel rounding).** Take the exact value
z_j = Σ_g (s_g Σ q_i x_i + b_g X_g). Every product x_i·q_i is exact
(8-bit × 4-bit significands). Every leaf passes through at most 51 roundings:
16 in the block chain, 1 for the bias product, 1 for the fma, 3 for the lane
accumulation, and ≤ 31 for subgroupAdd in any order. So
|S_j − z_j| ≤ γ_51·A_j < 2^-18·A_j. Stage 0 routes any |x_i| > 2^20 or NaN to
the full path, and the patch refuses |s|, |b| ≥ 2^20. There is therefore no
overflow. Subnormal or flush-to-zero error is ≤ 10^4·2^-126, absorbed by the
2^-40 term.

**L2 (sketch interval).** Write q = 2a + l with a = q >> 1 (the sketch) and
l ∈ {0, 1}. Then z_j = C_j + Σ_g s_g Σ (l_i − ½)x_i, where
C_j = Σ_g [2 s_g Σ a_i x_i + (½s_g + b_g) X_g]. So |z_j − C_j| ≤ W_j = ½ Σ_g |s_g| L1_g.

**L3 (bound-kernel rounding).** Each leaf of C̃, W̃, and Ã passes through
fewer than 128 roundings: 8+8 in the prologue sums, a 64-term fma chain, the
coefficient, and ≤ 31 in subgroupAdd. The leaf magnitudes sum to ≤ A_j. Hence
|C̃ − C| ≤ γ_128·A_j, W ≤ W̃ + γ_128·A_j, and Ã ≥ (1 − γ_128)·A_j.

**T1 (upper bound).** Stage 1 stores
UB_j = fl(C̃ + ½W̃ + 2^-12·Ã) + 2^-40, and UB_j ≥ S_j. The proof follows from
the chain S_j ≤ z_j + γ_51·A_j ≤ C_j + W_j + γ_51·A_j. The error terms
(2γ_128 + γ_51)·A_j plus the ≤ 3u·3A_j of the final adds total
≈ 2.4e-5·A_j. The slack 2^-12·Ã ≥ 2.44e-4·A_j·(1 − γ) covers that about
10 times over.

**T2 (threshold).** LB* is the max of candidate values computed by the
identical column: the same `Q4_ROW` macro source, the same matrix_n (so the
same qmv_fast tile), the same subgroupAdd, the same `bf16_store`. So
LB* = y_c for a real row c, and m ≥ LB*. The on-device qualification checked
this identity bit for bit (stats[2], 601 cases x 3 modes, 0 mismatches,
section 6).

**T3 (pruning).** Stage 3 drops row j only if
LB* − UB_j > G + |UB_j|·2^-7, with G = 0.13 + max(|LB*|, |UBmax|)/12800.
RNE moves a value by ≤ |v|·2^-8, and the doubled 2^-7 absorbs the fp32
evaluation of the test. So y_j ≤ UB_j + |UB_j|·2^-8 < LB* − G ≤ m − G. A
pruned row is strictly below m by more than G.

**L4 (logprob merge).** L ≥ m, because row_sum ≥ exp(0). Also
L − m ≤ ln V + 2e-4 + ulp: each exp term is ≤ 1 + 3 ulp, the sum error is
≤ γ_978, and log has 3 ulp error. For y_j < m, the values fl(y_j − L) and
fl(m − L) round to different bf16 values when m − y_j exceeds the bf16
spacing at L − y_j plus fp32 error. It suffices that
m − y_j > 0.00789·(13 + |m|/128). Both G and the stage 5 guard
0.13 + |m|/12800 exceed this for every m (0.13 > 0.1026, 1/12800 > 0.00789/128·1.004).

**T4 (selection).** Every row with y_j = m survives (T3). So stage 5's
first-index max over the survivors is i*. Suppose no survivor j < i* has
0 < m − y_j ≤ G_m. Then every j < i* has p_j < p_{i*} (L4, from T3 or from
the survivor gap), and every j > i* has p_j ≤ p_{i*}. So the composed token
is i*.

In every other case stage 5 flags. The flag cases are: such a survivor
exists, a NaN appears, x is refused, or no row survives. Stages 6–7 then
compute every y_j with the same column, and L with the LogSumExpBF16
statement sequence. They then take p_j by the bf16 subtract of
`binary_vec`/`elementwise` (fp32 `lhs − rhs`, then `bf16_store`). The
argmax uses the ArgMaxBF16 statement sequence. That equals the composed path
statement for statement.

What the proof rests on is verified: `greedy_qual.py` checked the shared
SPIR-V arithmetic, the copied logsumexp, and the argmax bit for bit on the
device through stats[2] — 601 cases x 3 modes, 0 token mismatches and 0 bits
mismatches (section 6).

## 4. Offline evidence (this workstation)

- The sketch layout: `test_plane.py` checked the mlx builder against the
  shader's `GREEDY_A` extraction on 37 random rows × 2048 codes. Result:
  0 mismatches.
- The kernel TUs (`primitives.cpp`, `compute.cpp`, `fast.cpp`, the python
  `fast.cpp`) compile in an x86 omarchy configure. The greedy shader
  compiles under glslc and under the build image's glslangValidator 16.4.
- The macstudio ALARM chroot built the wheel
  `mlx_omarchy-0.32.3.dev202609231347+a91adbf-cp314-cp314-linux_aarch64.whl`,
  sha256 `0cbb48ec9b9aa2be13d8380b88521a6533424aa94a5ecf9f0052c3788b236f7e`.
  It sits at macstudio:~/src/VocabPrune-build/dist-out/. Image
  dg-alarm-py314:sep23, `DEV_RELEASE=1 MLX_OMARCHY_SOURCE_COMMIT=a91adbf`.
  Commit 3cc605a58 changes only the mlx-lm patch and docs, not the wheel.
- The mlx-lm plumbing test (`test_lm_patch.py`, CPU, stub op) never finished
  and was superseded: the patched `generate_step` ran on the device in the
  candidate arm of the A/B (section 6).

## 5. Control for the A/B

The installed wheel is `0.32.3.dev202609231206+ab51769` (sha256 9011fe27…, the
PkDeep receipt, macstudio:~/src/ParakeetFinal-build/dist-out/). jwm1's
`/var/tmp/v072-venv-fused` currently has **no mlx-lm**, so both arms need
fresh venvs: the wheel, mlx-lm 0.31.3, and the dg GDN patch scripts. The
candidate also gets `patches/mlx-lm-greedy-prune.patch`.

## 6. The jwm1 window (2026-09-23, measured)

Staged on jwm1:/var/tmp/vp/: both wheels (sha256 9011fe27… and 0cbb48ec…,
verified on arrival), `vp-run.sh`, `greedy_qual.py`, and
`patches/mlx-lm-greedy-prune.patch` (identical to the committed copies).
Fresh venvs: `venv-ctl` (ab51769) and `venv-cand` (a91adbf + the greedy
mlx-lm patch), both mlx-lm 0.31.3 + the dg GDN patches.

**Qualification-first gate** (before any timing, `/var/tmp/vp/qual-pre/`,
each mode under a 600 s timeout — no hangs): all three modes rc=0 with
601 cases, **0 token mismatches and 0 stats[2] bits mismatches**.

| mode | full-path cases | corpus steps on full path | survivors median / p90 / max | greedy op ms | full ms |
| --- | ---: | ---: | --- | ---: | ---: |
| prune | 61/601 | 6/320 | 193.5 / 2202.5 / 31510 | 5.50 | 6.55 |
| keep (pruning off) | 61/601 | 6/320 | all 248320 | 14.47 | 6.55 |
| full (forced full path) | 601/601 | 320/320 | — | 14.13 | 6.55 |

**On-device prune fraction:** 99.62% of rows pruned per decode step on the
10-prompt corpus (mean); the selection fell to the full path on 6 of 320
corpus steps.

**Decode A/B** (3 interleaved reps per arm, contract bench, warmup 2,
prefill 512; window 09:54:34–10:01:56 EDT under flock /tmp/m1-gpu.lock):

| arm | decode tok/s per rep | mean ± 95% CI |
| --- | --- | --- |
| ctl (ab51769) | 35.42, 35.43, 35.22 | 35.36 ± 0.29 |
| cand (a91adbf + prune) | 36.48, 36.42, 36.44 | 36.45 ± 0.08 |
| candoff (kill switch) | 35.39 | — |

Paired per-rep deltas +1.06, +0.99, +1.22 tok/s → **mean +1.09 ± 0.29 tok/s
(95%, t, df = 2), ratio 1.031 (1.028–1.035)**. The CI is entirely positive:
a win outside noise, and consistent with the section 1 ceiling of about +4%.
The kill-switch arm lands on ctl, so the gain is the prune route, not the
rebuild.

**Correctness, identical on every measure:**

- Ordered digest `ordered_records_sha256 = bc519c03…` is the same for all 7
  runs (both arms, all reps, kill switch).
- Logits gate v4: `flipped_prompts: 0`, `max_abs_logit_delta_prefix: 0.0`.
- Qualification: 0 token / 0 bits mismatches in all three modes.

**Installed.** `mlx_omarchy-0.32.3.dev202609231347+a91adbf` (0cbb48ec…) is
now in `/var/tmp/v072-venv-fused` (`pip install --force-reinstall
--no-deps`), replacing ab51769 as the control venv's wheel. Smoke in that
venv under the lock: the greedy op matches the composed reference on 64
synthetic quantized cases on the GPU (the synthetic ties engage the full
path; tokens identical). Rollback:
`pip install /var/tmp/vp/mlx_omarchy-0.32.3.dev202609231206+ab51769-*.whl`.

Two notes for the next person:

- `/var/tmp/tdtfused-venv` is a filesystem copy of `v072-venv-fused`: its
  `bin/pip` shebang points at v072's python, so pip scans of it report
  v072's versions. Its own site-packages holds a stale
  `bf8793f` dist-info. It was never an ab51769 venv and was left as found;
  upgrade it only after rebuilding the venv.
- Wiring `patches/mlx-lm-greedy-prune.patch` into
  `scripts/apply-mlx-lm-patches.sh` and `install.sh` is still open — needed
  only when fresh mlx-lm installs should adopt the prune.
