# Kernel feature and serve-patch flags (Qwen3.8-2B integration, 2026-09-22)

Runtime switches for the integrated kernel set (fused GDN decode/prefill,
bf16 coopmat qmm prefill, q4 gemv xpack) and the vendored mlx-lm serve
patches. All default to the shipped configuration; the flags exist for
A/B gating and qualification, not for normal operation.

## GPU kernel flags (backend, wheel)

| Flag | Default | Effect |
|---|---|---|
| `MLX_OMARCHY_KV_DIRECT` | on (`1`) | Direct decode KV window storage in fused chains. `0` composes ops; output-neutral. |
| `MLX_OMARCHY_NO_COOPMAT` | unset (route on) | `1` disables the bf16 coopmat qmm-prefill route; output-neutral. |
| `MLX_OMARCHY_NO_COOPMAT_GDN` | unset (route on) | `1` forces the GDN prefill T>=64 token chunks onto the two-pass scan kernel instead of the single-pass coopmat kernel (2026-09-23 install, receipts 2026-09-23-gdn-coopmat-install). Wheels built before 2026-09-29 carry a defective chunk inverse (N^6 computed as 0) that turns real-text prefill logits NaN; set `1` on those. The fixed kernel is held to the contract below against the scan route (receipts/2026-09-29-prefill-gdn-fix-sdpa-direct.md). 2026-10-03: the captured 27B operands (layer0/layer12, incl. g=5.9e-11) became checked-in fixtures (`tests/omarchy/fixtures/gdn_coopmat`) alongside the fp64 sweep — fused state max-err 1.4e-5 / 1.7e-6 vs fp64, zero NaN; gate floor lowered to FLT_MIN (2026-09-29..10-02 wheels clamp at 1e-6, mis-stating sub-floor decays ~1.7e4) and the batch kernel's wave store->read accesses are now explicitly ordered (receipts/2026-10-03-gdn-coopmat-fix). |
| `MLX_OMARCHY_GDN_DECODE_BATCH` | unset (route on) | `0` restores the `B == 1` gate on the fused GDN decode kernel, so batched decode (B > 1, T = 1, no padding mask) runs the composed per-token chain again. On by default since 2026-10-07: oMLX c4 aggregate 58.1 -> 71.8 tok/s on an M1 Max; each batch row is bit-identical to the same row decoded alone. |
| `MLX_OMARCHY_SDPA_DECODE_BATCH` | unset (route on) | Read live. `0` restores the `batch == 1` gate on the native bf16 SDPA decode kernel, so batched single-query decode runs the ~10-dispatch composition again. The kernel walks the batch on workgroup z; each row is bit-identical to the same row decoded alone (and to the composition, which the bf16 arm reproduces exactly). |
| `MLX_OMARCHY_KV_MASKLESS` | unset (on) | mlx-lm patch (`scripts/patch-mlx-lm-kv-maskless.py`). `0` makes `BatchKVCache.make_mask(1)` return the array mask again when no row is padded. With it on, unpadded batched decode passes no mask, so the native SDPA decode kernel stays reachable; padded batches, prefill and caches whose `left_padding` was assigned from outside the class keep the mask. |
| `MLX_OMARCHY_KV_HOST_OFFSET` | unset (off) | mlx-lm patch (`scripts/patch-mlx-lm-kv-host-offset.py`, after kv-maskless). `1` makes `BatchKVCache` store `offset` as an array built from its host mirror (same int32 values) after every in-class update, so the RoPE trig gate reads it without the per-call `rope_offset_scalar` host join that a lazy device offset forces. Outside assignments clear the mirror and keep the assigned array. Opt-in until the oMLX c1 A/B is in. |
| `MLX_OMARCHY_BATCH_GREEDY` | unset (off) | mlx-lm patch (`scripts/patch-mlx-lm-batch-greedy-head.py`, 0.32 line). Opt-in. `1` makes a `GenerationBatch` whose rows all sample greedily without logits processors take each token from the pruned greedy head (`mx.fast.greedy_quantized_argmax`, one dispatch per row; same token as the full-logits argmax) and leave the logprobs lazy. On jw16 (2026-10-07) it cost +1.9% ms/token at B=1 and +1.6% at B=4 in-process and -2.5% at oMLX c1; only oMLX c4 gained (+4.7%). |
| `MLX_OMARCHY_QMV_BATCH` | unset (off) | Experimental, read live. `1` sends a bf16 `[B, 1, K]` decode batch (2 <= B <= 4) against one transposed 2D 4-bit/group-64 weight to `QmmVecQ4WordSubgroupBatch4BF16`, which reads each weight word once per block instead of once per row; each row keeps the single-row chain and must match the per-row kernel bit for bit (`tests/omarchy/test_qmv_batch.cpp`). Off, the existing kernels and their SPIR-V are unchanged. |
| `MLX_OMARCHY_SDPA_CAUSAL_BLOCK` | unset (off) | Experimental prefill lever, mlx-lm patch (`scripts/patch-mlx-lm-sdpa-causal-blocks.py`). `<rows>` splits a causal prompt longer than `rows` into row blocks that each attend only to their key prefix (bottom-right aligned `mask="causal"`), skipping the fully masked half of the composed `[heads, L, L]` score square; causal prompts never reach the flash route (non-causal hd128 only). Each row sees exactly the keys it saw before. |
| `MLX_OMARCHY_QMM_VEC_Q4_WORD` | on | Packed q4 word path in qmm gemv (xpack). `0` reverts to scalar; may change which ulp-level near-ties flip. |
| `MLX_OMARCHY_QMM_TILE`, `_QMM_TILE_RB`, `_QMM_COOPMAT_WG_PER_CORE` | tuned defaults | Tile/workgroup sizing overrides for qualification. |
| `MLX_OMARCHY_GREEDY_PRUNE_TEST` | unset | Qualification only. `full` forces `mx.fast.greedy_quantized_argmax` onto its full exact path (every row, logsumexp, subtract, argmax); `keep` keeps every row as a survivor. Token-neutral by construction (receipt 2026-09-23-vocab-prune). |
| `GDN_FALLBACK_DEBUG` | unset | Prints fused-eligibility inputs for the GDN prefill route when set. |
| `MLX_OMARCHY_FUSED_AB` | — | ANE-side experimental flag; **ignore on the GPU path** — it has no effect on the Vulkan backend. |

## mlx-lm serve patches (vendored in `patches/`, applied by
`scripts/apply-mlx-lm-patches.sh`, wired into `install.sh`)

- `patches/mlx-lm-0.32/` — the same series rebuilt for the mlx-lm 0.32 API
  line (commit 94cdcae, the pin oMLX uses; the PyPI 0.32.0 wheel carries
  byte-identical patch-site files), so oMLX can run on the Omarchy
  stack. The apply script picks it when `mlx_lm.generate` defines
  `StopSequences`. conv-ring has no 0.32 port; the script refuses
  `MLX_OMARCHY_CONV_RING=1` there before touching the venv. Receipt:
  receipts/2026-10-02-mlx-lm-032. Token note: the 0.32 stack does not
  reproduce the 0.31.3 stack bit-for-bit on every prompt (upstream
  rescaled the q/k-norm eps, and the raw decode route is conditional);
  on the T6001 10-prompt contract corpus, 3/10 Qwen3.8-2B prompts flip
  one near-tie token. The quality gate (harness v2 GSM8K/IFE vs the
  shipping stack) passed within its pre-declared noise band, which is
  what the acceptance bar below requires for a token change; 4B/9B are
  bit-identical and perf is neutral.
- `mlx-lm-gated-delta-fast-route.patch` — **default ON.** Routes
  `mx.fast.gated_delta_update` in mlx-lm 0.31.3 to the fused kernel in the
  mlx-omarchy wheel. Required for the measured serve numbers.
- `patch-mlx-lm-gdn-raw-repeat.py` — **default ON on all chips.** Set
  `MLX_OMARCHY_GDN_RAW_REPEAT=0` to opt out of the Qwen3.5-9B fused raw
  decode route (+31-35% on G14-class; +35-43% on G13). The published-wheel
  audit measured 99.51% teacher-forced agreement; all first free-run
  divergences were within one bf16 ULP. See docs/numerics-gate.md for the
  criteria and receipts/2026-10-04-gdu-bar-audit/README.md for evidence.
- `mlx-lm-convring.patch` — **default OFF.** Rolling conv-state ring; enable
  per venv with `MLX_OMARCHY_CONV_RING=1 scripts/apply-mlx-lm-patches.sh
  /path/to/venv`.
- `mlx-lm-greedy-prune.patch` — **default ON** (jwm1 A/B in
  receipts/2026-09-23-vocab-prune.md). Greedy decode steps
  (default sampler, no logits processors, one input token) of tied
  4-bit/group-64 Qwen3.5-family heads take `mx.fast.greedy_quantized_argmax`:
  a certified bound on a 3-bit code sketch skips the rows that cannot hold
  the argmax, so the token equals `argmax(logits - logsumexp(logits))`
  bit for bit while most of the vocabulary projection is never read.
  `logprobs` stay lazy and run the full projection only when a consumer
  evaluates them. Kill switch: `MLX_OMARCHY_NO_GREEDY_PRUNE=1` at run
  time restores the upstream step exactly. The sketch holds 3 bits per
  code, three quarters of the packed codes (190.7 MB for Qwen3.8-2B's
  248320-row head), built once per process at the first greedy call.
- The patches are idempotent and fail loudly on mlx-lm version drift.

## Acceptance bar

Cross-host digest equality is not a gate (see README performance section
and `ane-linux-experiments` receipts 2026-09-22-qwen38-integration /
2026-09-22-qwen38-correctness): per-host determinism plus logit-level
equivalence (≤ 2 bf16 quanta) is.
