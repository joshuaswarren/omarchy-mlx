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
| `MLX_OMARCHY_NO_COOPMAT_GDN` | unset (route on) | `1` forces the GDN prefill T>=64 token chunks onto the two-pass scan kernel instead of the single-pass coopmat kernel (2026-09-23 install, 2.86x prefill512, receipts 2026-09-23-gdn-coopmat-install); output-neutral at the logit gate. |
| `MLX_OMARCHY_QMM_VEC_Q4_WORD` | on | Packed q4 word path in qmm gemv (xpack). `0` reverts to scalar; may change which ulp-level near-ties flip. |
| `MLX_OMARCHY_QMM_TILE`, `_QMM_TILE_RB`, `_QMM_COOPMAT_WG_PER_CORE` | tuned defaults | Tile/workgroup sizing overrides for qualification. |
| `MLX_OMARCHY_GREEDY_PRUNE_TEST` | unset | Qualification only. `full` forces `mx.fast.greedy_quantized_argmax` onto its full exact path (every row, logsumexp, subtract, argmax); `keep` keeps every row as a survivor. Token-neutral by construction (receipt 2026-09-23-vocab-prune). |
| `GDN_FALLBACK_DEBUG` | unset | Prints fused-eligibility inputs for the GDN prefill route when set. |
| `MLX_OMARCHY_FUSED_AB` | — | ANE-side experimental flag; **ignore on the GPU path** — it has no effect on the Vulkan backend. |

## mlx-lm serve patches (vendored in `patches/`, applied by
`scripts/apply-mlx-lm-patches.sh`, wired into `install.sh`)

- `mlx-lm-gated-delta-fast-route.patch` — **default ON.** Routes
  `mx.fast.gated_delta_update` in mlx-lm 0.31.3 to the fused kernel in the
  mlx-omarchy wheel. Required for the measured serve numbers.
- `mlx-lm-convring.patch` — **default OFF.** Rolling conv-state ring; enable
  per venv with `MLX_OMARCHY_CONV_RING=1 scripts/apply-mlx-lm-patches.sh
  /path/to/venv`.
- `mlx-lm-greedy-prune.patch` — **not applied yet** (no jwm1 A/B so far;
  see receipts/2026-09-23-vocab-prune.md). Greedy decode steps
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
