# 2026-09-29 jwm1 (M1, T8103): GDN decode conv routed to mx.fast.gdn_conv_update by default

Same host/stack/protocol as receipts/2026-09-28-jwm1-gdn-qk-scaled.md (installed wheel 42fbbc5 already exports `gdn_conv_update`; the
mlx-lm routing was never applied by the installer). Control = installed venv with the q/k patch; candidate = byte copy plus
`patches/mlx-lm-qwen35-gdn-conv.patch` (from scripts/patch-mlx-lm-qwen35-gdn-conv.py). Interleaved, one flock hold per run,
loadavg 0.01 before start. Private notebook entry jwm1-parity 20260929T001500Z (H10).

| cell | ctl tok/s | cand tok/s | ratio | digest (both arms) |
|---|---:|---:|---:|---|
| decode64 (n=15) | 39.94 | 40.64 | 1.0175 | 7fe6badf4d560e25 |
| decode128 (n=10) | 38.95 | 39.52 | 1.0146 | da5568eeb4b6a1c1 |
| decode256 (n=5, one round) | 37.91 | 38.66 | 1.020 | 7d0523ae795bf83f |
| prefill512 digest | - | - | prefill 240.13 vs 241.64 (noise) | 26475860d8a29819 |

Bit-identical in every arm; ranges do not overlap at 64 tokens (ctl max 40.09 < cand min 40.26). The pre-registered pass bar was >= 1.02x
and decode64/128 came in at 1.0175/1.0146, i.e. missed the bar but above the 1.01x reject floor; shipped because the change is bit-identical
and consistently positive, not because it met the bar. Routing is decode-only (S==1), bf16, no-lengths cache branch; prefill is unchanged.
