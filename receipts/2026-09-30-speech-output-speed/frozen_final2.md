
# Update to frozen-window report, 2026-09-30 21:34 CDT

The vendored layer bodies ran against the upstream model on M2 (boot
0e2c3743) and revealed a code bug: the script used
`cfg.hidden_size // cfg.num_attention_heads = 64` to compute head_dim,
but the actual Qwen3-TTS talker uses `cfg.head_dim = 128` (a stored
config field, not derived). The shape error is `reshape 2048 -> (1,1,16,64)`
because the q_proj output (16 × 128 = 2048) doesn't fit 16 heads of 64.

Fix committed in this turn: equiv_layer.py now uses `head_dim = cfg.head_dim`
and the vendored module's split sizes will work.

The disk-boot tests at 21:45 CDT prevent running another ticket.
Resume plan: re-submit `tests/equiv_layer.py` after the M2 comes back up.

What was measured:
- Model loads cleanly (no missing weights, no schema errors).
- The bug was on the test side, not in the vendored forward.
- No equivalence numbers yet.

Verified vendored module structure (from the syntax check during
import): FusedQKV, FusedGateUp, fused_rms_norm, fused_rope, fused_sdpa,
StaticKVCache, vendorize_talker, vendorize_cp, gumbel_max_sample,
talker_layer_forward, cp_layer_forward. Import succeeds under M2's
mlx wheel.
