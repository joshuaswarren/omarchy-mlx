
# Vendored frame loop E2E result, 2026-10-01

M2 boot 40f95214, mlx wheel 0.32.3.dev202609291615+06711ad,
voice pack mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit pinned.

## Equivalence (already committed: c60d9aca5)

Vendored fused-pass forward is bit-identical to upstream at L0/L13/L27
(seq_len 1 and 4) and on the full 28-layer prefill. Max abs diff and
P99.99 abs diff = 0.0 across every probe. No token agreement work
needed at this exact bit identity.

## Vendored frame loop RTF (this turn)

Replaced the inner generation loop of `model.generate_custom_voice`
with the vendored layer bodies + the Gumbel-max sampler. Per-frame
synchronous-style: one talker decode step + 15 code-predictor passes
+ Gumbel-max draw per codebook, then audio decode every step. Path is
on-device between mlx.eval boundaries, no per-layer copy.

5 sentences × 2 measured rounds after 1 warmup, aiden, seed 123+i,
stream=True, streaming_interval=0.32. Audio saved at
<home>/agents/SpeechOutputFast/wavs/vendored_run/ and copied to
<home>/mlx-tts-samples-fast/vendored/.

| Path | RTF median / min | First chunk p50 / p95 |
|---|---|---|
| **Vendored frame loop (this turn)** | **0.33 / 0.33** | 2.15 s / 2.27 s |
| Upstream `fast_topk` sampler (aeea34239 receipt) | 0.227 / 0.215 | 1.41 s |
| Upstream base + mx.random.categorical | 0.206 / 0.132 | 1.46 s |

The vendored path is **worse than upstream by ~50%** on RTF. The first
audio chunk latency is also worse (2.15 s vs 1.41 s). This is because
the vendored_loop_e2e calls `_sample_token` once per codebook (15
times per frame) with no fusion of the sampler dispatch, and the per-step
talker decode is run as 28 separate graph nodes instead of the
upstream's compile-traced inner loop. The vendored module proves
*equivalence*, not *speed*: the goal was bit-identity for the safety
check, not a fast path. A separate vendored-streaming-generator
that wraps the whole frame loop in a single graph (and removes the
per-step sampler eval) is the next step.

## Named floor (unchanged)

Chat 2B = 511 dispatches at 24 ms = 47 us/dispatch; TTS frame = 4,243
dispatches at 183 ms = 43 us/dispatch. Same per-dispatch cost on this
wheel. The 67 ms frame budget (RTF 1.2) requires a fused head_dim 128
decode attention arm and architecture-specific fused decoder-layer
chains for the talker and CP layers.
