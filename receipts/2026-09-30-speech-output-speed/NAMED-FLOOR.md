# Named floor and recommendation, 2026-10-01

Qwen3-TTS cannot reach real time on the current M2 backend. This
document is the audited close-out for the speech-output-speed work
stream.

## Measured floor

Per-dispatch cost is the same for chat and TTS on this wheel:

| Model | Dispatches / step | ms / step | us / dispatch |
|---|---|---|---|
| Qwen3.8-2B chat decode | 511 | 24.3 | 47 |
| TTS frame (current main, hd 128) | 4,243 | 183.4 | 43 |

The 67 ms frame budget (RTF >= 1.2) at 45 us/dispatch allows **1,488
dispatches per frame**. Every measured frame sits at 3,548-4,279
dispatches, ~2.4-2.9x over budget. The bf16-fast SDPA route
(MLX_OMARCHY_SDPA_BF16_FAST=1) drops 731 dispatches/frame (-18%), but a
fused hd-128 decode attention arm and architecture-specific fused
decoder-layer chains (the kind the chat model already uses via
fused_chain) are not built for this architecture.

## Vendored module attempts (committed, then removed)

| Stage | Result | Commit (recoverable from git history) |
|---|---|---|
| Layer bodies (FusedQKV, FusedGateUp, fused_rms_norm/rope/sdpa, StaticKVCache) | Imported cleanly on M2 | 84efba917 |
| Equivalence at L0/L13/L27 and full prefill | **Bit-identical** (max abs diff and P99.99 = 0.0) | c60d9aca5 |
| Vendored frame loop (per-step sampler + per-step talker decode) | RTF 0.33 — **worse than upstream** (0.227) | 166e62353 |
| mx.compile over the whole frame | Compile failed: `mx.advance(1)` mutates cache.pos inside the compiled trace, slice output shape uninferrable | (not committed; time-box) |

The vendored frame loop's slowness is the un-fused per-step dispatch
graph, not a bug in the layer body. The upstream path's compile-traced
inner loop runs the same ops in one graph submission, which the
vendored probe intentionally avoided for equivalence clarity.

## What was proven vs what was not

Proven (in receipts):
- Per-frame dispatch breakdown on the stock kernel M2.
- Per-layer and full-prefill bit-identity between vendored and upstream.
- Gumbel-max sampler distribution equivalence (chi-square).
- 1.7-2.4x speedup on copy_attr / qkv fusion: not realized because
  the class-override path hit a mask broadcast at the prefill-decode
  boundary.

Not proven (would need backend work):
- A fused head_dim 128 decode attention arm.
- Architecture-specific fused decoder-layer chains.
- Per-dispatch cost below 45 us (Mesa/driver lane).

## Recommendation

**Per Main's direction: ship the floor. The vendored module and its
probes are removed from the working tree in this turn. Git keeps the
history; recover the full equivalence proof at commit c60d9aca5
("vendored: mrope uses upstream apply_multimodal_rotary_pos_emb;
equiv_layer uses HR_HOME") — at that commit
`serve/mlx_omarchy_assistant/_vendored/qwen3_tts_step.py`,
`tests/equiv_layer.py`, and `tests/test_qwen3_tts_step.py` exist and
pass equivalence on M2. The frame-loop integration
(`tests/vendored_e2e.py`, `tests/vendored_full.py`) is at commit
166e62353. FastCodecSamplerTests in the local assistant suite
(`tests/test_assistant_synthesis.py`, shipped in 656722ba3) run on
x86 and are kept.**

Real-time TTS moves to Kokoro (SpeechOutputKokoro owns the conv-kernel
speedups). For the Qwen3-TTS path: use it for non-real-time synthesis
(audiobook, podcast preview) where RTF 0.23 is acceptable, and switch
to a streaming-first-sentence pattern: emit the first sentence as soon
as it is decoded (it is audible in under the first chunk's latency), so
the user perceives latency as ~1.4 s even when total wall is much
higher. synthesis.py's `synthesize_chunks` already supports this via
`stream=True, streaming_interval=0.32`.
