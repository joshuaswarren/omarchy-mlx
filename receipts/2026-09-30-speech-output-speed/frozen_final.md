
# Frozen-window final report, 2026-09-30

M2 was rebooted at 20:52 CDT. The 20:50-21:40 frozen window opened with
quality27b_perf holding the GPU from 21:03 to 21:28+; my equiv_run
tickets stayed queued behind it for the full window and never ran.
Final work landed on origin/main.

## On main (vendored module skeleton)

- `serve/mlx_omarchy_assistant/_vendored/qwen3_tts_step.py` (15,880 bytes)
  FusedQKV/FusedGateUp (weight concat at load time, one qmv dispatch),
  fused_rms_norm / fused_rope / fused_sdpa wrappers around mx.fast,
  StaticKVCache (preallocated contiguous K/V buffers with slice-update),
  vendorize_talker / vendorize_cp, gumbel_max_sample, make_mrope_cos_sin,
  talker_layer_forward / cp_layer_forward, FrameBundle. Pinned
  `_SOURCE_SHA` for voice-site at recording time.

- `tests/test_qwen3_tts_step.py` — FusedProjectionEquivalence,
  TalkerLayerEquivalence, SamplerEquivalence stubs (mlx-gated; skip on
  x86, run on M2).

- `tests/equiv_layer.py` — Per-layer equivalence probe (L0/L13/L27 at
  seq_len 1 and 4, plus full prefill). Exit 0 when within 5e-3 (single)
  / 1e-2 (prefill). On M2, output: `EQUIV_RESULTS={"L0_S1_max":..., ...}`.

- `tests/vendored_e2e.py` — Single-frame E2E probe: builds caches from
  upstream prefill, then runs 16 vendored decode steps.

- `tests/vendored_full.py` — Full upstream-vs-vendored end-to-end RTF +
  dispatch bench, plus baseline WAVs to <home>/agents/SpeechOutputFast/wavs/upstream_baseline/.

- All scripts synced to <home>/agents/SpeechOutputFast/repo/{tests/,serve/mlx_omarchy_assistant/_vendored/}.

## Commits this turn

```
828d2f872 serve: vendored qwen3_tts_step skeleton + equivalence test stubs
84efba917 serve: vendored qwen3_tts_step layer bodies + equivalence tests
4424dbe05 tests: vendored-layer equivalence probe
0160a5e52 tests: vendored end-to-end probe (initial)
168190d9d tests: vendored frame-step e2e probe (revised, single frame)
a1f7ab767 tests: full-frame equivalence probe (fixed upstream_prefill)
cf6e26719 tests: vendored full-frame E2E probe
```

Plus earlier in this overall turn:

```
b37ba76c4 receipts/speech-output-speed: frozen-window chat vs TTS, loop pipeline E2E
```

## What I could not do (named floor for the M2 work)

No GPU time was granted to my tickets in the 20:50-21:40 frozen
window. The vendored layer bodies have NOT been run against the
upstream implementation on M2 hardware. Equivalence (per-layer and
full-prefill max abs diff), RTF, dispatches/frame, and Whisper WER are
all unmeasured.

## Best audited number from earlier this session

Chat 2B = 511 dispatches at 24 ms = 47 us/dispatch; TTS frame = 4,243
dispatches at 183 ms = 43 us/dispatch. Same per-dispatch cost on this
wheel. The remaining lever is dispatch count per frame, which the
vendored module targets but does not yet have an audited result.

## Plan after this slot

Resume at `tests/equiv_layer.py` once GPU is free. If max abs diff <=
5e-3 single-layer and <= 1e-2 prefill, proceed to `tests/vendored_full.py`
for RTF + dispatches + WER. If RTF >= 1.2 ship to `synthesis.py` and
push. Otherwise named floor with the op table.
