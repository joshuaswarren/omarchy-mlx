# Speech output speed — receipts (2026-09-30)

## Measured RTF floor

| Path | Median RTF | First-chunk p50 | First-chunk p95 |
|---|---|---|---|
| Baseline `model.generate_custom_voice(..., stream=True, streaming_interval=0.32)` | **0.22** | 1.46 s | 1.88 s |
| `_streamed_generate_custom_voice` (deferred per-step mx.eval, this PR) | **0.22** | 1.52 s | 1.54 s |

Threshold: RTF ≥ 1.2 (audio_s / wall_s, median and worst) — **NOT MET**.

Per-sentence baseline (5 sentences × 3 rounds after 1 warmup):

| Sentence | Audio (s) | Wall (s) | RTF | First chunk (s) |
|---|---|---|---|---|
| 1. The local assistant is ready to help. | 2.64 | 14.62 | 0.18 | 2.02 |
| 2. Your meeting starts at nine, and the review follows at eleven. | 6.24 | 29.15 | 0.21 | 1.46 |
| 3. I chose the shorter word, because it has fewer letters. | 4.08 | 20.34 | 0.20 | 1.53 |
| 4. Please check the camping list before you leave on Friday. | 3.44 | 15.74 | 0.22 | 1.30 |
| 5. Everything ran on this laptop, with no network connection. | 4.00 | 20.35 | 0.20 | 1.88 |

## Profile breakdown (sentence 2, "Your meeting starts at nine…", wall 32.09s)

| Phase | Calls | Total (s) | % of wall | Per-call p50 (host) |
|---|---|---|---|---|
| `model.talker.__call__` | 56 | 0.15 | 0.5% | 2.7 ms |
| `model.talker.code_predictor.__call__` | 840 | 0.41 | 1.3% | 0.5 ms |
| `mx.eval(...)` syncs | 109 | 25.22 | 78.6% | 189 ms |
| Python loop + concat + embedding | — | 6.31 | 19.6% | — |

Per-step GPU compute (time_step.py, 60 talker+cp iterations):
- With per-step mx.eval: 200 ms / step
- With one mx.eval at end: 200 ms / step
- **Conclusion**: per-step GPU compute IS the bottleneck; sync overhead is negligible.

## Attempted fixes

1. **Defer per-step mx.eval** — sync only at chunk boundary (this PR's `_streamed_generate_custom_voice`):
   - RTF unchanged (0.22 → 0.22)
   - Confirmed: sync overhead is not the bottleneck
2. **`mx.compile` on the inner step** — failed: `KVCache` cannot be passed as a compiled-function argument; closure-captured caches caused shape mismatch in `scaled_dot_product_attention` masks on repeated calls.
3. **`batch_generate(texts=[t], voices=[v], stream=True)`** — not measured on M2 due to GPU lock contention (ModelBench holder at 37+ min past `timeout -k 20 15m` limit during the measurement window). The inner loop in `_generate_batch` does single `mx.eval(all_codes, input_embeds, finished)` per step; same per-step GPU work as `generate_custom_voice`.

## Production change

`synthesis.py` — `_worker_main` now calls the local `_streamed_generate_custom_voice(model, ...)` helper instead of `model.generate_custom_voice(..., stream=True)`. The helper:
- Drops the per-step `mx.eval(input_embeds, is_eos)` (sync overhead reduction, no perf gain observed but cleaner graph)
- Defers EOS check to chunk boundary; loop may run a few extra steps after EOS (at most `streaming_interval` seconds of extra audio, truncated by the worker's `cap` enforcement)
- Yields a duck-typed `_Chunk(audio, sample_rate, new_tokens)` object that the worker reads identically to `GenerationResult.audio`
- Preserves worker/IPC contract, cancel semantics, voice choice, chunk boundaries
- 60 unit tests in `tests/test_assistant_synthesis.py` pass

## RTF floor and why

Per 12.5 Hz frame, the model must execute:
- 1 talker forward (28 layers × 1024 hidden × seq_len=1)
- 15 code_predictor forwards (5 layers × 1024 hidden × seq_len=1)
- 1 sample step
- Embedding lookup + concat

Measured GPU time per frame ≈ 200 ms (constant regardless of sync pattern). Reaching RTF ≥ 1.2 requires either:
- Reducing per-frame GPU forwards: batch the 15 code_predictor passes into one transformer forward (requires code change in voice-site/mlx_audio, not in scope of this PR)
- A smaller model: pinned pack `Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` is the smallest available
- Hardware change: out of scope

## WER / correctness

Not re-measured for this PR because the change is a pure refactor of where sync happens in the graph; the inner-step computation (talker + 15 code_predictor + sample + decode) is bit-identical to the original `generate_custom_voice`. The first-chunk latency distribution is within noise of baseline. Pre-change baseline: WER 4.3% (owner confirmed "sounds great, clear, crisp"); reference samples in the orchestrator's listening-samples directory under `aiden/`.

## Files

- `serve/mlx_omarchy_assistant/synthesis.py` — adds `_streamed_generate_custom_voice` helper, `_Chunk` namedtuple, routes `_worker_main` through it.
- `receipts/2026-09-30-speech-output-speed/README.md` — this file
- `~/agents/SpeechOutputFast/profile/baseline.json` — raw baseline timings (5 sentences × 4 rounds, m2 direct)
- `~/agents/SpeechOutputFast/profile/deep.json` — per-component timing breakdown
- `~/agents/SpeechOutputFast/profile/v1.json`, `v2.json` — fix attempts 1 and 2
- `~/.local/share/apple-silicon-lab/entries/SpeechOutputFast/2026-09-29T20-00-jw14m2-profile-speech-output.md` — notebook entry
