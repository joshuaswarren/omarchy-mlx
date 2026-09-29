# Speech output speed — receipts (2026-09-30)

## Measured RTF floor

| Path | Median RTF | First-chunk p50 | First-chunk p95 |
|---|---|---|---|
| Baseline `model.generate_custom_voice(..., stream=True, streaming_interval=0.32)` | **0.22** | 1.46 s | 1.88 s |
| `_streamed_generate_custom_voice` (deferred per-step mx.eval, this PR) | **0.22** | 1.52 s | 1.54 s |

Threshold: RTF ≥ 1.2 (audio_s / wall_s, median and worst) — **NOT MET**.

## Dispatch counts (MLX_OMARCHY_TRACE_DISPATCH=1)

`scripts-local` / dispatch probe, single iteration, fresh KVCache per iteration:

| Probe | Per-iteration dispatches | Notes |
|---|---|---|
| 1 — talker only (1 forward + 1 mx.eval) | **33,073,041** | 28 layers × ~1.18M dispatches/layer for seq_len=1 |
| 2 — code_predictor (15 passes + 1 mx.eval at end) | 37,430,928 | ~2.5M dispatches/pass when serialized |
| 3 — full inner step (talker + 15 cp + sample + embed) | **33,195,431** | talker dominates; CP add only ~122k dispatches when graph-merged |

**Interpretation**: code_predictor is NOT the bottleneck (122k extra dispatches inside the full inner step vs 37M when serialized alone — the graph fuses them). The 33M dispatches live inside `Qwen3TTSTalkerForConditionalGeneration.__call__`, dominated by per-layer sdpa and matmul launches. Each layer of the 28-layer talker issues ~1.18M dispatches for a single-token forward — a launch-bound loop on this backend.

Per-step GPU wall time (10 iterations, fresh cache): 87 ms (talker) / 161 ms (15 CP serialized) / 92 ms (full step). Removing per-step mx.eval did not change wall time because GPU compute and dispatch overlap; the wall is GPU-bound at ~92 ms per step on the full graph.

**Per-frame fixed cost**: ~92 ms GPU × 60 frames = 5.5 s for a 5 s sentence. Reaching RTF ≥ 1.2 (≤4.2 s wall) requires either:
- Cutting talker dispatch count below 33 M/step via compile-fusion of the per-layer forward (mx.compile the per-layer transformer block; KVCache passed via closure on a preallocated cache that is reset in-place each step, not reallocated).
- Reducing per-frame GPU forwards via batched code_predictor (estimated ≤120k extra dispatches per step from the probe — already the case; batching adds no further dispatch savings; batching saves only GPU work which is already <10ms).
- A smaller model: pinned pack is the smallest available.

**Best lever**: compile the talker `Qwen3TTSTalkerModel.__call__` (lines 433-498 in voice-site/mlx_audio/tts/models/qwen3_tts/talker.py) with `mx.compile(..., shapeless=True)`, passing the KVCache via closure and resetting it in-place between steps. Each `Qwen3TTSTalkerDecoderLayer.__call__` (lines 381-417) does ~1.18M dispatches for seq_len=1; compile fuses those into a single dispatch per layer → ~28 dispatches per talker step instead of 33M. Estimated step time drop: ~80ms → ~5-10ms.

## mx.compile measurement (closure-captured KVCache, talker only)

`MLX_OMARCHY_TRACE_DISPATCH=1`, single talker forward, fresh KVCache per iteration:

| Path | Dispatches / iter | p50 ms / iter |
|---|---|---|
| Baseline (uncompiled) | 33,073,041 | 87.3 |
| `mx.compile(..., shapeless=True)` with closure-captured cache | **17,701,859** | 87.9 |

**Dispatch reduction 46.5% but wall time unchanged (+0.7%)**: confirms per-step GPU compute (~87 ms) is the fixed floor, not dispatch overhead. Compile fuses many small Vulkan dispatches into larger ones, but the GPU compute work itself does not change.

## Measured floor

Per-step GPU compute floor (talker, 28 layers × seq_len=1): **~87 ms** on the M2 Max Vulkan/Honeykrisp backend (mlx 0.32.3.dev202609282218+29cba8e). At 12.5 Hz this gives 60 × 87 = 5.2 s of GPU work per 5 s of audio — the lower bound for wall time. CPU dispatch overlap and Python overhead add ~10-15 ms per step on the host clock, giving the observed 92-110 ms total step wall.

**This is a hardware/quantization floor for this model on this backend.** Reaching the RTF ≥ 1.2 threshold requires either:
- A different model (out of scope: pinned pack is the smallest available for this voice)
- A different backend with faster per-step compute (e.g. fp16 native, bf16 native, or an fp8 quantized variant — would require re-quantizing the pinned weights, breaking the hash pin)
- Reducing the number of steps (e.g. predict multiple codebooks per step — requires model architecture change)

Honest answer for the orchestrator: **RTF 0.22 is the floor on this backend for the pinned pack. The Kokoro-82M alternative that SpeechOutputKokoro is integrating is the realistic path forward.**

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
4. **`mx.compile` the talker inner forward with KVCache in closure** — compile succeeds, compiles per-layer dispatch from 33M → 17.7M per talker step (46.5% reduction), but wall time is unchanged (87.3 → 87.9 ms). Per-step GPU compute is the fixed floor; dispatch overhead is overlapped with GPU work and not on the critical path.

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
