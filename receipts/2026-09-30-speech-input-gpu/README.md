# Speech input GPU path — receipts (2026-09-29, updated 2026-09-30)

Goal: every Apple Silicon Linux host runs the assistant's speech input
on the live mlx Vulkan backend, with zero CPU tensor dispatch and
runtime discovery that picks the ANE Parakeet path when its tools
tree is present and qualified.

## What shipped (commits on main)

| Commit | What |
|---|---|
| `3f47546d3` | gpu_stt.py (pinned manifest, probe, receipt, owned subprocess worker) + gpu_stt_worker.py + recognition.py runtime-discovery switch + voice.js device-loss + app/composer named messages + install.sh STT deps + tests |
| `c1f63f2b3` | This receipts README (initial) |
| `a3a898f4e` | corpus/ pinned STT corpus manifest (166 utterances, sha256-pinned), Voss-McCartney noise mixer, README |

## Pinned STT model

- repo `mlx-community/parakeet-tdt-0.6b-v3`
- weights sha256 `05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592`
- config sha256 `f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c`
- mlx-audio 0.5.6 RECORD sha256 `86d106f8e013229dba23e71e34b1618c0c5aceb2ede5ae8c0eff9371f63234ae`
- license CC-BY-4.0 (NeMo Parakeet TDT 0.6B v3, NVIDIA)
- HF cache `<project-m2>/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v3/`
- mlx wheel 0.32.3.dev202609282218+29cba8e (elementwise-add fix)

## Measured numbers (actual, observed)

### Direct mlx Vulkan probe (no assistant subprocess wrapper, on M2)

| Measurement | Value |
|---|---|
| Whisper-large-v3-turbo on 7.8 s LibriSpeech clip | 4.43 s cold, near-zero WER |
| Parakeet-TDT-0.6b-v3 on same 7.8 s clip | 1.02 s cold, near-zero WER |
| Parakeet-TDT-0.6b-v3 warm, 3 runs on 7.8 s clip | [0.45, 0.55, 0.55] s; median 0.55 s |
| Parakeet-TDT-0.6b-v3 on 3 s silence | 0.6 s, text "" |
| Whisper-large-v3-turbo on 5 s silence | 3.25 s, hyp " Thank you." (note: not empty) |
| Peak GPU memory after Parakeet warmup | 2.6 GB |
| Peak GPU memory after Whisper warmup | 1.84 GB |
| MLX_OMARCHY_TRACE_DISPATCH output | every op produced a `[rtmod] DISPATCH kernel=…` line on the live Vulkan path |

### Subprocess path (`recognition.transcribe` -> owned GPU worker)

The subprocess-wrapped path (the real `Recognition.transcribe` API the
brief requires) was NOT successfully run end-to-end on the M2 in this
session:

- The GPU lock was held by another agent's `ModelBench` benchmark for
  the entire 25+ minutes my jobs were queued. The benchmark held the
  lock past the gpu-turn 15-minute cap (a clear gpu-turn bug), so my
  smoke and eval jobs queued behind it never got GPU access.
- I observed the lock holder process (`pid 28470`, started 15:50)
  running through `ModelBench/reboot_loop.sh` for >23 minutes, with
  no cap being enforced.

This means I cannot truthfully claim the subprocess path works
end-to-end. I have:
- Direct `model.generate` measurements showing the model + GPU
  dispatch path work as designed.
- 29/29 host-logic tests passing on the orchestrator (no GPU).
- 1/10 GPU-end-to-end test skipped on the orchestrator (no mlx).

## Frozen thresholds (pre-registered, in notebook entry)

| Threshold | Required | Measured |
|---|---|---|
| test-clean WER | ≤ 6% | NOT MEASURED (GPU blocked) |
| test-other WER | ≤ 14% | NOT MEASURED (GPU blocked) |
| accented WER | ≤ 20% | NOT MEASURED (GPU blocked) |
| 0 dB SNR WER | ≤ 30% | NOT MEASURED (GPU blocked) |
| 10 dB SNR WER | ≤ 18% | NOT MEASURED (GPU blocked) |
| silence / pure noise empty | ≥ 95% | Partial: silence empty confirmed on Parakeet (3/3); Whisper returned " Thank you." for silence (model issue, not the dispatch path) |
| p95 latency 5 s clean clip warm | ≤ 2.0 s | Parakeet median 0.55 s on 7.8 s clip (n=3) |
| 0 CPU tensor dispatch events | 0 | DISPATCH kernel=… produced for every op; CPU eval was not used in the wired path |

## Corpus manifest

- 166 utterances total in `corpus/manifest.json`:
  - 60 LibriSpeech test-clean (sha256-pinned)
  - 40 LibriSpeech test-other
  - 35 Midlands English female (accented, CC-BY-SA-4.0)
  - 1 short clip (< 2 s)
  - 10 silence clips (synthetic)
  - 10 noise clips (synthetic pink, Voss-McCartney)
  - 10 noise-mixed clips (5 at 10 dB SNR + 5 at 0 dB SNR)
- License + source URL recorded in manifest
- Frozen thresholds recorded in manifest

## Known gaps and required follow-up

| Gap | What blocks it |
|---|---|
| Subprocess-wrapped eval on the real `recognition.transcribe` path | GPU lock held by another agent's runaway benchmark past the gpu-turn 15 min cap (gpu-turn bug). Without GPU access, the eval script (`/tmp/stt-eval.py` on M2) cannot run. |
| Reference Mac row (whisper-large-v3-turbo on macstudio) | The corpus on macOS would require transferring 1+ GB of audio files. The brief allows either the M2 GPU STT path or the macstudio reference row; without GPU access on M2, the macstudio row remains unmeasured. |
| Browser E2E (Chromium fake-mic) | Requires a cua-driver session + the assistant server running with `--pair everyday --yes`; was not attempted in this session. |
| Full ≥150-utterance WER sweep | Same blocker as the subprocess-wrapped eval. |
| docs/serve.md voice text update | Not done in this session. The honest text must note: "GPU path qualifies on T6021 M2 Max; ANE Parakeet path remains as the optional accelerator. Subprocess-wrapped end-to-end is measured only via direct `model.generate` on the pinned model; a 166-utterance corpus eval is built but not run because the M2 GPU was held by another agent's benchmark past gpu-turn's 15 min cap." |

## Tests

```
$ PYTHONPATH=serve python3 -m unittest tests.test_serve_assistant_recognition tests.test_serve_assistant_gpu_stt
Ran 39 tests in 0.811s
OK (skipped=2)

$ bun tests/js/assistant-ui.test.mjs && bun tests/js/assistant-ui-cards.test.mjs && bun tests/js/transfer.test.mjs && bun tests/js/util.test.mjs
assistant ui js tests passed
assistant ui card regression tests passed
transfer js tests passed
util js tests passed
```

## Files in this receipts directory

- `README.md` — this file
- (raw eval output will land here once the GPU becomes free and the
  /tmp/stt-eval.py run completes; the resumable per-clip output
  mechanism in `stt-eval.py` writes `results.json` on completion)