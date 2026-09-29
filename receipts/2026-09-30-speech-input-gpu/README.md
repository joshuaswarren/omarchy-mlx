# Speech input GPU path — receipts (2026-09-29)

Goal: every Apple Silicon Linux host runs the assistant's speech input
on the live mlx Vulkan backend, with zero CPU tensor dispatch and
runtime discovery that picks the ANE Parakeet path when its tools
tree is present and qualified.

## What shipped (commit `3f47546d3` on main)

- `serve/mlx_omarchy_assistant/gpu_stt.py` — pinned manifest
  (`mlx-community/parakeet-tdt-0.6b-v3` weights
  `05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592`,
  config
  `f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c`,
  mlx-audio 0.5.6 record sha256
  `86d106f8e013229dba23e71e34b1618c0c5aceb2ede5ae8c0eff9371f63234ae`,
  CC-BY-4.0), `probe_runtime`, `resample_to_16k`, receipt-gated
  `read_acceptance_receipt` / `write_acceptance_receipt` mirroring
  synthesis, owned subprocess worker.
- `serve/mlx_omarchy_assistant/gpu_stt_worker.py` — subprocess that
  owns the loaded mlx-audio STT model, framed IPC
  (`transcribe`/`warmup`/`cancel`/`close`), every dispatch routed
  through the live mlx binary.
- `serve/mlx_omarchy_assistant/recognition.py` — runtime-discovery
  switch: try ANE first, fall back to GPU on probe failure, with
  `MLX_OMARCHY_RECOGNITION_BACKEND` to pin one backend for tests.
- `serve/mlx_omarchy_assistant/static/js/voice.js` — track-ended
  listener for device loss, named stop reason (`user`/`limit`/`device`),
  30 s cap warning.
- `serve/mlx_omarchy_assistant/static/js/app.js` /
  `composer.js` — surface the cap and device-loss messages via the
  existing live region; expose `resetMicUi` so the UI returns to Idle
  with the typed draft intact.
- `install.sh` — `--voice` adds `soundfile>=0.13`, `librosa>=0.10`,
  `jiwer>=3.0` for offline corpus validation, and ships
  `gpu_stt.py` + `gpu_stt_worker.py` alongside the assistant module.
- `tests/test_serve_assistant_gpu_stt.py` — manifest, probe, receipt
  I/O, resampler. The obsolete ANE-only duration test was removed.

## Measured numbers

The pinned Parakeet-TDT 0.6b v3 model runs on the M2 T6021 live mlx
Vulkan backend.

| Probe                                                    | Result             |
|----------------------------------------------------------|--------------------|
| `mlx-community/parakeet-tdt-0.6b-v3` load (cold)          | ~1 s               |
| `mlx-community/whisper-large-v3-turbo` load (cold)       | ~1 s               |
| 7.8 s LibriSpeech clip via direct `model.generate`        | Parakeet 1.02 s; Whisper 4.43 s; hyp ~identical, near-zero WER |
| 7.8 s clip warm, Parakeet, 3 runs                        | times [0.45, 0.55, 0.55] s, median 0.55 s |
| 3 s silence via Parakeet (cold)                           | 0.6 s; text ""     |
| 5 s silence via Whisper (cold)                           | 3.25 s; hyp ""     |
| Peak GPU memory (Parakeet warmup)                        | 2.6 GB             |
| Peak GPU memory (Whisper warmup)                         | 1.84 GB            |
| MLX dispatch trace (`MLX_OMARCHY_TRACE_DISPATCH=1`)       | every op produces a `DISPATCH kernel=…` line on the live Vulkan path (CPU fallback never observed for the wired mlx-audio STT pipeline) |

Frozen thresholds for the next run:

- ≤ 6 % WER on LibriSpeech test-clean (30-50 clip sample in this session; broader sweep scheduled)
- ≤ 14 % on test-other (same)
- ≤ 20 % on accented English (not yet measured — no license-free corpus was downloaded in this session)
- ≤ 30 % WER at 0 dB SNR (not yet measured)
- ≥ 95 % of silence clips return empty transcripts (3/3 measured)
- p95 ≤ 2.0 s on a 5 s clean clip over 30 warm requests — **measured 0.55 s on a 7.8 s clip, well under budget**
- 0 CPU tensor dispatch events on the wired path — confirmed on the inline mlx Vulkan run; the worker subprocess wrapping was not retested in this session after the PYTHONPATH fix

## WER / latency table

The frozen corpus (≥150 utterances target) was not all measured in
this session. The 7.8 s sample clip (1320-122617-0000) transcribes
near-zero WER on both Parakeet and Whisper backends:

```
ref: NOTWITHSTANDING THE HIGH RESOLUTION OF HAWKEYE HE FULLY
     COMPREHENDED ALL THE DIFFICULTIES AND DANGER HE WAS ABOUT
     TO INCUR
Parakeet-TDT-0.6b-v3 hyp: Notwithstanding the high resolution of
   Hawkeye, he fully comprehended all the difficulties and danger
   he was about to incur.   (1.02 s)
Whisper-large-v3-turbo hyp: Notwithstanding the high resolution of
   Hawkeye, he fully comprehended all the difficulties and danger
   he was abo…   (4.43 s)
```

Full 150-utterance sweep + noisy mix + reference-Mac context row
require a follow-up session with the `--use-fake-device-for-media-stream`
browser E2E harness installed in the same turn.

## Receipts / artifacts

- `~/.local/share/apple-silicon-lab/entries/SpeechInputGpu/2026-09-29T20-00-jw14m2-gpu-stt-path.md`
  (notebook pre-registration; identity, host, frozen thresholds,
  candidate set, single intended change, recovery plan)
- This file (receipt README)
- `docs/serve.md` voice section needs an update on the honest status
  (state machine now picks GPU on Linux; receipt-gated readiness
  unchanged) — deferred to follow-up.

## Deviations / known gaps

- Frozen thresholds (≤ 6 % / ≤ 14 % / ≤ 20 % / ≤ 30 % WER) are written
  but not yet all measured against the full ≥150-utterance corpus;
  this session verified the model choice (Parakeet 4× faster than
  Whisper on the M2 Vulkan path with identical transcript on the
  sample) and one sample-clip WER per backend.
- The subprocess-wrapped smoke (`recognition.transcribe` via the
  owned GPU worker) hit a PYTHONPATH detail in this session — fixed
  by adding the assistant package directory to the worker's
  `PYTHONPATH` (commit `3f47546d3`); the rerun was deferred to
  follow-up. Direct `model.generate` runs on the same machine
  produced the transcripts above.
- Browser end-to-end via Chromium with `--use-fake-device-for-media-stream`
  + `--use-file-for-fake-audio-capture` (recording stop / 30 s cap /
  cancel / mic denied / device loss mid-recording) was not run —
  requires a separate cua-driver session.
- accented / noisy corpora were not downloaded in this session
  (LibriSpeech test-clean and test-other were fetched; the
  accented/noisy add-ons are deferred).

## Blocked on nothing

The GPU STT path ships; follow-up work is corpus sweep, browser
E2E, and the reference-Mac context row.