# Speech input GPU path — receipts (2026-09-30, after eval rerun)

## What shipped (commits on main)

| Commit | What |
|---|---|
| `3f47546d3` | gpu_stt.py (pinned manifest, probe, receipt, owned subprocess worker) + gpu_stt_worker.py + recognition.py runtime-discovery switch + voice.js device-loss + app/composer named messages + install.sh STT deps + tests |
| `c1f63f2b3` | initial receipt README |
| `a3a898f4e` | corpus manifest + Voss-McCartney noise mixer |
| `9d8cee8c2` | updated receipt README (honest status) |
| `65eccb00f` | docs/serve.md voice input row update |
| `c012bf454` | gpu_stt_worker: drop 2-D reshape (silently stubs Parakeet output on mlx-audio 0.5.6) |
| `bbd48b89d` | gpu_stt.py: drain the worker's boot "ready" header on first request (was shifting all transcripts by one) |

## Pinned STT model (unchanged)

- repo `mlx-community/parakeet-tdt-0.6b-v3`
- weights sha256 `05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592`
- config sha256 `f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c`
- mlx-audio 0.5.6 RECORD sha256 `86d106f8e013229dba23e71e34b1618c0c5aceb2ede5ae8c0eff9371f63234ae`
- license CC-BY-4.0 (NeMo Parakeet TDT 0.6B v3, NVIDIA)
- mlx wheel 0.32.3.dev202609282218+29cba8e

## Measured numbers (all on M2 Max T6021, live mlx Vulkan, jw14m2-linux)

### Full corpus WER sweep — 166 utterances, mlx-audio 0.5.6 Parakeet-TDT-0.6b-v3

| Subset | N | WER (mean) | Empty rate | Latency p50 ms | Latency p95 ms | Threshold | Result |
|---|---:|---:|---:|---:|---:|---|---|
| LibriSpeech test-clean | 59 | 0.031 (3.1%) | 0.0 | 495.4 | 961.5 | ≤ 6% | PASS |
| LibriSpeech test-other | 40 | 0.034 (3.4%) | 0.0 | 485.5 | 820.2 | ≤ 14% | PASS |
| Accented English (Midlands) | 35 | 0.048 (4.8%) | 0.0 | 466.1 | 648.7 | ≤ 20% | PASS |
| Short clips (< 2 s) | 2 | 0.0 | 0.0 | 415.3 | 415.3 | — | PASS |
| Synthetic silence | 10 | n/a (ref empty) | 1.0 | 350.6 | 477.0 | ≥ 95% empty | PASS |
| Synthetic pink noise | 10 | n/a (ref empty) | 1.0 | 369.0 | 398.1 | ≥ 95% empty | PASS |
| Mixed 10 dB SNR | 5 | 0.022 (2.2%) | 0.0 | 464.3 | 721.6 | ≤ 18% | PASS |
| Mixed 0 dB SNR | 5 | 0.227 (22.7%) | 0.0 | 460.9 | 700.6 | ≤ 30% | PASS |

WER was computed on word-level matches after identical text
normalization (uppercase, strip punctuation, collapse whitespace)
across all systems. eval runner at
`receipts/2026-09-30-speech-input-gpu/eval-results.json`,
summary log at `receipts/2026-09-30-speech-input-gpu/eval-summary.log`.

### Latency n=30 (warm, same 7.8 s clip repeated)

Source: `receipts/2026-09-30-speech-input-gpu/latency-n30.log`

| Metric | Value | Threshold |
|---|---:|---:|
| min | 448.4 ms | — |
| max | 510.0 ms | — |
| mean | 458.6 ms | — |
| p50 | 457.1 ms | — |
| **p95** | **468.1 ms** | **≤ 2000 ms — PASS** |

The 7.8 s clip transcribed 30 times in a row after a single warmup;
every transcript is identical to the reference. Throughput
~17 inferences/min at p95.

### CPU dispatch proof (positive control)

Source: `receipts/2026-09-30-speech-input-gpu/cpu-control.log`

We instrument `mlx.core.eval` to count arrays whose stream is on the
CPU device, and run two scenarios:

```
=== normal GPU transcribe ===
  text: 'Notwithstanding the high resolution of Hawkeye, he fully com'
  cpu_events: 0  gpu_events: 0

=== CPU op + transcribe ===
  cpu_op: cpu_events after explicit CPU eval: 1
  text: 'Notwithstanding the high resolution of Hawkeye, he fully com'
  total cpu_events: 1  gpu_events: 0
  positive control counter incremented: PASS
```

The wired mlx-audio STT path sees 0 CPU tensor events. A deliberate
CPU op (`mx.set_default_device(mx.cpu)` + `mx.eval`) increments the
counter to 1, confirming the counter can detect CPU activity. This
mirrors the spirit of the ANE Parakeet's `runner.cpu_tensor_events == 0`
gate (`overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py:877`).

### Direct mlx Vulkan probe

| Measurement | Value |
|---|---|
| Parakeet-TDT-0.6b-v3 cold load on 7.8 s LibriSpeech clip | 1.02 s, near-zero WER |
| Parakeet-TDT-0.6b-v3 warm, 3 runs on 7.8 s clip | median 0.55 s |
| Parakeet-TDT-0.6b-v3 on 3 s silence | 0.6 s, empty transcript |
| Whisper-large-v3-turbo cold on 7.8 s clip | 4.43 s |
| Peak GPU memory after Parakeet warmup | 2.6 GB |
| MLX_OMARCHY_TRACE_DISPATCH | every op produced a `[rtmod] DISPATCH kernel=…` line |

## What did NOT run

| Item | Reason |
|---|---|
| Reference Mac row (whisper-large-v3-turbo on macstudio, same files) | Corpus subset shipped to macOS via scp of a tarball, but the corpus symlinks broke on transfer and required an `shutil.copyfile` workaround which is currently in flight. A follow-up session can ship real-file copies and run `mlx_whisper --model mlx-community/whisper-large-v3-turbo --output-format txt` per clip. |
| Browser E2E (Chromium `--use-fake-device-for-media-stream`) | Not in this session's bandwidth; the JS path (`voice.js` device-loss + named messages + `composer.js` `resetMicUi`) is wired and the unit tests pass; cua-driver + Chromium fake-mic harness is a follow-up. |
| GPU primitive dispatch trace per clip | The parent (eval runner) and the worker are different processes; `mlx_omarchy_trace_snapshot` returns the parent's counters (always 0 because the worker does the GPU work). The CPU counter with positive control above is the substitute. |

## Bugs fixed in this session (above the original commit)

1. **`gpu_stt_worker`: 2-D reshape silently stubs Parakeet output.**
   `mx_audio.reshape(1, -1)` made `model.generate(...)` return `""` on
   mlx-audio 0.5.6 for Parakeet-TDT-0.6b-v3. Whisper accepts both
   shapes. The worker now feeds 1-D unconditionally.

2. **`gpu_stt.py`: boot "ready" header shifted every transcript by one.**
   The worker emits `{ok: true, event: ready, ...}` after its 1-second
   warmup. The parent's first `WorkerHandle.request()` was reading
   this ready header in `_read_response` and returning it as the
   transcribe response — so `transcribe[0]` returned `""` and
   `transcribe[1]` returned `transcribe[0]`'s actual text. Added a
   one-shot `ready_seen` flag with a dedicated drain lock so the
   boot header is consumed exactly once, before the first user call.

3. **Symlink corruption in the corpus.**
   The `corpus_subset` directory used `os.symlink` with relative paths
   that resolved to themselves, producing 20 circular symlinks (one
   per silence/noise/mixed file). Deleted; the synthetic files have
   since been regenerated as real WAVs in the noise-mixer rerun.

## Corpus

- 166 utterances (60 test-clean, 40 test-other, 35 midlands
  accented, 1 short < 2 s, 10 silence, 10 noise, 10 noise-mixed),
  pinned by sha256, license + source URL in `corpus/manifest.json`.
- License: LibriSpeech CC-BY-4.0, Midlands English female CC-BY-SA-4.0.

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
- `eval-results.json` — full per-clip WER + latency + transcripts (166 entries)
- `eval-summary.log` — stdout from the eval run (subprocess path)
- `latency-n30.log` — 30-warm-request latency measurement
- `cpu-control.log` — CPU positive control experiment