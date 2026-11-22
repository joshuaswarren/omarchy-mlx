# silero VAD stage for the voice loop (replaces the energy gate on request)

Follow-up to the V25 receipt (the earlier mini50 measurement of the
energy gate, kept on its own branch): the shipped energy gate scored
F1 0.667 / specificity 0.00 on FluidAudio's mini50 — every MUSAN noise clip
contains a chunk louder than any useful floor. This change adds a real VAD
stage and keeps the energy gate only as an opt-in pre-filter.

## License check (done first, per Main)

- **silero-vad v5: MIT** — verified via the GitHub API (`license.spdx_id`
  = MIT on snakers4/silero-vad). Chosen.
- **FSMN-VAD: excluded.** FluidInference's card carries
  `license: other / license_name: fsmn-vad-upstream` — not a clear MIT
  grant, so it was not considered further.

## What changed

- `serve/mlx_omarchy_assistant/vad.py` — `SileroVad`: silero-vad v5 ONNX on
  CPU (onnxruntime, single-threaded), 512-sample (32 ms) frames at 16 kHz,
  internal state tensor, threshold + reset; `prepare()` stages the model
  under `home/vad/` against a pinned SHA-256
  (`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`,
  silero-vad master `src/silero_vad/data/silero_vad.onnx`), mirroring the
  wake-model flow — the `--vad silero` flag is the download approval.
- `voice_loop.Endpointer` gains `vad=` and `prefilter_floor=`: the speech
  decision is the VAD when one is set; the energy gate survives only as a
  pre-filter that rejects silent chunks before the model (it never
  overrides the VAD).
- **Default (Main, post-review): the voice loop runs silero when the model
  is prepared, falling back to the energy gate with a one-line notice when
  it is absent.** `install.sh` stages the pinned model into
  `$PREFIX/vad/silero_vad.onnx` (verified against the same SHA-256; on a
  failed fetch it says so and the loop degrades to the gate). Unset
  `--vad` = silero-if-prepared; `--vad energy` opts out; explicit
  `--vad silero` allows the download. The prepared/absent/fallback paths
  are covered by `DefaultVadTests` plus an end-to-end check with the real
  pinned model (absent → notice + gate; staged → SileroVad, real session).

## Red tests first

`tests/test_assistant_vad.py` failed on collection before `vad.py` existed,
then: framing (640-sample mic chunks buffer into exactly 512-sample frames),
state propagation, threshold crossing, reset clears buffer+state, the
SHA-256 pin refuses wrong bytes, `Endpointer` uses the VAD for endpointing
(quiet-but-speech chunks hold the turn — the energy gate's blind spot), the
pre-filter skips the model for silent chunks (measured 0 VAD calls), the
no-VAD path keeps the energy gate, and the default-flip set (parser default
unset = silero-if-prepared, prepared model builds SileroVad, absent model
falls back with exactly one notice). **12 passed**; neighboring suites
(voice loop loopback, install contract, wake eval) green; `vad.py` added to
install.sh's staged file list, which the install contract test enforces. An
end-to-end check with the real pinned model confirms both default paths.

## mini50 table (the V25 protocol, real model, desktop CPU)

Same set, labels and per-clip rule as the V25 receipt (clip = speech iff any
frame clears the threshold; 25 speech + 25 noise clips, 2,445.9 s):
`summary.json` carries per-clip decisions.

| threshold | accuracy | precision | recall | F1 | specificity |
|---|---|---|---|---|---|
| 0.3 | 0.98 | 0.962 | 1.00 | **0.980** | 0.96 |
| 0.5 (upstream default) | 0.98 | 0.962 | 1.00 | 0.980 | 0.96 |
| 0.7 | 0.98 | 0.962 | 1.00 | 0.980 | 0.96 |

Threshold is not a lever on this set — every speech clip clears all three
thresholds and the same single noise clip trips all three.

For scale on the same protocol: the shipped energy gate F1 0.667
(specificity 0.00 at every floor); FluidAudio's published silero baseline
84.7 F1, their FSMN-VAD 98.0 F1 — the corrected wrapper matches the FSMN
row almost exactly (98.0 vs 98.0).

Review-fix note: the first mini50 table (F1 0.939) was measured with two
input bugs — raw int16 values instead of [-1, 1] floats, and silero v5's
64-sample cross-frame context missing (its own ONNXWrapper prepends the
previous frame's tail 64 samples; without it the mel window misaligns and
scores collapse to ~0.001 on real speech — both measured). A second pass
found the carried context was being scaled a second time (0.5 -> 1.5e-5);
the samples are now scaled exactly once and the context keeps its scale,
which a two-frame continuity test pins against upstream's wrapper
semantics. The numbers above reflect the corrected wrapper.

## Latency (Apple M1 Max lab host, CPU, single-thread onnxruntime)

- **0.106 ms per 32 ms frame** (500 timed frames after warm-up, the
  576-sample input the wrapper feeds) =
  **0.099 ms per 30 ms of audio**.
- Pre-filter potential: 27.9 % of mini50's 40 ms-chunk frames are silent
  under the RMS-300 gate (61,173 frames measured), so enabling
  `--vad-prefilter-floor 300` skips ~28 % of model calls. At 0.1 ms/frame
  that is ~0.03 ms per 30 ms of audio — kept as an opt-in flag, not a
  default: the saving is noise next to the loop's other costs, and silence
  frames score ~0.0006 through silero, so the flag changes no decision.

## Receipt files

- `summary.json` — per-clip decisions per threshold, frame/prefilter stats.
- `score_mini50.py` — the scorer; the decision code under test is the
  shipped `SileroVad` itself (self-tested in `tests/test_assistant_vad.py`).
- `vad_latency.sh` — the latency probe (`VAD_PY=` picks the python).
