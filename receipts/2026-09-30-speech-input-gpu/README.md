# Speech input on the GPU: receipt (in progress)

Status: **not qualified.** Re-measurement is running on the M2 (T6021). This file is replaced
with the measured tables when the run lands.

## Withdrawn results

The tables this file carried at commits `36551e970` and `68ae25089` are withdrawn:

- WER was the mean of per-clip WER, not total word edits over total reference words.
- Numbers were not normalized, while the Midlands references contain digits.
- The 35 Midlands rows carried no subset label and were scored under `?`.
- The 10 dB and 0 dB mixes used synthetic pink noise, not a pinned public noise file.
- The "CPU positive control" patched `mx.eval` in the parent process. The model runs in the
  worker subprocess, so a zero count there proves nothing about the recognition path.

`eval-results.json`, `eval-summary.log`, `latency-n30.log` and `cpu-control.log` are kept as
the raw record of that run and must not be cited as qualification.

## Defects fixed since

- Every transcript was shifted by one request: the worker's boot "ready" frame was read as
  the first response.
- A `(1, N)` input made Parakeet-TDT on mlx-audio 0.5.6 return `""`; the worker sends 1-D.
- The parent resampled before the worker, which would resample twice on the ANE path. The
  GPU worker now resamples on the mlx device (Kaiser polyphase).
- Every request that carried a cancel event waited 6 s for the watcher thread. The server
  always passes one.
- A cancel returned the transcript if generation finished within the grace window. A cancel
  now kills the worker's process group and confirms it is gone.
- The worker resolved the model through the Hugging Face hub (a network lookup). It now loads
  the pinned directory with `HF_HUB_OFFLINE=1`.
- The model was a manual download. Voice setup now downloads it approve-first into
  `<home>/voice/parakeet-tdt-0.6b-v3/` and checks every file's sha256.
- The acceptance receipt was not tied to the runtime. It now names the mlx backend identity
  and the model hash, and refuses unless every frozen threshold passes.
- UI: the 30 s auto-stop and Escape-cancel left the button on "Listening…"; the 30 s message
  was overwritten in the same tick; a denied microphone showed the raw browser error.
