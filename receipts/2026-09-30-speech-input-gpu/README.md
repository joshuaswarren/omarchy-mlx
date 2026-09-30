# Speech input on the GPU: receipt, 2026-09-30

Status: **not qualified.** No pair or catalog entry changes here.

On the project M2 every speech-input gate measured here passes on the final code:

- the frozen corpus thresholds;
- zero CPU-stream dispatches in the recognition worker;
- a 30-turn browser run with no empty transcript and p95 1397.8 ms, against a 2 s budget;
- the browser scenarios at 375 and 1440 px.

A real screen reader was not run.

## Runtime

| Item | Value |
|---|---|
| Host | `<project-m2>`, Apple M2 Max (T6021), Linux 7.1.13-ARCH-polltx. Final runs on boot `35df33be…`. The first corpus run (`turn4`) ran on boot `2a4f18f7…` |
| mlx wheel | `0.32.3.dev202609291615+06711ad` (v0.7.6 release wheel), provenance `verified: match`, libmlx sha256 `004d24b6…5b240d9a29`, extension sha256 `0b720eb9…ec1d282621` ([env](raw/turn6-env.txt)) |
| Model | `mlx-community/parakeet-tdt-0.6b-v3` at `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`, `model.safetensors` sha256 `05e01c7f…625464592`, loaded offline from `<home>/voice/parakeet-tdt-0.6b-v3` |
| Stack | mlx-audio 0.5.6 in an owned worker subprocess (`gpu_stt_worker.py`); resampling runs on the mlx device |
| Corpus | `corpus/manifest.json` sha256 `2bcb7527…d7728e0f4`, 192 rows (see [corpus/README.md](../../corpus/README.md)) |

Every GPU run held the M2 GPU lock. The tables say when another agent's GPU process was resident.

## Word error rate (frozen thresholds)

WER is total word edits over total reference words, with one normalizer for every system
(`corpus/score.py`). A clip without a transcript counts every reference word as a deletion.
Parakeet ran through `Recognition.transcribe` over the worker subprocess on the final code
(`turn6`, alone on the GPU). Its WER matches `turn4`, the run before the empty-transcript retry,
on every subset. Whisper is a context row only: large-v3-turbo on the reference Mac (M1 Ultra,
macOS 26.6.2, mlx-whisper 0.4.3, mlx 0.30.6), with a new CLI process per clip.

| Subset | n | Parakeet WER | Threshold | Result | Parakeet p50 / p95 ms (turn6; turn4) | Whisper WER |
|---|---:|---:|---:|---|---|---:|
| test-clean | 60 | 3.42% | ≤ 6% | pass | 475 / 853; 482 / 845 | 2.36% |
| test-other | 40 | 2.92% | ≤ 14% | pass | 451 / 805; 447 / 780 | 3.21% |
| accented (Midlands) | 35 | 4.39% | ≤ 20% | pass | 440 / 553; 431 / 571 | 1.34% |
| babble at 0 dB | 10 | 11.43% | ≤ 30% | pass | 1185 / 2591; 446 / 897 | 8.00% |
| babble at 10 dB | 10 | 0.58% | none | — | 1120 / 2275; 441 / 741 | 2.34% |
| short | 12 | 3.85% | none | — | 673 / 979; 382 / 500 | 1.92% |
| final chunk | 5 | 1.49% | none | — | 1235 / 1765; 449 / 462 | 0.00% |

| Subset | n | Parakeet empty rate | Threshold | Result | Whisper empty rate |
|---|---:|---:|---:|---|---:|
| silence | 10 | 1.00 | ≥ 0.95 | pass | 0.00 (text on every silent clip) |
| pink noise | 10 | 1.00 | ≥ 0.95 | pass | 1.00 |

The corpus ran in manifest order. In `turn6`, the subsets decoded late in the run were 2 to 2.5
times slower than the same clips in `turn4`. No other agent's process was listed at the start of
`turn6`, and the retry does not fire on those clips (all of them returned text). The slowdown was
not attributed.

One test-clean clip, `121-123859-0002`, is 30.04 s long. The recognizer refuses anything over
30 s, and the browser recorder now cuts every upload at exactly 30.0 s. The gate row scores the
product path: the first 30.0 s, scored against the full reference. If the raw 30.04 s submission
counts as all deletions, test-clean is **8.15%**, which fails. Both numbers are in
[scores.json](raw/scores.json). The scoring rule was recorded before the product-path run.

## Zero CPU tensor dispatch

Method: a gdb breakpoint on the exported `mlx::core::cpu::get_command_encoder(Stream)` inside
the worker process, which every CPU-stream evaluation reaches
([count_cpu.gdb.py](harness/count_cpu.gdb.py)).

| Run | CPU encoder calls | Breakpoint resolved |
|---|---:|---|
| Control, `mx.add` on `mx.gpu` | 0 | yes ([log](raw/control-gpu-gdb.txt)) |
| Control, `mx.add` on `mx.cpu` | 3 | yes ([log](raw/control-cpu-gdb.txt)) |
| Final worker: load, warm-up, 21 requests (16 kHz and 48 kHz; silence and noise included, so the voicing check ran) | **0** | yes ([log](raw/turn6-worker-gdb.txt), [requests](raw/turn6-trace-requests.json)) |

All 21 requests got answers. Before the models-package fix, the worker made **152** CPU-stream
calls ([log](raw/prefix-worker-gdb.txt), old wheel `+29cba8e`). The call chains
([tally](raw/prefix-cpu-dispatch-native-chains.txt)) and Python stacks
([stacks](raw/prefix-cpu-dispatch-pystacks.txt)) put every first hit at
`mlx_audio/stt/models/granite_speech5_ctc/granite_speech5.py:40`. That line builds a float64
filterbank on `mx.cpu` when the models package `__init__` imports every family. The worker now
registers that package without running its `__init__`.

## Empty transcripts (fixed)

Before the fix, each of three 30-turn browser runs had one turn of clear read speech come back
empty. Across the 96 saved uploads, the model returned nothing for 3.

| Evidence | Result |
|---|---|
| Upload 17 on the M2 GPU | `""` 5 of 5 times ([log](raw/empty-upload-gpu-repeats.json)) |
| Stock mlx 0.32.3 on CPU (x86) and NeMo 3.0.0 (`nvidia/parakeet-tdt-0.6b-v3`) on the empty uploads | `""` on both ([mlx](raw/empty-upload-cpu-reference-variants.json), [NeMo](raw/empty-upload-nemo-reference.json)). The GPU and the stock CPU give identical text for upload 17 and its 14 pad/trim/gain variants ([GPU](raw/empty-upload-gpu-variants.json)); all 24 decoder steps are blank ([trace](raw/empty-upload-cpu-reference-decoder-trace.txt)) |
| Whisper large-v3-turbo on upload 17 | "She promised to do this, and she mentioned to me that when for a moment…" |
| NeMo with the retry's padding ([results](raw/retry-nemo-reference-padding.json)) | 0.4 s of padding: still empty 3 of 3. 1 s of padding: text 3 of 3 |

The fix is in `gpu_stt_worker.py` and documented there. When a decode comes back empty and the
clip has at least 0.5 s of voiced frames, the worker decodes it once more with 1 s of fixed
low-level noise (about -80 dBFS) on both ends. A voiced frame is a 20 ms frame above both 0.005
RMS and 12 dB over the clip's 10th-percentile frame. The unit tests are in
`tests/test_serve_assistant_gpu_stt.py` (`EmptyTranscriptRetryTest`).

| Dated row | Result |
|---|---|
| 2026-09-30 before: 96 saved uploads, as-is | 3 of 96 empty ([results](raw/retry-96-uploads-and-long-clips.json)) |
| 2026-09-30 after: same 96 uploads, retry path | **0 of 96 empty**; tiled 10 to 30 s clips and the real 30 s clip give the same word count as before (for example 75 words at 30 s) |
| Voicing gate on the corpus | 0 of 10 silence and 0 of 10 noise clips have 0.5 s of voiced frames, so the retry cannot fire on them. Every speech subset clears the gate |
| Browser before, n=30 per run | 1 empty per run in three runs; p95 1616.7, 1367.5 (another agent's app resident) and 1579.4 ms ([latest](raw/before-retry-latency-browser-latency.json)) |
| **Browser after, n=30** | **0 empty; p50 861.8 ms, p95 1397.8 ms, max 1410.1 ms** ([results](raw/latency-browser-latency.json)); no other agent's process in the [monitor](raw/latency-monitor.txt) |

Two other designs were measured and rejected:

- **Padding every request** with 0.4 s ([long clips](raw/rejected-edge-pad-long-clips.json), [corpus](raw/rejected-edge-pad-corpus-scores.json)). It fixed the 96 uploads but emptied tiled 20, 28 and 30 s clips (47, 68 and 75 words down to 0). It also broke the 30 s browser scenario and raised test-other WER from 2.92% to 4.38%.
- **Dithering the whole clip** ([results](raw/rejected-dither.json)). At 1e-5, 3 of 96 uploads stayed empty; at 1e-4, 11 did.

Level normalization was tested on the padding-every-request path. At 0.1x level the 96 uploads
gave 0 empty, the same as at full level ([results](raw/retry-step1-edge-pad-96-uploads.json)).
It changed no outcome, so it was not added. The retry path was not rerun at 0.1x.

A capture defect found along the way is also fixed. The recorder had asked the browser for noise
suppression, and Chromium also applied gain control. That doubled the captured level (median
2.14x the source; [fidelity](raw/prefix-latency-upload-fidelity.json)). The recorder now requests
unprocessed audio, which is how the corpus WER was measured: 0.90x
([fidelity](raw/before-retry-latency-upload-fidelity.json)).

## Latency and memory

| Measurement | Result | Other GPU work resident |
|---|---|---|
| Browser, stop click to transcript in the composer, 1440 px, 5.0 s recordings, 2 warm-ups + 30 | p50 861.8 ms, p95 1397.8 ms (budget p95 ≤ 2000 ms): **pass** | none |
| HTTP `POST /api/transcribe` with the Everyday pair resident, 2 warm-ups + 30 | final code: p50 442.7 / p95 492.7 ms ([E2E run](raw/e2e-http-latency.json)) and p50 1052.2 / p95 1467.3 ms ([latency run](raw/latency-http-latency.json)); earlier code, four runs: p95 1295.9 to 1507.9 ms ([1](raw/http-latency-e2e-20260930T054943Z.json), [2](raw/http-latency-e2e-latency-20260930T055241Z.json), [3](raw/http-latency-e2e-latency-20260930T061333Z.json)) | none |
| Direct `Recognition.transcribe`, 30 warm requests on a 5.075 s clip | final code p50 817.4 / p95 1167.4 ms ([turn6](raw/turn6-latency.json)); before the retry p50 424.7 / p95 556.4 ms ([turn4](raw/turn4-latency.json)). The retry did not fire (the clip returns text), so this 2x gap is run-to-run variance and was not attributed | none |
| Cold first call (worker spawn, model load, warm-up, first request) | 2982.8 ms (turn6), 2883.8 ms (turn4). The first dictation after setup pays it | none |
| `GET /api/status`, 30 requests ([probe](raw/route-probe-status-latency.json)) | p50 77.2 ms, p95 255.3 ms. The transcribe route calls the pair manager's full `status()` on every request | none |

Memory is system MemAvailable sampled every 0.25 s
([summary](raw/mem-probe-20260930T091655Z-summary.json),
[samples](raw/mem-probe-20260930T091655Z-memavail.txt),
[phases](raw/mem-probe-20260930T091655Z-marks.txt)). RSS is not used. The earlier probe gave the
same figures within 160 MiB ([summary](raw/mem-probe-20260930T083759Z-summary.json)), but
another agent's idle 27B app was resident during it.

| Phase (alone on the GPU) | MemAvailable drop |
|---|---:|
| Recognizer alone, peak over 30 transcriptions | 3192 MiB |
| Everyday pair resident (chat + decision workers) | 2091 MiB |
| Recognizer loaded beside the pair, resident | 3182 MiB |
| Recognizer beside the pair, peak during 30 transcriptions | 3195 MiB |
| Pair + recognizer peak, from the idle baseline | **5218 MiB** |
| After stopping everything | back to baseline (+122 MiB available) |

The worker's own mlx peak was 2,707,193,856 bytes (2.52 GiB). The rest of the recognizer's
3.1 GiB is the Python process and its libraries. PairGates owns the pair's peak measurements.

## Recognition smoke (subprocess path, final code)

From [turn6-smoke.json](raw/turn6-smoke.json):

- The first call took 1663.6 ms and matched the reference. A 48 kHz input took 452.9 ms and matched. 3 s of silence returned `""` in 353.2 ms. Exactly 30 s returned 75 words in 1227.3 ms.
- Named refusals, all before any model work: 30.10 s of audio, a 51,380,228-byte payload, NaN samples, rate 0, rate 500,000, and 8-bit PCM.
- Cancel mid-request raised in 702.1 ms, with the worker and its process group confirmed gone. The next call started a new worker and matched the reference in 1666.9 ms.

## Browser end-to-end (Chromium 153 headless, fake microphone, final code)

Every run launched its own browser. Screenshots are in [screenshots/](screenshots/). The
scenario results are in [e2e-results.json](raw/e2e-results.json) and the denial results in
[e2e-denied-results.json](raw/e2e-denied-results.json). Each scenario first types a draft.

| Scenario | 1440 px | 375 px |
|---|---|---|
| User stop | The transcript was appended after the draft; 1 transcribe request | same |
| 30 s limit | "About to hit the 30 second limit.", then "Recording stopped at the 30 second limit. Transcribing…"; the transcript was appended; 1 request | same |
| Escape | "Recording cancelled."; 0 requests; draft kept | same |
| Device loss (track ended) | "Recording stopped: microphone disconnected."; 0 requests; draft kept | same |
| Permission denied (browser permission set to denied, prompts denied) | "Microphone permission denied. Allow microphone access for this page, then try again."; the button returned to Microphone; 0 requests; draft kept | same |

These messages now also appear as visible text under the composer, not only in the
screen-reader live region. The two voice status labels no longer run together.

## Not run

- A real screen reader. This session had no desktop and no screen reader to use.
- Speech input on M1-class chips. The standing M1 battery belongs to its own workstream.

## Excluded runs

- **Stale browser.** The E2E and latency runs from `e2e-20260930T034523Z` through `e2e-latency-20260930T042424Z` drove a Chromium left over from a crashed run on a fixed port. That browser was started with the fake-UI flag, and it kept recording tabs open outside the GPU lock.
- **Superseded corpus runs.** `turn1`, `turn2` (152 CPU calls), `turn3` (old wheel) and `turn5` (the rejected padding design) are superseded by `turn6`.
- **Withdrawn tables.** The tables at commits `36551e970` and `68ae25089` are withdrawn. They used a per-clip mean WER without number normalization, unlabelled accented rows, synthetic noise, and a CPU counter in the parent process. Their raw files are in git history.

## Reproduce

The scripts are in [harness/](harness/):

- `turn6.sh`: smoke, corpus, over-limit clip, traced worker.
- `controls.sh`: CPU and GPU controls.
- `e2e.sh`, `e2e_denied.sh`, `e2e_latency.sh`: browser runs.
- `mem_probe.sh`: memory.
- `probe_*.py`, `ref_*.py`, `nemo_*.py`, `step1_uploads.py`: the empty-transcript work.

Published copies replace host paths with `<home>` and `<app-home>` (the test app home) and host
names with `<project-m2>` and `<reference-mac>`, and they redact one-use launch tokens. Each GPU
script ran as its own ticket under the M2 GPU lock. Scoring:
`python3 corpus/score.py corpus/manifest.json clips.jsonl+overlimit.jsonl whisper.json`.
