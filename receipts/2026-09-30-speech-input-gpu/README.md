# Speech input on the GPU: receipt, 2026-09-30

Status: **not qualified.** No pair or catalog entry changes here. On the project M2, every
frozen corpus threshold passes, the recognition worker makes zero CPU-stream dispatches, and the
browser meets the 2 s transcript budget. One pre-registered browser criterion fails: in each
30-turn browser run, one turn of clear speech returned no transcript. NeMo's own implementation
of the model does the same on those recordings (see "Empty transcripts"). A real screen reader
was not run.

## Runtime

| Item | Value |
|---|---|
| Host | `<project-m2>`, Apple M2 Max (T6021), Linux 7.1.13-ARCH-polltx. Corpus, trace and controls ran on boot `2a4f18f7…`; browser runs on boot `35df33be…` (the host rebooted twice in between) |
| mlx wheel | `0.32.3.dev202609291615+06711ad` (v0.7.6 release wheel), provenance `verified: match`, libmlx sha256 `004d24b6…5b240d9a29`, extension sha256 `0b720eb9…ec1d282621` ([env](raw/turn4-env.txt)) |
| Model | `mlx-community/parakeet-tdt-0.6b-v3` at `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`, `model.safetensors` sha256 `05e01c7f…625464592`, loaded offline from `<home>/voice/parakeet-tdt-0.6b-v3` |
| Stack | mlx-audio 0.5.6 in an owned worker subprocess (`gpu_stt_worker.py`); resampling runs on the mlx device |
| Corpus | `corpus/manifest.json` sha256 `2bcb7527…d7728e0f4`, 192 rows (see [corpus/README.md](../../corpus/README.md)) |

Every GPU run held the M2 GPU lock. Each table says whether another agent's GPU process was
resident during the run.

## Word error rate (frozen thresholds)

WER is total word edits over total reference words. One normalizer is used for every system
(`corpus/score.py`). A clip without a transcript counts every reference word as a deletion.
Parakeet ran through `Recognition.transcribe` over the worker subprocess, alone on the GPU.
Whisper is a context row only: large-v3-turbo on the reference Mac (M1 Ultra, macOS 26.6.2,
mlx-whisper 0.4.3, mlx 0.30.6). It started a new CLI process for every clip, so its latency does
not compare.

| Subset | n | Parakeet WER | Threshold | Result | Parakeet p50 / p95 ms | Whisper WER |
|---|---:|---:|---:|---|---|---:|
| test-clean | 60 | 3.42% | ≤ 6% | pass | 482 / 845 | 2.36% |
| test-other | 40 | 2.92% | ≤ 14% | pass | 447 / 780 | 3.21% |
| accented (Midlands) | 35 | 4.39% | ≤ 20% | pass | 431 / 571 | 1.34% |
| babble at 0 dB | 10 | 11.43% | ≤ 30% | pass | 446 / 897 | 8.00% |
| babble at 10 dB | 10 | 0.58% | none | — | 441 / 741 | 2.34% |
| short | 12 | 3.85% | none | — | 382 / 500 | 1.92% |
| final chunk | 5 | 1.49% | none | — | 449 / 462 | 0.00% |

| Subset | n | Parakeet empty rate | Threshold | Result | Whisper empty rate |
|---|---:|---:|---:|---|---:|
| silence | 10 | 1.00 | ≥ 0.95 | pass | 0.00 (text on every silent clip) |
| pink noise | 10 | 1.00 | ≥ 0.95 | pass | 1.00 |

One test-clean clip, `121-123859-0002`, is 30.04 s long. The recognizer refuses anything over
30 s, so the raw submission failed. That failure also showed that the browser recorder could
upload slightly more than 30.0 s. The recorder now cuts the upload at exactly 30.0 s. The gate
row above scores the product path: the first 30.0 s went through `Recognition.transcribe` and
was scored against the full reference. If the raw submission counts as all deletions,
test-clean is **8.15%**, which fails. Both numbers are in [scores.json](raw/scores.json)
(`parakeet_gpu`, `parakeet_gpu_raw_submission`). The scoring rule was recorded before the
product-path run.

## Zero CPU tensor dispatch

Method: a gdb breakpoint on the exported `mlx::core::cpu::get_command_encoder(Stream)` inside
the worker process, which every CPU-stream evaluation reaches. The script is
[count_cpu.gdb.py](harness/count_cpu.gdb.py). The controls ran on the same wheel.

| Run | CPU encoder calls | Breakpoint resolved |
|---|---:|---|
| Control, `mx.add` on `mx.gpu` | 0 | yes ([log](raw/control-gpu-gdb.txt)) |
| Control, `mx.add` on `mx.cpu` | 3 | yes ([log](raw/control-cpu-gdb.txt)) |
| Worker: load, warm-up, 21 requests (16 kHz and 48 kHz inputs) | **0** | yes ([log](raw/turn4-worker-gdb.txt), [requests](raw/turn4-trace-requests.json)) |

All 21 requests got answers. The backend trace of the same run logged 55,952 Vulkan dispatch
lines.

Before the fix, the same worker made **152** CPU-stream calls ([log](raw/prefix-worker-gdb.txt),
old wheel `+29cba8e`). The native call chains
([tally](raw/prefix-cpu-dispatch-native-chains.txt)) and the Python stacks
([stacks](raw/prefix-cpu-dispatch-pystacks.txt)) put the first hit of every chain at
`mlx_audio/stt/models/granite_speech5_ctc/granite_speech5.py:40`. That line evaluates a float64
mel filterbank on `mx.cpu`. It runs because `load_model` imports the models package, and the
package `__init__` imports every model family. The worker now registers that package without
running its `__init__`, so only Parakeet's modules load.

## Latency and memory

The design gate is a p95 editable transcript within 2 s after recording stops, for a clean
five-second input, measured over 30 warm turns after two warm-ups with the pair and voice
present.

| Measurement | Result | Other GPU work resident |
|---|---|---|
| **Browser, stop click to transcript in the composer**, 1440 px, 5.0 s recordings, 2 warm-ups + 30, timed in the page ([results](raw/latency-browser-latency.json)) | p50 990.2 ms, p95 1579.4 ms, max 1768.1 ms; **1 of 30 turns empty** | none during the browser section ([monitor](raw/latency-monitor.txt)) |
| Same, before the capture fix ([results](raw/prefix-latency-browser-latency.json)) | p50 1046.7 ms, p95 1616.7 ms; 1 of 30 empty | none |
| Same, after the fix, another agent's app resident ([results](raw/notalone-latency-browser-latency.json)) | p50 939.8 ms, p95 1367.5 ms; 1 of 30 empty | yes |
| HTTP `POST /api/transcribe` with the Everyday pair resident, 2 warm-ups + 30, upload included | p95 1295.9 / 1440.4 / 1440.4 / 1507.9 ms and p50 961 to 1077 ms in four runs ([1](raw/http-latency-e2e-20260930T054943Z.json), [2](raw/http-latency-e2e-latency-20260930T055241Z.json), [3](raw/http-latency-e2e-latency-20260930T061333Z.json), [4](raw/e2e-http-latency.json)) | none |
| Direct `Recognition.transcribe`, no app, 30 warm requests on a 5.075 s clip ([latency](raw/turn4-latency.json)) | p50 424.7 ms, p95 556.4 ms, max 559.4 ms | none |
| `GET /api/status`, 30 requests ([probe](raw/route-probe-status-latency.json)) | p50 77.2 ms, p95 255.3 ms | none |
| Cold first call (worker spawn, model load, warm-up, first request) | 2883.8 ms. The first dictation after setup pays it: 2796 ms stop-to-transcript at 1440 px in [the E2E run](raw/e2e-results.json) | none |
| Worker mlx peak memory | 2,707,128,320 bytes (2.52 GiB); worker VmHWM 143,888 kB | none |

The time budget passes in every browser run. The transcribe route calls the pair manager's full
`status()` on every request, which costs about 77 ms at p50. That does not explain the rest of
the gap between the HTTP path (p50 about 1 s) and the direct path (p50 425 ms). The rest of the
gap was not attributed.

## Empty transcripts

In each browser latency run, one 5 s turn of clear read speech came back empty, and the page
showed "No speech detected." The page uploaded audio in every case, and the uploads are kept:
[pre-fix upload 17](raw/prefix-latency-empty-upload-17.wav) and
[post-fix upload 5](raw/postfix-empty-upload-05.wav).

| Check | Upload 17 (pre-fix) | Upload 5 (post-fix) |
|---|---|---|
| M2 GPU, this stack | `""` 5 of 5 times ([log](raw/empty-upload-gpu-repeats.json)) | `""` in the run |
| Stock mlx 0.32.3 on CPU (x86), mlx-audio 0.5.6, same pinned weights | `""` ([variants](raw/empty-upload-cpu-reference-variants.json)); all 24 decoder steps blank ([trace](raw/empty-upload-cpu-reference-decoder-trace.txt)) | `""` ([variants](raw/empty-upload-05-cpu-reference-variants.json)) |
| NeMo 3.0.0, `nvidia/parakeet-tdt-0.6b-v3`, CPU ([results](raw/empty-upload-nemo-reference.json)) | `""` | `""` |
| Whisper large-v3-turbo | "She promised to do this, and she mentioned to me that when for a moment…" | not run |

For upload 17, the GPU and the stock CPU reference give identical text for the input and for all
14 variants: 13 pads and trims plus half gain ([GPU](raw/empty-upload-gpu-variants.json)). Adding 10 ms of silence at the end
recovers the text on both. The empty output therefore comes from the model, not from the Vulkan
backend or the mlx-audio port.

The first run also exposed a capture defect. The recorder asked the browser for noise
suppression, and Chromium also applies gain control and echo cancellation by default. The
captured level was 2.14 times the source level (median; [fidelity](raw/prefix-latency-upload-fidelity.json)).
The clean source window behind upload 17 transcribes correctly
([window](raw/empty-upload-clean-source-window.json)). The recorder now requests unprocessed
audio, which matches how the corpus WER was measured. The level ratio is now 0.90
([fidelity](raw/latency-upload-fidelity.json)). The empty-turn rate did not change: 1 of 30
before the fix and 1 of 30 after.

## Recognition smoke (subprocess path)

From [turn4-smoke.json](raw/turn4-smoke.json):

- The first call took 1592.3 ms and matched the reference. A 48 kHz input took 435.1 ms and matched. 3 s of silence returned `""` in 342.4 ms. Exactly 30 s returned 75 words in 1120.2 ms.
- Named refusals, all before any model work: 30.10 s of audio, a 51,380,228-byte payload, NaN samples, rate 0, rate 500,000, and 8-bit PCM.
- Cancel mid-request raised in 651.2 ms, with the worker and its process group confirmed gone. The next call started a new worker and matched the reference in 1578.2 ms.

## Browser end-to-end (Chromium 153 headless, fake microphone)

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

The screenshots showed that these messages existed only in the screen-reader live region.
Sighted users saw nothing. They now also appear as text under the composer, and the two voice
status labels no longer run together.

## Not run

- A real screen reader. This session had no desktop and no screen reader to use.
- Speech input on M1-class chips. The standing M1 battery belongs to its own workstream.
- Peak GPU memory of the pair and the recognizer together. Only process RSS was captured with the pair resident.
- A fix for the empty transcripts. The model produces them, and the product reports "No speech detected" instead of inventing text.

## Excluded runs

The earlier E2E and latency runs, `e2e-20260930T034523Z` through `e2e-latency-20260930T042424Z`,
drove a Chromium left over from a crashed run on a fixed debugging port. That browser was
started with the fake-UI flag, so it never denied a permission, and it kept recording tabs open
outside the GPU lock. None of those runs is cited. `turn1`, `turn2` (152 CPU calls) and `turn3`
(old wheel) are superseded by `turn4`. The tables at commits `36551e970` and `68ae25089` are
withdrawn. There, WER was a per-clip mean without number normalization, the accented rows had
no label, the noise was synthetic, and the CPU counter ran in the parent process. Their raw
files are in git history.

## Reproduce

The scripts are in [harness/](harness/):

- `turn4.sh`: smoke, corpus, over-limit clip, traced worker.
- `controls.sh`: the CPU and GPU controls.
- `e2e.sh`, `e2e_denied.sh`, `e2e_latency.sh`: browser runs.
- `probe_*.py`, `ref_*.py`, `nemo_probe.py`: the empty-transcript checks.

Published copies replace host paths with `<home>` and `<app-home>` (the test app home) and
host names with `<project-m2>` and `<reference-mac>`, and they redact one-use launch tokens.
Each GPU script ran as its own ticket under the M2 GPU lock. Scoring:
`python3 corpus/score.py corpus/manifest.json clips.jsonl+overlimit.jsonl whisper.json`.
