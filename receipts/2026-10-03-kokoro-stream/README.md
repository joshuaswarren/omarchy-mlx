# Streamed Kokoro decoding — 2026-10-03

Read-aloud audio now leaves the voice worker while Kokoro's vocoder is
still running. Before this change, `KokoroPipeline` produced nothing until
the whole decoder had run over a whole phoneme chunk. The head/rest text
split (`_first_segment`) bought first-audio time at the cost of a
mid-sentence gap: on the M2 serve path the shipped tree starved playback in
50 of 60 sentence runs. `serve/mlx_omarchy_assistant/kokoro_stream.py`
replaces that split.

## What runs

1. The pipeline's own English G2P and ≤510-phoneme chunking. Each chunk is
   then cut at word boundaries into utterances of about 29 phonemes (~2 s
   of audio), preferring a cut after punctuation. At each such cut, the
   silence the model predicts for the utterance's boundary pad tokens is
   trimmed to one aligned frame (25 ms).
2. Per utterance, mlx_audio 0.5.6 runs its front half, the decoder's
   low-rate stack and the generator's first upsampling stage over the
   whole utterance, exactly as the whole call computes them.
3. The generator's last stage and the inverse STFT run in windows of 24
   aligned frames (600 ms) with 1 frame of context per side, then trimmed.
   Its 24 AdaIN layers use frozen per-voice statistics from
   `kokoro_gen_stats.npz` (121,754 B, sha256 `6e33b3b9…574c1`).
   `scripts/kokoro_gen_stats.py` calibrated them over 47 utterances of a
   24-sentence corpus that is disjoint from the evaluation corpus, per
   voice, pinned to pack revision `a71e4d38…`.
4. An utterance that fits in one window runs the unchanged decoder.
   `MLX_OMARCHY_KOKORO_STREAM=0` restores upstream's whole-call decoding.

## Why this shape

The investigation is in the lab notebook, entries `infer-floor` and
`kokoro-stream-product`.

| Finding | Evidence |
|---|---|
| The decoder is ~85% of a sentence's infer time, and the generator is 98.5% of the decoder (low-rate stack 45 ms, generator 2.9 s on the gate sentence) | M2 stage table; jw16 cost split ×3 |
| Windowing is exact once a window has enough context: 4 frames per side for the whole generator, 1 for the last stage alone | oracle-statistics runs: corr 1.000000, max_abs 0.0 |
| AdaIN normalises over the whole utterance, which a window cannot see. Frozen statistics for the whole generator give corr 0.971 (in-domain) / 0.950 (shipped corpus); a voiced/unvoiced class-mixture prior did worse (0.949–0.965) | jw16 v5–v8, M2 ticket 1 |
| The error lives in stage 0: exact stage-0 statistics give corr 0.994, while an exact noise path or an exact stage 1 does not help (0.972, 0.978) | jw16 v9 locator, 15 sentences |
| Recomputing context for the whole generator costs the real-time margin: stream RTF 1.003 against whole-call 1.32 on the M2, with 1 underrun | M2 ticket 1 |
| Stage 0 must finish before an utterance's first window (~0.29 ms per ms of audio), so whole sentences give 1.2–2.6 s to first audio; ~2 s utterances keep it under 1.5 s | jw16 v10 + cost model |
| Untrimmed, every cut added 536 ms of boundary silence (+16% audio over the corpus); trimming the pad tokens to one frame brings that down to 89 ms per cut | M2 product checks before and after |

## M2 product check (jw14m2, boot `3d3b1e2e`, load 0.01–0.28, PSI cpu avg10 0.00)

This run exercised the product module itself on the 15-sentence evaluation corpus, af_heart:

| Check | Result |
|---|---|
| Windows vs per-utterance whole decode, oracle last-stage statistics | corr 1.000000, max_abs 0.0, identical lengths, 15/15 |
| Shipped statistics vs per-utterance whole decode | corr median 0.9878 (0.9759–0.9925) |
| Same-wheel run-to-run floor, same utterances (seed k vs k+100) | median 0.9903 (0.9876–0.9915); shipped minus floor: median −0.0014, 4/15 at or above, worst −0.015 |
| First audio from text (product entry point) | 991–1374 ms |
| Stream RTF (audio / wall) | 1.184–1.311 |
| Compute-bound playout | 0 underruns on 15/15, minimum slack 344 ms |
| Audio added by the utterance cuts | 89 ms per cut, 2.05 s over the corpus (23 cuts) |

## M2 serve-path protocol

The probe (`rtf_probe2.py`, lab run-004) seeds the corpus and a 114-word
paragraph as stored assistant turns, then speaks through `/api/speak` with
a real pair resident. Each sentence is one request. The paragraph is six
sentence requests, each issued when the previous stream closes, as the
browser does. Every chunk arrival is logged, and playout is simulated
against the arrivals. Cells alternated main (origin/main `e8a02e9b8`) and
stream on one boot.

| Tree, rounds | Sentence RTF median | First audio median (max) | Underruns / starved | Paragraph |
|---|---|---|---|---|
| main, r1–r4 (60 runs) | 1.330 (r1–r3), 1.324 (r4) | 1.412 s (r1–r3), 1.445 s (r4); max 1.66 s | 50 / 69.5 s | 2 underruns per round, 0.69–0.99 s starved |
| stream, r1–r3, untrimmed (45 runs) | 1.368 | 1.067 s (1.28 s) | 0 / 0 | 0 underruns, first audio 1.06–1.10 s |
| **stream, r4, shipped (15 runs)** | **1.278** | **1.065 s (1.29 s)** | **0 / 0** | **0 underruns, first audio 1.10 s, 41.98 s audio** |

## Other gates

| Gate | Result |
|---|---|
| Zero CPU dispatch, gdb count of `mlx::core::cpu::get_command_encoder` | positive control `mx.add(…, stream=mx.cpu)`: 3 calls; product streamer alone: 0 calls, stream completed on `Device(gpu, 0)` |
| Memory peak, one process, 15-sentence corpus | mlx peak 392 MB streamed vs 709 MB whole call; max RSS 283 MB vs 273 MB |
| WER, macstudio Whisper large-v3-turbo, 216 words | see below |

| Audio set (15-sentence corpus, af_heart) | WER |
|---|---|
| One whole call per sentence (jw16 reference) | 2/216 = 0.9% |
| Per-utterance whole decode, same cuts and trim as shipped | 3/216 = 1.4% |
| Streamed, shipped statistics (product check) | 3/216 = 1.4% |
| Serve path, main tree (round 1) | 4/216 = 1.9% |
| Serve path, streamed shipped build (round 4) | 3/216 = 1.4% |

All pass the gate. The streamed path matches the same utterances decoded
whole; the utterance cuts themselves cost 1 word on this set.

## Tests

- `tests/test_assistant_kokoro_stream.py` checks that:
  - the shipped statistics cover every pack voice at the pinned revision;
  - statistics from another revision, or a missing voice, are refused;
  - utterance cuts keep every word in order, prefer punctuation, and never leave a tiny tail.
- `tests/test_assistant_kokoro_stream_decode.py` runs where mlx and mlx_audio 0.5.6 are installed. It uses mlx_audio's real `Decoder` with Kokoro's istftnet config and checks that:
  - windows reproduce the whole decoder across a seam;
  - wrapped norms change nothing outside a stream;
  - a one-window utterance takes the unchanged decoder.

  On the dev box (CPU mlx 0.32.3) it passes in 79 s. It fails with the context mutated to 0 frames.
- `tests/test_install_sh_contract.py` checks that the installer fetches `kokoro_stream.py` and `kokoro_gen_stats.npz`.

## Status

Voice output stays unqualified. The frozen statistics sit 0.0014 (median)
below the same-wheel run-to-run floor. The ~2 s utterance cuts also change
prosody at the cuts, and WER cannot hear that. A listening A/B is part of
the owner decision; the per-sentence wavs from both trees are in lab
run-004 (`proto/rtf2/{main,stream}-r4-wav/`).
