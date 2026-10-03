# Objective qualification of streamed Kokoro decoding — 2026-10-03

Voice output is recorded QUALIFIED on the objective bars, per the owner's
2026-10-03 direction: qualification is decided by measurements, with **no
human listening**. This receipt states what was measured, what failed
first, what changed, and what objective checks cannot show.

## Pre-registered bars

Written into the private lab notebook before any metric was computed
(`entries/KokoroQualify/20261003T181100Z-omp-studio-local-kokoro-objective-qualify.md`;
the notebook is private, the bars are quoted here in full):

1. Per-pair duration delta ≤ 2 %.
2. Voiced-frame F0 contour: Pearson correlation ≥ 0.95 and median pitch
   error ≤ 20 cents (estimator `librosa.pyin` 60–500 Hz; pure-numpy
   autocorrelation fallback; frames voiced in both arms, compared along a
   band-constrained DTW alignment after round 1 showed frame-index
   comparison is invalid once the two arms have different timelines).
3. Mel-cepstral distortion after DTW alignment ≤ 3 dB, or at most the
   same-wheel run-to-run floor + 1 dB. MCD = (10·√2/ln 10) × mean aligned
   frame distance over MFCC coefficients 1–24 (24 kHz, 80 mels, 25 ms/10 ms
   windows, Sakoe–Chiba band 15 %).
4. ASR WER (Whisper large-v3-turbo on the reference mac, same normalizer as
   every prior Kokoro receipt): ≤ 8 % overall and streamed ≤ whole + 1 word
   (216 reference words, the 15-sentence corpus; the long paragraph is
   transcribed and reported, not gated).
5. Clicks: at every logged chunk boundary, the max |x[n]−x[n−1]| in ±2 ms
   and the spectral flux (5 ms hop) must not exceed max(p99.9 × 1.10,
   global max) of the whole-call file's own distribution.
6. Pad-silence insertion per utterance join ≤ 100 ms; total duration delta
   reported with bar 1.

Round-1 acceptance additionally required the frozen voice-output gates
(corpora RTF ≥ 1.2, first audio ≤ 1.5 s, WER ≤ 8 %, zero CPU tensor
dispatches, memory), re-derived from the recorded serve-path logs.

## Round 1 — the shipped streamer (run-004, M2 lab host, boot `3d3b1e2e`)

Pairs: 15 sentences + one paragraph, streamed serve-path build vs
whole-call serve path, both arms 16-bit PCM 24 kHz. Inputs bound by the
lab manifest `artifacts/KokoroQualify/run-001/SHA256SUMS-inputs`; floor =
two whole-call renders of the same texts (rounds 3 vs 4 — same wheel, same
boot, independent noise draws). Suite: `scripts/kokoro_objective_ab.py`
(unit-tested on synthetic signals: an injected click is detected; identical
files give zero deltas; the F0 fallback recovers a 150 Hz sine within
5 cents).

| pair | dur Δ % | F0 corr | F0 cents | MCD dB | clicks | join silence s |
|---|---|---|---|---|---|---|
| para | 6.5 | 0.535 | 150 | 189.0 | pass | 0.091 |
| sent00 | 22.8 | 0.689 | 130 | 210.2 | pass | 0.000 |
| sent01 | 13.5 | 0.645 | 125 | 160.6 | pass | 0.090 |
| sent02 | 12.0 | 0.549 | 120 | 174.4 | pass | 0.092 |
| sent03 | 15.8 | −0.058 | 235 | 213.9 | pass | 0.090 |
| sent04 | 8.3 | 0.668 | 100 | 185.1 | pass | 0.090 |
| sent05 | 7.7 | 0.375 | 230 | 254.0 | pass | 0.090 |
| sent06 | 8.0 | 0.071 | 260 | 209.6 | pass | 0.090 |
| sent07 | 5.4 | 0.124 | 265 | 219.8 | pass | 0.090 |
| sent08 | 14.4 | 0.444 | 90 | 169.0 | pass | 0.090 |
| sent09 | 8.8 | 0.694 | 105 | 164.3 | FAIL | 0.090 |
| sent10 | 4.5 | 0.278 | 170 | 191.1 | pass | 0.090 |
| sent11 | 11.0 | 0.188 | 240 | 212.9 | pass | 0.090 |
| sent12 | 7.1 | 0.307 | 140 | 230.3 | pass | 0.090 |
| sent13 | 6.5 | 0.562 | 140 | 193.2 | pass | 0.091 |
| sent14 | 6.8 | 0.398 | 200 | 207.0 | pass | 0.090 |

Floor (whole vs whole, different noise seed): MCD median 28.9 dB
(max 30.3), F0 corr median 0.9990, duration delta 0.000 — the scale every
stream number must be read against.

| bar | round-1 result | verdict |
|---|---|---|
| 1 duration ≤ 2 % | stream SHORTER by 4.5–22.8 % per pair (median 8.2 %) | FAIL |
| 2 F0 ≥ 0.95, ≤ 20 cents | corr median 0.42 (−0.06–0.69), cents median 145 | FAIL |
| 3 MCD ≤ floor + 1 dB (≈ 29.9) | median 207 dB (164–254) | FAIL |
| 4 WER | main 4/216 = 1.9 %, stream 3/216 = 1.4 %, re-transcribed fresh | pass |
| 5 clicks | 15/16 pass; sent09 one join step 0.388 > whole-call max 0.375 | FAIL |
| 6 silence ≤ 100 ms | max 91.75 ms | pass |

WER was re-run fresh for this receipt (same model, same normalizer) and
reproduced the earlier serve-path numbers exactly.

## Root cause and the fix

The first streamer ran the pipeline front half (bert, duration predictor,
alignment, text encoder) **per utterance** on re-cut phonemes. Each ~2 s
utterance re-predicted its own phone durations, so the streamed sentence
was a chain of prosodic islands: 4.5–22.8 % shorter than the whole call
with diverging pitch (bars 1–3), and each join carried its own predicted
pad (the ~90 ms silences of bar 6) with one marginal seam step (bar 5,
sent09). Crossfading cannot fix re-predicted prosody.

`kokoro_stream.py` was restructured: the front half runs **once per
pipeline chunk** over the whole phoneme string, and utterances become
frame ranges of that single predicted timeline (cut positions chosen by
the same word-boundary/punctuation policy, mapped to aligned frames
through the vocab-filtered per-token durations). Each utterance decodes
with `UTTERANCE_CONTEXT_FRAMES` aligned frames of context on both sides
and joins its neighbour through a linear-gain crossfade over a few ms —
linear, not equal-power, because the two seam renders derive from the same
features and equal-power gains would add a spurious +3 dB bump. The last
generator stage stays windowed (600 ms) with the frozen per-voice
statistics; the `MLX_OMARCHY_KOKORO_STREAM=0` kill switch is unchanged.

Because durations, F0 and N now come from the whole-sentence prediction,
bars 1–3 measure only the windowing/frozen-statistics error, by
construction.

## Round 2 — the frame-sliced streamer (re-render)

TBD after the re-render (16 pairs + same-seed floor, product entry point,
timing recorded).

## Qualification receipt and status

`record_qualification` now accepts two forms: the listener-verified
receipt, or an OBJECTIVE receipt (`objective_verified` with the
`objective_results` of these bars and a `verification_basis` naming this
owner directive). Every receipt is bound — computed at write time, never
taken from the caller — to the pack revision, the pinned model hash, the
mlx backend identity, and hashes of `kokoro_stream.py` and
`kokoro_gen_stats.npz`; `status()` reports qualified only while a receipt
matching THIS runtime exists, and a streamer or statistics change
invalidates it. No `recommended` flag moved.

**No human listened.** At the owner's direction the qualification is
objective. These bars cannot hear prosody naturalness, an unnatural but
consistent intonation, or voice quality that WER still transcribes; the
pad-silence bar tolerates up to 100 ms per join by design. A listening A/B
remains open as an owner option; it is not a gate for this receipt.

TBD: provenance block (wheels, commits, pack revision, render host details).
