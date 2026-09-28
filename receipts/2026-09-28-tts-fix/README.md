# TTS hum fix — local voice output on <project-m2> (Apple M2 Max, Omarchy Linux, Vulkan/Honeykrisp)

Date: 2026-09-28. Symptom: the pinned Qwen3-TTS pack
(`mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`, revision
`08c72cad5e2fd0f41730c8bd1f28149585e46361`, voice `serena`, via
`mlx-audio==0.5.6`) produced a low-frequency hum instead of speech in
MLX Chat on <project-m2>.

## Root cause

The `<project-m2>` venv held a **stale mlx-omarchy wheel**:
`0.32.3.dev202609232032+4fd2130ed` (built 2026-09-23), whose Omarchy
Vulkan backend corrupted large tensor graphs — the codec decoder
composition (RVQ embedding sum, k=1 projections, transposed-view adds)
decoded fixed codes to garbage while every isolated op passed. The
current wheel `0.32.3.dev202609282218+29cba8e` (installed 17:24 on
2026-09-28) decodes the same codes to the Metal reference within
floating-point noise. No synthesis.py change is required or made.

Evidence chain:

1. Whisper (`mlx_whisper`, whisper-large-v3-turbo, on <reference-mac>,
   M1 Ultra) transcribed a `say`-generated control perfectly, then the
   five hum WAVs as junk: "I don't know." / "Thank you." / "Thank you."
   / "you" / "Thank you." — no words (tool proven, audio is noise).
2. The same five sentences generated on <reference-mac> with the same
   pack, mlx-audio 0.5.6, and the same synthesis.py transcribed
   essentially perfectly (1 word error overall: "chose"→"choose") —
   the pack and the calling code are correct.
3. Greedy (temperature 0) cross-machine codes for sentence 1 diverge
   from frame 2 group 4 (talker numerics; see "Talker divergence"),
   so decoder A/B used the <reference-mac> codes as a fixed input.
4. Stale-wheel decoder A/B on <project-m2> (identical codes):
   rel-RMS 1.52 vs the Metal reference in both the full and streaming
   paths — pure noise out, valid codes in.
5. Stage decomposition on the stale wheel: every stage from the RVQ
   quantized composition onward diverged (quantized rel-RMS 1.42,
   maxdiff 100) while single ops (embedding gather, k=1 conv, plain
   add, strided copies) matched NumPy bit-for-bit; decoder weight
   digests were identical across machines (sha256
   `df4f339bd8770453...37db58`), so weights were not the cause.
6. Fixed-wheel decoder A/B on <project-m2>, same codes, same scripts:
   rel-RMS 3.0e-06, maxdiff 2e-6 — bit-near-exact.
7. Regression test `tests/test_qwen3_tts_codec_regress.py` (pinned
   Metal fixtures + tolerances) passes on <project-m2> with the fixed
   wheel and fails on a 0.1%-corrupted fixture, so it discriminates.

## Talker divergence (benign, characterized)

Greedy talker codes on <project-m2> first differ from Metal at frame 2
group 4 of sentence 1 and cascade (603/656 prefix positions differ;
generation runs 44 vs 41 frames). The divergence is deterministic
across processes and across both wheels and is ordinary sampling
drift: the same process's audio transcribes exactly
("The local assistant is ready to help.", 0% WER). Different valid
samples, not a defect; no fix required.

## Non-causes tested and ruled out (receipted during triage)

- Pack and synthesis calling code: identical files on both machines
  (`diff` of synthesis.py: identical); Metal reference is clean.
- Decoder weights: sha256 digest of every decoder parameter identical
  across machines.
- Buffer reuse (quarantine/recycle): `MLX_OMARCHY_NO_BUFFER_CACHE=1`
  and `MLX_OMARCHY_TAPE_NO_REUSE=1` do not change stale-wheel results.
- Poisoned recycled storage: `MLX_OMARCHY_POISON_FREED=1` produced no
  poison-signature values.
- Fusion: `MLX_OMARCHY_FUSED_CHAIN=0` and `MLX_OMARCHY_DEFER_COMMIT=0`
  do not change stale-wheel results.
- Honeykrisp swallow recovery: deliberate single/multi submission drops
  (`MLX_OMARCHY_TEST_DROP_SUBMIT=1,2,3,5,8,13,21,34`, `1,2`, `1,2,3`,
  `2,3,4,5`, `3,4`, `5,6,7`) and signal strips
  (`MLX_OMARCHY_TEST_DROP_SIGNAL=1,2,3,5,8,13`) all recover
  value-correctly on the fixed wheel (maxdiff 3.8e-05, fp32 rounding).
- Concurrency: two uncoordinated TTS processes, and a matmul load
  process beside the probe, both decode correctly on the fixed wheel.

## Fix deployment

The fix is the current wheel. Rebuild/reinstall on a machine only if
its wheel predates 29cba8e: check with
`pip show mlx-omarchy` (must be `0.32.3.dev202609282218+29cba8e` or
newer) and `scripts/mlx_provenance.py` beside any measurement. The
repository itself already contains the fix; the worktree adds the
regression test and fixtures only.

## Regression test

`tests/test_qwen3_tts_codec_regress.py` + `tests/fixtures/qwen3_tts_codec/`
(pinned codes, reference quantized tensor, reference waveform, shapes;
Metal reference from <reference-mac>). Honest-skip: without the mlx
wheel, mlx-audio, an accelerator, or `MLX_OMARCHY_TTS_TEST_PACK`, the
test skips and names the prerequisite; execution is never faked.

Run on <project-m2> (receipt):

```
cd /tmp/ttsfix_test && MLX_OMARCHY_TTS_TEST_PACK=$PACK \
  PYTHONPATH=$VOICE_SITE:$SERVE python -m unittest test_qwen3_tts_codec_regress -v
# Ran 2 tests ... OK
```

Sensitivity: perturbing the fixture by 0.1% fails the test; restored,
it passes.

## End-to-end proof

Five sentences regenerated on <project-m2> with the fixed wheel
(`0.32.3.dev202609282218+29cba8e`, `scripts/mlx_provenance.py`:
version_match=true) and transcribed on <reference-mac> with
whisper-large-v3-turbo.

| sentence | stale wheel (hum) | fixed wheel | Metal reference |
|---|---|---|---|
| 1 | 100% | 0% | 0% |
| 2 | 100% | 0% | 0% |
| 3 | 100% | 0% | 10% |
| 4 | 90% | 10% | 0% |
| 5 | 100% | 11.1% | 0% |
| overall | 97.9% | **4.3%** | 2.1% |

Targets: overall <= 10% (got 4.3%), no sentence > 25% (got 11.1%).
Transcripts: `stale-wheel-hum-transcript-*.txt` (before),
`m2-fixed-transcript-*.txt` (after), `reference-mac-transcript-*.txt`
(reference ceiling). `wer_check.py` normalizes digits to words and
punctuation.

Timing (round 0 of the lock-held run, GPU alone, flock
/tmp/m2-gpu.lock held; first-chunk includes lazy warm-up):

| sentence | audio s | first chunk s | total s | RTF |
|---|---|---|---|---|
| 1 | 3.20 | 3.94 | 17.23 | 5.4 |
| 2 | 4.80 | 1.37 | 21.92 | 4.6 |
| 3 | 5.04 | 1.40 | 23.12 | 4.6 |
| 4 | 4.32 | 1.41 | 19.67 | 4.6 |
| 5 | 5.44 | 1.41 | 25.21 | 4.6 |

Warm steady state (rounds 1-2): the sampler process was killed
externally after round 0 (shared machine), so the p50/p95 warm-up set
did not land; the round-0 numbers above are the receipt (first-audio
1.37-3.94 s incl. lazy warm-up, RTF 4.6-5.4). Lock-held spot check
transcribed exactly (`timed-spotcheck.txt`). The whole pipeline remains ~4.6x slower than realtime on <project-m2>
(Metal reference is faster than realtime on <reference-mac>) — the
known second problem, explicitly out of scope here ("treat speed as a
second problem after correctness"). Beside-load receipt: a full
sampler run completed with correct transcripts while another agent's
benchmark held the GPU (`beside-load-timing.txt`: first_audio p50
1.40 s, p95 1.43 s) — correctness holds under concurrent load.

Listening copies for the owner: `<home>/mlx-tts-samples-fixed/`
(index.html, same layout as `mlx-tts-samples/`). Qualification receipt
(`synthesis.record_qualification`) is left for the parent to record
after the owner's audible listener check: the receipt requires
`listener_verified=true`, which only the owner can supply.
