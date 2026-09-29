# Speech output — Kokoro-82M second engine (2026-09-30)

Kokoro-82M (StyleTTS2-based, non-autoregressive, 24 kHz, Apache-2.0) was
evaluated as a second voice engine next to the default Qwen3-TTS pack, on
the M2 Max (jw14m2-linux, Vulkan mlx). **The speed threshold fails, so
Kokoro is NOT the default and not recommended; the owner listens before
any decision.** The engine ships behind the honest picker: voices are
offered, assets hash-pinned, and every number below is reproducible from
the receipt sources.

## Verdict vs frozen thresholds

Thresholds were fixed before measurement (notebook entry
`20260929T213728Z-jw14m2-kokoro-tts-engine.md`).

| Threshold | Required | Measured | Verdict |
|---|---|---|---|
| RTF (audio s / wall s), per-voice median over 15 sentences x 3 rounds | >= 5 | **0.69** (af_heart 0.686, af_bella 0.694, am_michael 0.660) | **FAIL** (~7x short) |
| Worst sentence median RTF | >= 5 | 0.65 (sentence 1, all voices) | **FAIL** |
| Time to first audio p95 | <= 1.0 s | **10.8 s** (af_bella), 10.5 s (af_heart) | **FAIL** |
| Whisper large-v3-turbo WER | <= 8% overall, no sentence > 25% | **not measured** — the speed gate already fails; intelligibility does not change the decision | not run |
| Zero CPU tensor dispatch, whole-synthesis trace | 0 | trace pass **did not run** (queue cap); the backend has no CPU evaluator and the run-001 gate crash proves unsupported ops fail loudly, not silently | incomplete, named |
| Peak memory / cold start | record | cold model load **0.79 s**; peak record **not captured** (bench capped mid-run; resumable continuation appends it) | partial |

Per-voice medians are stable across 95 measured runs (RTF min 0.65, max
0.71) — the failure is architectural, not noise.

## What the run found

1. **Backend finding (run-001):** the first forward crashed on the
   omarchy trig accuracy gate — `Sin argument magnitude 127261.73 >
   100000` (Snake activation `sin(alpha*x)`, istftnet.py:382/389; source
   phase :516/:597/:621). The gate is by design: the driver's range
   reduction is untrusted above the limit and the software Payne-Hanek
   fallback miscompiles, so the backend refuses rather than computing
   wrong values. No CPU fallback occurred.
2. **Graph-level fix (shipped):** `_kokoro_install_trig_reduction` in
   `synthesis.py` folds sin/cos arguments in-graph (3-term Cody-Waite,
   float32: `k = floor(x/2π); r = (x − k·6.28125) − k·0.0019353072`),
   applied only for arguments above 1e4 via `mx.where`. Offline
   validation (numpy, 1e6 samples in [1e3, 3e5]): max error 4.3e-6 rad
   (~-107 dB). The recording probe confirmed exactly three reduced sites
   per sentence (max args 127261, 180160, 12138; all other trig arguments
   ≤ 36). Audio generates on the Vulkan backend after the fix.
3. **Speed verdict:** with the fix, one forward per sentence costs
   4.5-6.6 s wall for 2.8-4.5 s of audio on the M2 Max — RTF ≈ 0.69,
   TTFA p95 ≈ 10.5 s. The eager graph runs hundreds of small conv/deconv
   kernels per sentence; a `mx.compile` attempt through the supported
   seam (`model.decoder = mx.compile(model.decoder)`) fails on this
   wheel ("Attempting to eval an array without a primitive" — the
   compiled callable is not a pure function over captured inputs), and
   mlx-audio 0.5.6's pipeline never passes its `decoder` parameter, so
   the seam needs an mlx-audio-side change. Reaching RTF >= 5 needs
   either that compile path or fused Vulkan kernels for the
   Snake/AdaIN/ISTFT stack — neither is in this change.

## What shipped

- `serve/mlx_omarchy_assistant/synthesis.py` — `KOKORO_PACK`
  (mlx-community/Kokoro-82M-bf16 @ a71e4d38b236d968966a2002c4c895dbd12b1c3c,
  5 files sha256-pinned, 328,684,463 bytes), the `VOICE_ENGINES`
  registry, per-engine resident workers (init handshake, per-pack asset
  verification and dependency probes), offline Kokoro runtime
  (misaki 0.7.4 + num2words + spacy/en_core_web_sm + phonemizer +
  espeakng-loader user-space wheel — no root, no system package, no
  network at run time), the trig reduction, and honest voice options
  (`af_heart`, `af_bella`, `am_michael` — American English, pack's own
  voice tensors).
- Voice picker groups options by engine (`optgroup`); an engine whose
  assets are not downloaded renders disabled and labeled. The default
  stays Qwen3-TTS / `aiden`.
- Tests: 69 synthesis tests (9 new Kokoro-focused: pinned manifest,
  engine registry, voice validation, per-engine worker residency),
  475/475 assistant tests, 4/4 JS suites pass.
- `docs/serve.md` — two-engine voice text.

## Sources

- Raw bench: `~/agents/SpeechOutputKokoro/artifacts/results.jsonl` (M2),
  summarized per voice; probe: `probe-20260929T222435Z/probe.json`
  (trig sites, per-voice probe RTFs); profile attempt:
  `profile-stdout.log` (compile seam failure).
- Listening samples (owner): `<home>/mlx-tts-samples-kokoro/`
  with `index.html`; all three voices carry all five sentences.
- Notebook (private):
  `apple-silicon-lab/entries/SpeechOutputKokoro/20260929T213728Z-jw14m2-kokoro-tts-engine.md`,
  artifacts under `artifacts/SpeechOutputKokoro/run-002/` with
  SHA256SUMS.
- Branch: `feat/speech-output-kokoro` (3 commits on origin/main
  f29c101db), pushed to main per repo rules.

## Not done / blocked (named)

- WER, full dispatch trace, and the peak-memory record are not captured
  (bench turn caps + queue contention; the speed verdict does not depend
  on them).
- No qualification was recorded (`record_qualification` untouched); no
  pair is qualified; `recommended` stays false everywhere.
