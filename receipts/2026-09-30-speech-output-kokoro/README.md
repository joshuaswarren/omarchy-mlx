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
| Time to first audio p95 | <= 1.0 s | **10.8 s** (af_bella), 10.5 s (af_heart) eager; clause-split first-chunk lands only with a passing RTF | **FAIL** |
| Whisper large-v3-turbo WER (33 of 35 files; 2 truncated WAVs repaired and pending re-transcription) | <= 8% overall, no sentence > 25% | **0.87% overall** (4/461 words), worst sentence **12.5%** (af_bella "oclock") | **PASS** |
| Zero CPU tensor dispatch, whole-synthesis trace | 0 | trace pass in flight (`artifacts/trace-2026*/summary.json`); the backend has no CPU evaluator and run-001's gate crash proves unsupported ops fail loudly, not silently | in flight, named |
| Peak memory / cold start | record | cold model load **0.79 s**; peak record lands with the resumable bench completion (`results.jsonl` peak line) | partial |

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
   kernels per sentence; the backend's default elementwise chain fusion
   (FuseDecodeChains, `MLX_OMARCHY_FUSED_CHAIN`, on by default) already
   collapses the float unary/binary runs, so what remains is conv-heavy.
4. **`mx.compile` root cause (step 1 of the follow-up): unusable on this
   wheel, two distinct failures.**
   - A pure 4-op Snake function `x + (1/a)*(sin(a*x)**2)` wrapped in
     `mx.compile` **hangs** — no trace completion within 100 s (killed by
     the turn cap; expected trace time is milliseconds), repro
     `bench/kokoro_compile2.py` / `-c` one-liner, 2026-09-30.
   - Compiling graph-bearing blocks (whole `Decoder` instance, or
     block-level `mx.compile` of `decoder.encode`, `decode[]`,
     `generator.resblocks[]`, `generator.noise_res[]`) traces but dies at
     the final `mx.eval` with `RuntimeError: [eval] Attempting to eval an
     array without a primitive` (kokoro.py:175), repro
     `artifacts/compile2-stdout.log` / `profile-stdout.log`.
   - Both repros ran WITHOUT the trig wrapper installed for the pure
     hang, so the trig reduction is not the cause. This is a backend
     tape/compile defect in the omarchy wheel (0.32.3.dev Vulkan build),
     not fixable from the synthesis layer; it needs the overlay owners.
   - Note for the fuse path: `shaders/fast_rope.comp` documents that
     fused chains remove the per-primitive trig gate sites — the
     graph-level argument reduction here protects fused Snake chains
     too, since arguments are folded before any sin site, eager or
     fused.
5. **WER (step 4):** Whisper large-v3-turbo on the reference Mac, 33
   files: overall 0.87%, worst sentence 12.5% — both gates pass with
   margin. Transcript quality is not the blocker; speed is.

## Follow-up table (Main's 2026-09-30 order)

| Step | Before | After | Evidence |
|---|---|---|---|
| 1. mx.compile root cause | unknown crash | pure Snake compile **hangs** (>100 s, no trace); block/graph compile fails at eval: `[eval] Attempting to eval an array without a primitive` — backend tape/compile defect, reported not fixed (out of synthesis scope) | `compile2-stdout.log`, `profile-stdout.log`, 2-min `-c` repro |
| 2. compiled Snake/AdaIN blocks | eager RTF 0.69 | **not achievable**: compile unusable (step 1) | same |
| 3. TTFA clause split | single-segment synthesis, TTFA = whole sentence | shipped: `_kokoro_generate` splits the first clause into its own segment (`_CLAUSE_SPLIT`, maxsplit=1); first-chunk shortness now scales with RTF — still FAIL until RTF passes | synthesis.py, tests 69/69 |
| 4. WER | not measured | 0.87% overall / 12.5% worst — PASS | `/tmp` macstudio hypotheses + `wer.json` staged in notebook artifacts |
| 4. dispatch trace, peak memory, live smoke | not run | chain queued on the M2 (bench remainder -> trace -> smoke); artifact paths `trace-2026*/summary.json`, `smoke.json`, `results.jsonl` peak line | `gpu-turn --status` |

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
