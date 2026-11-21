# FluidAudio TTS benchmark contract

Audit of FluidAudio's published TTS benchmark at commit
`e0c6103bc87a88d25276cad491c86e757f58c865` (v0.18.4 podspec bump), read
for the on-device-voice comparison card. Checker:
`scripts/check_fluidaudio_tts_contract.py`; tests:
`tests/test_fluidaudio_tts_contract.py`.

## English corpus: pin and order

- **Source:** `MiniMaxAI/TTS-Multilingual-Test-Set` (Hugging Face,
  CC-BY-SA-4.0), `text/english.txt`.
- **Pinned revision:** `cb416f0ac3658da0577e97873065e19fe6488917` — the
  initial public release, pinned in code at
  `Sources/FluidAudioCLI/Commands/MinimaxCorpusCommand.swift:27`
  (`defaultRevision`) and in `Documentation/TTS/MinimaxCorpus.md`.
- **Content:** 100 lines, `<cloning_audio_filename>|<text>` per line.
  Raw file sha256:
  `1020375295c73eef7ff66a611885b0ba91aaa2b6d07cfe76ca2332e05f942e30`.
- **Order:** upstream file order is the benchmark order. The harness
  conversion (`MinimaxCorpusCommand.convert`, `MinimaxCorpusCommand.swift:136-155`)
  trims each line, skips empties, strips the `<filename>|` prefix at the
  **first** pipe, and drops empty texts — no reordering. The benchmark's
  own parser (`parseCorpus`, `TtsBenchmarkCommand.swift:1184-1193`)
  additionally skips `#` comments and preserves file order. Result:
  100 phrases, position 1 = `english_female|I guess it comes down a
  simple choice…`, position 100 = `english_male|The culinary tour
  introduces visitors…`.
- The checker verifies the raw sha256, the 100-phrase count, and the
  full ordered list against the vendored pin
  (`tests/fixtures/fluidaudio_tts/minimax_english_cb416f0.txt`, CC-BY-SA-4.0
  upstream content vendored for the check).

## English WER / CER normalisation

Scorer: `WERCalculator.calculateWERAndCER`
(`Sources/FluidAudioCLI/Utils/WERCalculator.swift:36-96`). Hypothesis is
a Parakeet TDT roundtrip transcript of each rendered WAV
(`TtsBenchmarkCommand.swift:996-1003`); `--skip-asr` drops the pass.

- Both sides run through `TextNormalizer.normalize`
  (`Sources/FluidAudioCLI/Utils/TextNormalizer.swift:397-606`), the
  HuggingFace/Open-ASR-Leaderboard English path: lowercase →
  British→American word-boundary map (`english.json`, 1739 entries,
  longest-key-first) → abbreviation expansion → bracket/parenthesis
  strip → filler-word strip → stutter-prefix strip → apostrophe and
  number/letter boundary fixes → diacritic fold → `$`/`&`/`%` → words →
  punctuation→space (word chars and `'` survive) → contraction
  expansion → multi-word number folding (`convertNumbers`,
  `TextNormalizer.swift:636-712`) → number-word→digit map →
  digit-comma merge → symbol cleanup → whitespace collapse.
- WER = Levenshtein total over whitespace tokens ÷ reference words;
  `editDistance` (`WERCalculator.swift:186-248`) backtraces
  substitutions → insertions → deletions. Edge: empty reference →
  `0.0`, not an error.
- CER = Levenshtein total over characters with spaces stripped ÷
  reference characters.
- Numbers fold both ways: `2024` and `two thousand twenty four`
  normalise to the same tokens, so the scorer does not count
  digit-vs-word-form as an error. Derived probe counts are in the
  checker (`scorer` mode): numeric spell-out 0 errors; a wrong digit
  (`2025` for `2024`) is 1 word + 1 char substitution (5.26% / 1.27% on
  the 19-word / 79-char line); an empty transcript is all deletions
  (8/8 words, 30/30 chars → 100% / 100%).

## Timing boundaries — three separate measurements

These numbers are not interchangeable; the doc keeps them apart.

| Metric | What it wraps | Source |
|---|---|---|
| **Batch completion** `synth_ms` | `Date()` before the synthesize call → after the full waveform returns, per phrase. For batch backends `ttft_ms == synth_ms`. Aggregates: `warm_synth_ms_p50/p95`, `agg_rtfx = Σaudio/Σsynth`. | `TtsBenchmarkCommand.swift:423-439` (Kokoro), same shape per backend (`:609-617`, `:705-712`, `:781-785`) |
| **First streamed PCM** `ttft_ms` | `Date()` → first frame yielded by `synthesizeStreaming` — recorded inside `for try await frame in stream` at `frameCount == 0`. PocketTTS only; first frame is an 80 ms Mimi frame (1920 samples @ 24 kHz). | `TtsBenchmarkCommand.swift:522-536`; granularity `Documentation/TTS/Benchmarks.md` "Audio chunk window size" |
| **v3 token-to-PCM** | Timer around `pipeline.synthesize(inputIds:style:speed:)` through **complete PCM** — includes native DSP/STFT and array handling; **excludes** text processing, phonemization, model loading, WAV encoding, playback. Median of 5 timed runs after 2 discarded warmups; identical cached token IDs and style vector per policy. | Method: `Documentation/TTS/Measurements/KokoroV3/comparison.json` (`method` field), `Documentation/TTS/Benchmarks.md` "Kokoro ANE v3: selected M5 Pro measurements"; in-tree stage timer: `KokoroAneSynthesizer.predict` `ContinuousClock`, `Sources/FluidAudio/TTS/KokoroAne/Pipeline/KokoroAneSynthesizer.swift:213-217` |

Adjacent but distinct: `cold_start_s` (`manager.initialize()` wall,
`TtsBenchmarkCommand.swift:390-392`) and `first_synth_ms` (first synth
after init, `:395-405`). The v3 numbers are complete-audio latencies —
explicitly *not* streaming first-audio measurements and not sums of the
per-stage timers.

## FLEURS sample identities — unknown

`fleurs-benchmark` fetches `FluidInference/fleurs` (or `fleurs-full`)
via `tree/main` — **no revision pin**
(`FleursBenchmark.swift:263`). `--samples N` takes
`audioFiles.prefix(N)` over the HF tree-listing order
(`FleursBenchmark.swift:305-310`), so the chosen set depends on the
listing at fetch time. Published results record per-language counts
only (e.g. en_us 350 in `Documentation/Benchmarks.md`); the JSON report
does carry `sampleId` per sample, so a run can name its own files, but
no published table ships them. **Public source cannot recover the
sample identities behind the published FLEURS numbers** — treated as
unknown for any comparison.

## Reproduce

```bash
# Offline: corpus pin/order + scorer probes
python3 scripts/check_fluidaudio_tts_contract.py

# Verify the live pinned upstream file
python3 scripts/check_fluidaudio_tts_contract.py corpus --fetch

# Verify every source anchor still names the timer/scorer it cites
python3 scripts/check_fluidaudio_tts_contract.py anchors \
  --fluidaudio-src /path/to/FluidAudio

pytest tests/test_fluidaudio_tts_contract.py
```
