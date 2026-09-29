# STT corpus manifest

Pinned offline corpus used to qualify the GPU STT path on Apple Silicon
Linux machines (mlx Vulkan backend).

## Files

- `build_manifest.py` — walks LibriSpeech SLR12 test-clean + test-other,
  picks ≥60/≥40 clips, hashes each, writes `manifest.json`.
- `build_noise.py` — generates silence + Voss-McCartney pink noise at
  fixed RMS, mixes 5 clean clips with noise at 10 dB SNR and 5 at
  0 dB SNR. Mixing is deterministic (rng seed = 20260930).

## Sources

- LibriSpeech SLR12: <https://openslr.elda.org/resources/12/>
  - `test-clean.tar.gz` 331 MB, license CC-BY-4.0
  - `test-other.tar.gz` 314 MB, license CC-BY-4.0
- Midlands English female SLR83: <https://openslr.elda.org/resources/83/midlands_english_female.zip>
  - 100 MB, license CC-BY-SA-4.0 (used as the accented-English subset)

## Frozen thresholds (pre-registered, see AGENTS.md)

- test-clean WER ≤ 6 %
- test-other WER ≤ 14 %
- accented WER ≤ 20 %
- 0 dB SNR WER ≤ 30 %
- 10 dB SNR WER ≤ 18 %
- silence / pure noise empty transcripts ≥ 95 % of clips
- p95 latency, 5 s clean clip over 30 warm requests ≤ 2.0 s
- 0 CPU tensor dispatch events on the wired mlx-audio path

## Building

```bash
# Place test-clean.tar.gz and test-other.tar.gz in this dir, plus
# midlands_english_female.zip
tar -xzf test-clean.tar.gz -C LibriSpeech/
tar -xzf test-other.tar.gz -C LibriSpeech/
unzip -d midlands midlands_english_female.zip
python3 build_manifest.py
python3 build_noise.py
```

## Notes

- The manifest is reproducible with seed 42 for clip selection; the
  noise mix uses seed 20260930. Re-running produces an identical manifest.
- All audio files are hashed with sha256; the manifest's `sha256` field
  identifies the exact byte sequence.
- The Midlands corpus's line_index CSV uses commas with leading spaces;
  `build_manifest.py` strips whitespace before resolving filenames.