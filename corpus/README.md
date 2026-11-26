# Speech-input evaluation corpus

The pinned corpus used to qualify the GPU speech-recognition path. Audio is not committed;
the scripts rebuild it byte-for-byte from public sources, and `manifest.json` (relative paths,
per-file sha256) is the record of what was scored.

## Contents (192 utterances)

| Subset | N | Source |
|---|---:|---|
| test-clean | 60 | LibriSpeech test-clean (OpenSLR SLR12, CC BY 4.0) |
| test-other | 40 | LibriSpeech test-other (SLR12) |
| accented | 35 | Crowdsourced UK/Ireland English, Midlands female (OpenSLR SLR83, CC BY-SA 4.0) |
| short | 12 | Natural LibriSpeech utterances under 2 s |
| final-chunk | 5 | test-clean utterances trimmed so speech runs to the final sample |
| silence | 10 | 3 s of digital zero |
| noise | 10 | 3 s of synthetic pink noise (Voss-McCartney, seed 20260929) |
| mixed_10dB, mixed_0dB | 10 each | test-clean speech plus pinned babble at 10 dB and 0 dB SNR |

Noise for the mixes: [High_school_cafeteria.ogg](https://commons.wikimedia.org/wiki/File:High_school_cafeteria.ogg),
public domain, Commons sha1 `5b92ff5f3b82a9984d2def1eb0300d12f29e37a2`, sha256
`b40fb3f28ef3b7e04dc0006695f089fc4699571a9cefe87650914f17c7907907`, decoded to 16 kHz mono
with `ffmpeg -map 0:a:0 -ac 1 -ar 16000`. Speech RMS and noise RMS are measured over the
whole clip; the noise segment (seeded offset) is scaled so that
`20*log10(rms_speech / rms_noise)` equals the target SNR.

## Rebuild

From this directory, with `test-clean.tar.gz`, `test-other.tar.gz` (SLR12) and
`midlands_english_female.zip` (SLR83) downloaded here:

```bash
tar -xzf test-clean.tar.gz && tar -xzf test-other.tar.gz      # creates LibriSpeech/
unzip -q -d midlands midlands_english_female.zip
python3 build_manifest.py   # LibriSpeech + Midlands selection (seed 42)
python3 build_noise.py      # silence and pure-noise clips
python3 build_extend.py     # test-clean top-up, short, final-chunk, babble mixes
```

## Scoring

References and hypotheses from every system go through one normalizer: `&` becomes AND,
`%` becomes PERCENT, digit groups (thousands separators, decimals, ordinals) become English
words, text is uppercased, and every other non-alphanumeric character becomes a space. WER per
subset is total word edits divided by total reference words.

Frozen thresholds (set before scoring): test-clean ≤ 6%, test-other ≤ 14%, accented ≤ 20%,
0 dB SNR ≤ 30%; silence and pure noise empty in ≥ 95% of clips; a ~5 s clean clip returns a
transcript with p95 ≤ 2.0 s over 30 warm requests.
