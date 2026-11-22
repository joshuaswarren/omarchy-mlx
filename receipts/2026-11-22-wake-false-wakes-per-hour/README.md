# Wake false wakes per hour — 0.5 vs 0.45

The follow-up measurement the tuning receipt asked for: what a lower wake
threshold costs on ordinary non-wake audio, and what it buys on the
recorded wake set. Both questions answered at thresholds 0.5 and 0.45
through the shipped detector.

## Corpus: 1.479 hours of openly licensed non-wake speech

LibriSpeech `test-clean`, 700 files, **1.479 hours**, 16 kHz mono, CC BY 4.0
(https://www.openslr.org/resources/12/). Read aloud books: continuous
speech with no wake phrase in it — the hardest ordinary case for a wake
word, and nothing from this house.

Reproduce the selection:

```
curl -fsSL -o test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
tar -tzf test-clean.tar.gz | grep '\.flac$' | sort | head -700 > all.txt
tar -xzf test-clean.tar.gz -T all.txt
find LibriSpeech -name '*.flac' | sort > corpus-list.txt   # absolute paths
```

Measured with the shipped tuner's corpus mode: one continuous stream
through one model, files joined by 0.5 s of silence, wake events counted
(not clips), hours counted as speech audio only (the synthetic gaps are
excluded).

```
python -m mlx_omarchy_assistant.wake_eval --mode corpus \
  --corpus-list corpus-list-abs.txt --model <wake model> \
  --thresholds 0.55,0.5,0.45,0.4 --floors 0 --out corpus-results.json
```

| threshold | wake events | wakes per hour |
|---|---|---|
| 0.55 | 0 | 0.0 |
| 0.50 | 0 | 0.0 |
| 0.45 | 0 | 0.0 |
| 0.40 | 0 | 0.0 |

Zero wake events at every threshold down to 0.40 across 1.4797 hours of
continuous read speech (`corpus-results.json`).

## Wake set: the recorded 20+20 fixtures

The fixtures from the offline-voice run
(`w7m-offline-voice-20261120/fixtures/wake-{positive,negative}` on the
16-inch M1 Max): 20 positive "hey jarvis" clips, 20 negative confusables
(Hey-Travis, Hey-apples, Hey-jealous, Hey-service, Jarvis across the
en-us / en-gb voices). These are the clips behind the baseline's
"38 of 40 and 2 of 40" numbers. Fetch them with:

```
scp -r '16m1mbp:~/scratch/w7m-offline-voice-20261120/fixtures/wake-positive/*.wav' wake-positive/
scp -r '16m1mbp:~/scratch/w7m-offline-voice-20261120/fixtures/wake-negative/*.wav' wake-negative/
```

Two measurement paths, because the baseline's numbers come from one
continuous listener session:

**One continuous session** (the baseline's conditions):

| threshold | positives accepted | negatives falsely accepted |
|---|---|---|
| 0.5 | 19/20 — miss `en-us+m3-195.wav` | 2/20 — `Hey-jealous-en-us+f2.wav`, `Hey-service-en-us.wav` |
| 0.45 | 19/20 — same miss | 4/20 — the two above plus `Hey-service-en-gb.wav`, `Hey-service-en-us+f2.wav` |
| 0.40 | 20/20 | 6/20 |

This reproduces the baseline's numbers exactly: 19 of 20 positives, 2 of
20 negatives, the same missing clip and the same first false accept.

**Fresh model per clip** (the tuning receipt's protocol):

| threshold | positives accepted | negatives falsely accepted |
|---|---|---|
| 0.50 | 20/20 | 1/20 |
| 0.45 | 20/20 | 1/20 |
| 0.40 | 20/20 | 3/20 |
| 0.35 | 20/20 | 7/20 |

Identical at 0.5 and 0.45: on per-clip scoring the lower threshold buys
nothing and costs nothing.

The baseline's miss (`en-us+m3-195.wav`) reproduces exactly in the
continuous session and disappears with a fresh model per clip: it is a
cross-clip state artifact, not a threshold problem. openwakeword keeps
cross-utterance state and its `reset()` does not clear it (measured in
the tuning receipt: fresh 0.4305, after reset 0.4762, without reset
0.9572 on the same clip).

## Verdict

On the recorded wake set, 0.45 buys nothing: the phrase the baseline
missed stays missed in the continuous session (a state artifact, not a
threshold issue), per-clip scoring is identical at 0.5 and 0.45, and the
deployment-like session's false accepts double from 2 to 4. 0.40 does
recover the missed phrase, at three times the false accepts.

Non-wake audio (the table above): **zero wake events at every threshold down to
0.40.** Open read speech does not separate the thresholds: on this
corpus, 0.45 costs nothing measurable and buys nothing measurable.

**Keep the default at 0.5.** The recorded wake set is the only place the
thresholds separate, and there 0.45 strictly loses: same missed phrase,
double the false accepts in the deployment-like session. The espeak-clip
gain in the tuning receipt (TPR 0.42 -> 0.58 at 0.45) does not transfer
to the recorded fixtures, whose positives all clear 0.5 with a fresh
model and whose one continuous-session miss is a state artifact, not a
threshold artifact. A noisier corpus (TV, music, household noise) can
still separate them; the tuner re-measures it in one command.
