#!/usr/bin/env python3
"""Generate the V10 wake-eval clip set: >=50 distinct positives, >=50
distinct negatives/confusables, deterministic espeak-ng manifest.

Keeps the two load-bearing clips from bc-voice-followup-1903 by name and
role: en-us+m3-195.wav (the positive both baseline runs missed) and
Hey-jealous-en-us+f2.wav (the negative that false-waked twice). The
baseline clips were not committed to the repo, so audio is regenerated
under this pinned manifest (espeak-ng 2.3.x, en-us + variant).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wAQ-v10-clips")
SR = 16000
VARIANTS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "f1", "f2", "f3"]

POSITIVE_PHRASES = [
    "hey jarvis",
    "hey jarvis. did you get that",
    "hey, jarvis",
    "hey jarvis what is the weather",
    "hey jarvis can you hear me",
]
NEGATIVE_TEXTS = [
    "hey jealous, jealous eye",
    "hey service, run the diagnostics",
    "hey jervis, how are you",
    "the jar of visco sat on the shelf",
    "january was colder last year",
]

manifest = []

def add(name, variant, text, label):
    manifest.append({"file": name, "variant": variant, "text": text, "label": label})

# 50 distinct positives: 10 variants x 5 phrasings.
i = 0
for v in VARIANTS:
    for phrase in POSITIVE_PHRASES:
        i += 1
        add(f"pos-{v}-{i:03d}.wav", v, phrase, "positive")

# Keep the missed-positive role: the baseline's en-us+m3 hard case is the
# short form in m3, so m3's plain "hey jarvis" carries the baseline name.
assert manifest[10]["variant"] == "m3" and manifest[10]["text"] == "hey jarvis"
manifest[10]["file"] = "en-us+m3-195.wav"

# 55 negatives: 10 variants x 5 confusables + 5 unrelated-digit clips.
j = 0
for v in VARIANTS:
    for text in NEGATIVE_TEXTS:
        j += 1
        add(f"neg-{v}-{j:03d}.wav", v, text, "negative")
for v in ("m1", "m4", "m7", "f1", "f3"):
    j += 1
    add(f"neg-{v}-digits-{j:03d}.wav", v,
        "one two three four five six seven eight nine ten", "negative")

# Keep the false-wake role on f2.
for entry in manifest:
    if entry["file"] == "neg-f2-041.wav":  # f2's first confusable = "hey jealous..."
        entry["file"] = "Hey-jealous-en-us+f2.wav"
        break
assert any(e["file"] == "Hey-jealous-en-us+f2.wav" for e in manifest)

OUT.mkdir(parents=True, exist_ok=True)
for entry in manifest:
    wav = OUT / entry["file"]
    subprocess.run(
        ["espeak-ng", "-v", f"en-us+{entry['variant']}", "-w", str(wav), entry["text"]],
        check=True)
    data, sr = sf.read(wav, dtype="int16")
    if sr != SR:
        n = int(len(data) * SR / sr)
        data = sig.resample_poly(data.astype("float64"), SR, sr).astype("int16")
    # 320 ms trailing silence so every clip carries the streaming context the
    # detector would see on a live wire.
    pad = np.zeros(int(0.32 * SR), dtype="int16")
    sf.write(wav, np.concatenate([data, pad]), SR, subtype="PCM_16")

counts = {}
for e in manifest:
    counts[e["label"]] = counts.get(e["label"], 0) + 1
manifest_path = OUT / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
print("positives:", counts.get("positive"), "negatives:", counts.get("negative"))
print("manifest:", manifest_path)
