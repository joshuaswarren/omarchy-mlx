"""fake_mic.wav (7.8 s LibriSpeech clip, 48 kHz for Chromium's capture), five_s.wav (test-clean clip
nearest 5 s), and fake_mic_long.wav (the first 60 s of test-clean clips back to back, 48 kHz): any
5 s window of the long file is continuous read speech."""
import json, sys, wave
from pathlib import Path

import numpy as np
import soundfile as sf

A = Path("<home>/agents/SpeechInputGpu")
out = Path(sys.argv[1])
m = json.loads((A / "corpus/manifest.json").read_text())


def write(path, x, rate):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


clip = A / "corpus/LibriSpeech/test-clean/1320/122617/1320-122617-0000.flac"
audio, rate = sf.read(str(clip), dtype="float32")
up = np.interp(np.arange(len(audio) * 3) / 3, np.arange(len(audio)), audio)
write(out / "fake_mic.wav", up, 48_000)
parts, total = [], 0
for s in (s for s in m["samples"] if s["subset"] == "test-clean"):
    x, r = sf.read(s["audio_path"], dtype="float32")
    parts.append(np.interp(np.arange(len(x) * 3) / 3, np.arange(len(x)), x))
    total += len(x) / r
    if total >= 60:
        break
write(out / "fake_mic_long.wav", np.concatenate(parts), 48_000)
five = min((s for s in m["samples"] if s["subset"] == "test-clean"), key=lambda s: abs(s["duration_s"] - 5))
audio, rate = sf.read(five["audio_path"], dtype="float32")
write(out / "five_s.wav", audio, rate)
print(json.dumps({"fake_mic": str(clip), "five_s": five["utt_id"], "five_s_seconds": len(audio) / rate,
                  "five_s_reference": five["reference"]}))
