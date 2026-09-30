"""upload_fidelity.py <run-dir>: for each saved upload, its best normalized cross-correlation against the
fake-mic source (16 kHz), RMS ratio, and the turn's result. Numpy only."""
import json
import sys
import wave
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])


def read(path):
    w = wave.open(str(path))
    return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768


src = read(D / "fake_mic_long.wav")[::3]
ring = np.concatenate([src, src])
turns = json.loads((D / "browser_latency_turns.json").read_text())
rows = []
for t in turns:
    f = D / f"upload-{t['i']:02d}.wav"
    if not f.exists():
        rows.append({"i": t["i"], "upload": None, "ms": t["stop_to_transcript_ms"], "words": t["words"]})
        continue
    up = read(f)
    size = 1 << (len(ring) + len(up)).bit_length()
    corr = np.fft.irfft(np.fft.rfft(ring, size) * np.conj(np.fft.rfft(up, size)), size)[:len(src)]
    off = int(np.argmax(corr))
    clean = ring[off:off + len(up)]
    rows.append({"i": t["i"], "warmup": t["warmup"], "seconds": round(len(up) / 16000, 3),
                 "corr": round(float(corr[off] / (np.linalg.norm(up) * np.linalg.norm(clean) + 1e-9)), 3),
                 "rms_ratio": round(float(np.sqrt((up ** 2).mean()) / (np.sqrt((clean ** 2).mean()) + 1e-9)), 3),
                 "ms": t["stop_to_transcript_ms"], "words": t["words"]})
print(json.dumps(rows))
measured = [r for r in rows if not r.get("warmup")]
print("corr min/median", min(r["corr"] for r in measured if "corr" in r),
      float(np.median([r["corr"] for r in measured if "corr" in r])),
      "missing", sum(1 for r in measured if r["ms"] is None))
(D / "upload_fidelity.json").write_text(json.dumps(rows, indent=1))
