"""window_align.py <run-dir> <turn>...: align each 0.25 s window of an upload to the fake-mic source;
report the per-window source offset (samples at 16 kHz) and correlation. A jump in offset = a gap or
repeat in the captured audio."""
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
for turn in sys.argv[2:]:
    up = read(D / f"upload-{int(turn):02d}.wav")
    win = 4000
    size = 1 << (len(ring) + win).bit_length()
    fr = np.fft.rfft(ring, size)
    out = []
    for start in range(0, len(up) - win, win):
        seg = up[start:start + win]
        corr = np.fft.irfft(fr * np.conj(np.fft.rfft(seg, size)), size)[:len(src)]
        off = int(np.argmax(corr))
        c = corr[off] / (np.linalg.norm(seg) * np.linalg.norm(ring[off:off + win]) + 1e-9)
        out.append((start, off - start, round(float(c), 3)))
    print(turn, out)
