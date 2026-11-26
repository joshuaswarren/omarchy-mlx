"""Find where upload-17 sits in the fake-mic source, then run Parakeet (stock mlx CPU) on the clean
source window and on the captured upload. Tells browser capture processing apart from model input."""
import importlib.util
import json
import sys
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

spec = importlib.util.find_spec("mlx_audio.stt.models")
sys.modules[spec.name] = importlib.util.module_from_spec(spec)
from mlx_audio.stt.utils import load_model  # noqa: E402

D = Path("/tmp/sig/pull/e2e-latency-20260930T061333Z")


def read(path):
    w = wave.open(str(path))
    return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768, w.getframerate()


src48, _ = read(D / "fake_mic_long.wav")
src = src48[::3]
model = load_model(Path("/tmp/sig/parakeet-tdt-0.6b-v3"))
rows = []
for name in sys.argv[1:]:
    up, rate = read(D / name)
    n = len(src) + len(up)
    size = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(np.concatenate([src, src[:len(up)]]), size) * np.conj(np.fft.rfft(up, size)), size)
    offset = int(np.argmax(corr[:len(src)]))
    clean = np.concatenate([src, src])[offset:offset + len(up)]
    peak = float(np.max(corr) / (np.linalg.norm(up) * np.linalg.norm(clean) + 1e-9))
    texts = {}
    for label, x in (("captured", up), ("clean_source", clean)):
        texts[label] = getattr(model.generate(mx.array(x), verbose=False), "text", "")
    rows.append({"upload": name, "offset_s": offset / 16000, "normalized_corr": round(peak, 3),
                 "captured_rms": round(float(np.sqrt((up ** 2).mean())), 4),
                 "clean_rms": round(float(np.sqrt((clean ** 2).mean())), 4),
                 "captured_clipped_fraction": round(float((np.abs(up) > 0.99).mean()), 5), **texts})
    print(json.dumps(rows[-1]))
Path("/tmp/sig/ref_window.json").write_text(json.dumps(rows, indent=1))
