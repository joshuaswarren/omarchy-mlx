"""probe_empty.py <wav> <out.json>: the failing upload through Parakeet (mlx-audio, pinned dir)
on the GPU and on mx.cpu, plus small pad/trim variants on the GPU. Diagnostic only."""
import json
import sys
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, "<home>/agents/SpeechInputGpu/repo/serve")
from mlx_omarchy_assistant.gpu_stt_worker import _register_bare_package  # noqa: E402

_register_bare_package("mlx_audio.stt.models")
from mlx_audio.stt.utils import load_model  # noqa: E402

w = wave.open(sys.argv[1])
x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
model = load_model(Path("<app-home>/voice/parakeet-tdt-0.6b-v3"))
model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)


def run(label, samples, device=None):
    t0 = time.perf_counter()
    if device is None:
        out = model.generate(mx.array(samples), verbose=False)
    else:
        with mx.stream(device):
            out = model.generate(mx.array(samples), verbose=False)
    return {"case": label, "n": int(samples.size), "seconds": samples.size / 16000,
            "ms": round((time.perf_counter() - t0) * 1000, 1), "text": getattr(out, "text", str(out))}


rows = [run("gpu", x)]
for pad in (160, 800, 1600, 4000, 8000):
    rows.append(run(f"gpu_pad_end_{pad}", np.concatenate([x, np.zeros(pad, np.float32)])))
for cut in (160, 800, 1600, 4000):
    rows.append(run(f"gpu_trim_start_{cut}", x[cut:]))
    rows.append(run(f"gpu_trim_end_{cut}", x[:-cut]))
rows.append(run("gpu_gain_0.5", x * 0.5))
Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
for r in rows:
    print(r["case"], r["n"], r["ms"], repr(r["text"][:50]))
