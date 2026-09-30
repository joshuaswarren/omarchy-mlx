"""step1_uploads.py <out.json>: every saved browser upload through the bare model (baseline) and the
worker's padded decode path, plus the padded path at one tenth of the level. M2 GPU."""
import glob
import json
import sys
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_omarchy_assistant import gpu_stt_worker as worker

worker._register_bare_package("mlx_audio.stt.models")
from mlx_audio.stt.utils import load_model  # noqa: E402

A = "<home>/agents/SpeechInputGpu/runs"
RUNS = ("e2e-latency-20260930T061333Z", "e2e-latency-20260930T064321Z", "e2e-latency-20260930T070223Z")
model = load_model(Path("<app-home>/voice/parakeet-tdt-0.6b-v3"))
model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)
noise = mx.array(worker._PAD_NOISE)


def generate(audio):
    return str(getattr(model.generate(audio, verbose=False), "text", "") or "")


rows = []
for run in RUNS:
    for path in sorted(glob.glob(f"{A}/{run}/upload-*.wav")):
        w = wave.open(path)
        x = mx.array(np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768)
        t0 = time.perf_counter()
        padded = worker.decode_padded(generate, mx, x, noise)
        padded_ms = (time.perf_counter() - t0) * 1000
        rows.append({"run": run, "upload": Path(path).name, "seconds": x.shape[0] / 16000,
                     "voiced_s": worker.voiced_seconds(mx, x), "baseline": generate(x),
                     "padded": padded, "padded_ms": round(padded_ms, 1),
                     "padded_quiet_x0.1": worker.decode_padded(generate, mx, x * 0.1, noise)})
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
for key in ("baseline", "padded", "padded_quiet_x0.1"):
    print(key, "empty", sum(1 for r in rows if not r[key]), "of", len(rows))
print("voiced min", min(r["voiced_s"] for r in rows))
