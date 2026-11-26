"""probe_dither.py <out.json>: digital-silence hypothesis. Each input decoded as-is, with fixed-seed
white noise added to the whole clip (1e-5 = the model's training dither, and 1e-4), and with the
current edge padding. Inputs: all saved browser uploads, tiled 10-30 s clips, the real 30 s clip."""
import glob
import json
import sys
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_omarchy_assistant import gpu_stt_worker as worker

worker._register_bare_package("mlx_audio.stt.models")
from mlx_audio.stt.utils import load_model  # noqa: E402

A = Path("<home>/agents/SpeechInputGpu")
model = load_model(A / "home/voice/parakeet-tdt-0.6b-v3")
model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)
rng_noise = np.random.default_rng(0).standard_normal(40 * 16000).astype(np.float32)
pad_noise = mx.array(worker._PAD_NOISE)


def text(x):
    return str(getattr(model.generate(mx.array(x.astype(np.float32)), verbose=False), "text", "") or "")


def variants(x):
    return {"none": text(x), "dither1e-5": text(x + 1e-5 * rng_noise[: x.size]),
            "dither1e-4": text(x + 1e-4 * rng_noise[: x.size]),
            "edge_pad_path": worker.decode_padded(lambda a: str(getattr(model.generate(a, verbose=False), "text", "") or ""),
                                                  mx, mx.array(x.astype(np.float32)), pad_noise)}


rows = []
for path in sorted(glob.glob(f"{A}/runs/e2e-latency-2026093*/upload-*.wav")):
    if not any(r in path for r in ("061333Z", "064321Z", "070223Z")):
        continue
    w = wave.open(path)
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    rows.append({"input": path.split("runs/")[1], "zero_fraction": float((x == 0).mean()),
                 **{k: len(v.split()) for k, v in variants(x).items()}})
clip, _ = sf.read(str(A / "corpus/LibriSpeech/test-clean/1320/122617/1320-122617-0000.flac"), dtype="float32")
for secs in (10, 20, 25, 28, 29, 29.6, 30):
    x = np.tile(clip, 5)[: int(secs * 16000)]
    rows.append({"input": f"tiled_{secs}s", "zero_fraction": float((x == 0).mean()),
                 **{k: len(v.split()) for k, v in variants(x).items()}})
m = json.loads((A / "corpus/manifest.json").read_text())
real, _ = sf.read(next(s["audio_path"] for s in m["samples"] if s["utt_id"] == "121-123859-0002"), dtype="float32")
x = real[: 30 * 16000]
rows.append({"input": "real_30s", "zero_fraction": float((x == 0).mean()), **{k: len(v.split()) for k, v in variants(x).items()}})
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
uploads = [r for r in rows if r["input"].startswith("e2e")]
for k in ("none", "dither1e-5", "dither1e-4", "edge_pad_path"):
    print(k, "uploads empty", sum(1 for r in uploads if r[k] == 0), "of", len(uploads))
for r in rows:
    if not r["input"].startswith("e2e"):
        print(r)
