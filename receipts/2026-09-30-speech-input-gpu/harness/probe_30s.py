"""probe_30s.py <out.json>: why the padded path returns nothing on a 30 s tiled clip. M2 GPU."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_omarchy_assistant import gpu_stt_worker as worker

worker._register_bare_package("mlx_audio.stt.models")
from mlx_audio.stt.utils import load_model  # noqa: E402

A = Path("<home>/agents/SpeechInputGpu")
clip, _ = sf.read(str(A / "corpus/LibriSpeech/test-clean/1320/122617/1320-122617-0000.flac"), dtype="float32")
m = json.loads((A / "corpus/manifest.json").read_text())
real, _ = sf.read(next(s["audio_path"] for s in m["samples"] if s["utt_id"] == "121-123859-0002"), dtype="float32")
model = load_model(A / "home/voice/parakeet-tdt-0.6b-v3")
model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)
noise = worker._PAD_NOISE


def words(x):
    out = model.generate(mx.array(x.astype(np.float32)), verbose=False)
    return len((getattr(out, "text", "") or "").split())


def pad(x, s, kind="noise", ends="both"):
    n = int(s * 16000)
    p = noise[:n] if kind == "noise" else np.zeros(n, np.float32)
    left = p if ends in ("both", "start") else np.zeros(0, np.float32)
    right = p if ends in ("both", "end") else np.zeros(0, np.float32)
    return np.concatenate([left, x, right])


rows = []
for secs in (10, 20, 25, 28, 29, 29.6, 30):
    x = np.tile(clip, 5)[: int(secs * 16000)]
    rows.append({"input": f"tiled_{secs}s", "none": words(x), "noise0.4": words(pad(x, 0.4)),
                 "zeros0.4": words(pad(x, 0.4, "zeros")), "noise0.4_start": words(pad(x, 0.4, ends="start")),
                 "noise0.4_end": words(pad(x, 0.4, ends="end")), "noise1.0": words(pad(x, 1.0))})
    print(rows[-1], flush=True)
x = real[: 30 * 16000]
rows.append({"input": "real_30s", "none": words(x), "noise0.4": words(pad(x, 0.4)), "noise1.0": words(pad(x, 1.0))})
print(rows[-1])
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
