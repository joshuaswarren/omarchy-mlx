"""Reference: stock mlx CPU + mlx-audio 0.5.6, pinned Parakeet revision, same variants as probe_empty.py."""
import importlib.util
import json
import sys
import time
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

spec = importlib.util.find_spec("mlx_audio.stt.models")
sys.modules[spec.name] = importlib.util.module_from_spec(spec)
from mlx_audio.stt.utils import load_model  # noqa: E402
import mlx_audio  # noqa: E402

path = snapshot_download("mlx-community/parakeet-tdt-0.6b-v3",
                         revision="ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
                         cache_dir="/tmp/sig/hf")
w = wave.open(sys.argv[1])
x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
link = Path("/tmp/sig/parakeet-tdt-0.6b-v3")
if not link.exists():
    link.symlink_to(path)
model = load_model(link)
model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)


def run(label, samples):
    t0 = time.perf_counter()
    out = model.generate(mx.array(samples), verbose=False)
    return {"case": label, "n": int(samples.size), "ms": round((time.perf_counter() - t0) * 1000, 1),
            "text": getattr(out, "text", str(out))}


rows = [run("cpu", x)]
for pad in (160, 800, 1600, 4000, 8000):
    rows.append(run(f"cpu_pad_end_{pad}", np.concatenate([x, np.zeros(pad, np.float32)])))
for cut in (160, 800, 1600, 4000):
    rows.append(run(f"cpu_trim_start_{cut}", x[cut:]))
    rows.append(run(f"cpu_trim_end_{cut}", x[:-cut]))
rows.append(run("cpu_gain_0.5", x * 0.5))
meta = {"mlx": mx.__version__, "device": str(mx.default_device()), "mlx_audio": getattr(mlx_audio, "__version__", "?"),
        "model_path": path}
Path(sys.argv[2]).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
print(meta)
for r in rows:
    print(r["case"], r["n"], r["ms"], repr(r["text"][:50]))
