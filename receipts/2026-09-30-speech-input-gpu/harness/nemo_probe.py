"""NeMo reference for the empty-transcript uploads: nvidia/parakeet-tdt-0.6b-v3, CPU, default greedy
TDT decoding. Same inputs and pad/trim variants as the mlx runs."""
import json
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import nemo
import nemo.collections.asr as nemo_asr

P = Path("/tmp/sig/pull")
cases = {
    "pre_fix_upload17": P / "e2e-latency-20260930T061333Z/upload-17.wav",
    "pre_fix_upload16": P / "e2e-latency-20260930T061333Z/upload-16.wav",
    "post_fix_upload05": P / "e2e-latency-20260930T064321Z/upload-05.wav",
    "post_fix_upload03": P / "e2e-latency-20260930T064321Z/upload-03.wav",
}
out_dir = Path("/tmp/sig/nemo_inputs")
out_dir.mkdir(exist_ok=True)
files, labels = [], []
for label, path in cases.items():
    w = wave.open(str(path))
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    variants = {label: x, f"{label}_pad160": np.concatenate([x, np.zeros(160, np.float32)]),
                f"{label}_trim_end_800": x[:-800], f"{label}_trim_start_800": x[800:]}
    for name, v in variants.items():
        f = out_dir / f"{name}.wav"
        sf.write(f, v, 16000, subtype="PCM_16")
        files.append(str(f)); labels.append(name)
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
model.eval()
hyps = model.transcribe(files, batch_size=1)
texts = [h.text if hasattr(h, "text") else str(h) for h in (hyps[0] if isinstance(hyps, tuple) else hyps)]
rows = [{"case": l, "text": t} for l, t in zip(labels, texts)]
Path("/tmp/sig/nemo_probe.json").write_text(json.dumps({"nemo": nemo.__version__, "rows": rows}, indent=1))
print("nemo", nemo.__version__)
for r in rows:
    print(r["case"], repr(r["text"][:60]))
