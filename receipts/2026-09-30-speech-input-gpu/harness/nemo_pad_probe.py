"""NeMo reference for the worker's edge padding on the three empty browser uploads: none, 0.4 s and
1 s of the worker's fixed pad noise on both ends."""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import nemo
import nemo.collections.asr as nemo_asr

sys.path.insert(0, "<home>/.config/superpowers/worktrees/mlx-omarchy/SpeechInputGpu/serve")
from mlx_omarchy_assistant import gpu_stt_worker as worker  # noqa: E402

P = Path("/tmp/sig/pull")
cases = {
    "run1_upload17": P / "e2e-latency-20260930T061333Z/upload-17.wav",
    "run2_upload05": P / "e2e-latency-20260930T064321Z/upload-05.wav",
    "run3_upload24": P / "e2e-latency-20260930T070223Z/upload-24.wav",
}
out_dir = Path("/tmp/sig/nemo_pad_inputs")
out_dir.mkdir(exist_ok=True)
files, labels = [], []
noise = worker._PAD_NOISE
for label, path in cases.items():
    w = wave.open(str(path))
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    for seconds in (0.0, worker.EDGE_PAD_SECONDS, worker.RETRY_PAD_SECONDS):
        n = int(seconds * 16000)
        v = np.concatenate([noise[:n], x, noise[len(noise) - n:]]) if n else x
        f = out_dir / f"{label}_pad{seconds}.wav"
        sf.write(f, v, 16000, subtype="FLOAT")
        files.append(str(f)); labels.append(f"{label}_pad{seconds}")
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
model.eval()
hyps = model.transcribe(files, batch_size=1)
texts = [h.text if hasattr(h, "text") else str(h) for h in (hyps[0] if isinstance(hyps, tuple) else hyps)]
rows = [{"case": l, "text": t} for l, t in zip(labels, texts)]
Path("/tmp/sig/nemo_pad_probe.json").write_text(json.dumps({"nemo": nemo.__version__, "rows": rows}, indent=1))
for r in rows:
    print(r["case"], repr(r["text"][:70]))
