"""latency_only.py <out.json> <label>: fresh Recognition worker, cold call, 2 warm-ups, then 30 warm
requests on the 5.075 s test-clean clip, in order (per-request ms kept unsorted)."""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from mlx_omarchy_assistant import recognition

A = Path("<home>/agents/SpeechInputGpu")
m = json.loads((A / "corpus/manifest.json").read_text())
five = next(s for s in m["samples"] if s["utt_id"] == "121-127105-0022")
audio, rate = sf.read(five["audio_path"], dtype="float32")
data = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
payload = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body

rec = recognition.Recognition(A / "home")
t0 = time.monotonic()
rec.transcribe(payload)
cold = (time.monotonic() - t0) * 1000
for _ in range(2):
    rec.transcribe(payload)
ms, texts = [], set()
for _ in range(30):
    t0 = time.monotonic()
    texts.add(rec.transcribe(payload))
    ms.append(round((time.monotonic() - t0) * 1000, 1))
rec.close()
s = sorted(ms)
out = {"label": sys.argv[2], "clip": five["utt_id"], "audio_s": len(audio) / rate, "cold_ms": round(cold, 1),
       "warmups": 2, "ms_in_order": ms, "p50_ms": s[14], "p95_ms": s[28], "max_ms": s[-1],
       "distinct_transcripts": sorted(texts)}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("label", "cold_ms", "p50_ms", "p95_ms", "max_ms")}))
