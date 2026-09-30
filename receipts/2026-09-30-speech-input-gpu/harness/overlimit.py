"""The one corpus clip over 30 s, cut to 30.0 s as the browser recorder uploads it."""
import json, struct, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

from mlx_omarchy_assistant import recognition

A = Path("<home>/agents/SpeechInputGpu")
m = json.loads((A / "corpus/manifest.json").read_text())
over = [s for s in m["samples"] if s["duration_s"] > 30.0]
rec = recognition.Recognition(A / "home")
with open(sys.argv[1], "w") as out:
    for s in over:
        audio, rate = sf.read(s["audio_path"], dtype="float32")
        cut = audio[: 30 * rate]
        data = (np.clip(cut, -1, 1) * 32767).astype("<i2").tobytes()
        fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
        body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
        t0 = time.perf_counter()
        text = rec.transcribe(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
        out.write(json.dumps({"utt_id": s["utt_id"], "subset": s["subset"], "audio_s": len(cut) / rate,
                              "original_s": s["duration_s"], "hypothesis": text, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "error": None}) + "\n")
rec.close()
print(len(over), "over-limit clips")
