"""probe_upload.py <out.json> <wav>... : each WAV through Recognition.transcribe five times (one worker)."""
import json
import sys
import time
from pathlib import Path

from mlx_omarchy_assistant import recognition

rec = recognition.Recognition(Path("<home>/agents/SpeechInputGpu/home"))
rec.transcribe(Path(sys.argv[2]).read_bytes())
rows = []
for path in sys.argv[2:]:
    data = Path(path).read_bytes()
    for k in range(5):
        t0 = time.perf_counter()
        text = rec.transcribe(data)
        rows.append({"wav": Path(path).name, "try": k, "ms": round((time.perf_counter() - t0) * 1000, 1),
                     "text": text})
rec.close()
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
print(json.dumps([(r["wav"], r["try"], r["text"][:60]) for r in rows]))
