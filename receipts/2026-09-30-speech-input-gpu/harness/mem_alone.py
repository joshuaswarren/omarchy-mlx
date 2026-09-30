"""mem_alone.py <wav>: Recognition alone: load, 30 transcribes, close. Prints worker peak memory."""
import json
import sys
import time
from pathlib import Path

from mlx_omarchy_assistant import recognition

rec = recognition.Recognition(Path("<home>/agents/SpeechInputGpu/home"))
data = Path(sys.argv[1]).read_bytes()
texts = set()
for _ in range(30):
    texts.add(rec.transcribe(data))
time.sleep(1)
peak = rec._worker.request({"op": "transcribe", "sample_rate": 16000}, b"\0" * 64000, timeout=30).get("peak_memory_bytes") \
    if getattr(rec, "_worker", None) else None
rec.close()
print(json.dumps({"requests": 30, "distinct_transcripts": sorted(texts), "worker_mx_peak_bytes": peak}))
