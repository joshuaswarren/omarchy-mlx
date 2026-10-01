"""Corpus + latency run over the real Recognition.transcribe path. Resumable.

Writes one JSON line per clip to <run>/clips.jsonl (skips ids already there),
then <run>/latency.json: cold first call, 30 warm requests on a ~5 s clip,
and the worker's lifetime mx peak memory. Scoring happens in score.py.
"""
import json, os, struct, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

from mlx_omarchy_assistant import recognition

A = Path("<home>/agents/SpeechInputGpu")
RUN = Path(sys.argv[1])
RUN.mkdir(parents=True, exist_ok=True)
manifest = json.loads((A / "corpus/manifest.json").read_text())


def wav(path):
    audio, rate = sf.read(path, dtype="float32")
    data = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body, len(audio) / rate


clips = RUN / "clips.jsonl"
done = {json.loads(l)["utt_id"] for l in clips.read_text().splitlines()} if clips.exists() else set()
rec = recognition.Recognition(A / "home")
t0 = time.monotonic()
rec.transcribe(wav(manifest["samples"][0]["audio_path"])[0])
cold_ms = (time.monotonic() - t0) * 1000
with clips.open("a") as out:
    for s in manifest["samples"]:
        if s["utt_id"] in done:
            continue
        payload, seconds = wav(s["audio_path"])
        t0 = time.monotonic()
        try:
            text, error = rec.transcribe(payload), None
        except Exception as exc:
            text, error = None, f"{type(exc).__name__}: {exc}"
        row = {"utt_id": s["utt_id"], "subset": s["subset"], "audio_s": round(seconds, 3),
               "latency_ms": round((time.monotonic() - t0) * 1000, 1),
               "hypothesis": text, "error": error}
        out.write(json.dumps(row) + "\n")
        out.flush()
        os.fsync(out.fileno())

five = min((s for s in manifest["samples"] if s["subset"] == "test-clean"),
           key=lambda s: abs(s["duration_s"] - 5.0))
payload, seconds = wav(five["audio_path"])
rec.transcribe(payload)
warm = []
for _ in range(30):
    t0 = time.monotonic()
    rec.transcribe(payload)
    warm.append((time.monotonic() - t0) * 1000)
peak = rec._worker.request({"op": "transcribe", "sample_rate": 16_000},
                           np.zeros(16_000, "<f4").tobytes(), timeout=60)["peak_memory_bytes"]
worker_pid = rec._worker._process.pid
vm = {k: v.strip() for k, v in (l.split(":", 1) for l in
      Path(f"/proc/{worker_pid}/status").read_text().splitlines()) if k in ("VmHWM", "VmRSS")}
rec.close()
warm.sort()
(RUN / "latency.json").write_text(json.dumps({
    "cold_first_call_ms": round(cold_ms, 1),
    "warm_clip": {"utt_id": five["utt_id"], "audio_s": round(seconds, 3)},
    "warm_ms": [round(x, 1) for x in warm],
    "p50_ms": round(warm[14], 1), "p95_ms": round(warm[28], 1), "max_ms": round(warm[-1], 1),
    "worker_mx_peak_memory_bytes": peak, "worker_proc_status": vm}, indent=1))
print("done", len(manifest["samples"]), "clips")
