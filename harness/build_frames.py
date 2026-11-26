"""Framed stdin for gpu_stt_worker: 20 corpus clips (one resent at 48 kHz) + close."""
import json, struct, sys
from pathlib import Path

import numpy as np
import soundfile as sf

manifest = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
by_subset = {}
for s in manifest["samples"]:
    by_subset.setdefault(s["subset"], []).append(s)
picked = []
for subset, n in (("test-clean", 4), ("test-other", 3), ("accented", 3), ("short", 2),
                  ("final-chunk", 2), ("silence", 2), ("noise", 2), ("mixed_0dB", 2)):
    picked += by_subset[subset][:n]
frames = bytearray()
index = []


def frame(header, payload=b""):
    body = json.dumps(dict(header, payload_bytes=len(payload))).encode()
    frames.extend(struct.pack("<Q", len(body)) + body + payload)


for i, s in enumerate(picked):
    audio, rate = sf.read(s["audio_path"], dtype="float32")
    frame({"op": "transcribe", "id": i, "sample_rate": rate}, audio.astype("<f4").tobytes())
    index.append({"id": i, "utt_id": s["utt_id"], "subset": s["subset"], "rate": rate,
                  "reference": s["reference"]})
first = picked[0]
audio, rate = sf.read(first["audio_path"], dtype="float32")
up = np.interp(np.arange(len(audio) * 3) / 3, np.arange(len(audio)), audio).astype("<f4")
frame({"op": "transcribe", "id": len(picked), "sample_rate": 48_000}, up.tobytes())
index.append({"id": len(picked), "utt_id": first["utt_id"] + "@48k", "subset": "resample-48k",
              "rate": 48_000, "reference": first["reference"]})
frame({"op": "close", "id": len(picked) + 1})
out.write_bytes(bytes(frames))
Path(str(out) + ".index.json").write_text(json.dumps(index, indent=1))
print(f"{len(index)} transcribe frames, {len(frames)} bytes")
