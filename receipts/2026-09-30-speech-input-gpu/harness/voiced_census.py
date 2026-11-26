"""voiced_census.py: voiced_seconds (numpy) for every corpus clip, by subset. Host CPU, no mlx."""
import collections
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mlx_omarchy_assistant import gpu_stt_worker as worker

A = Path("<home>/agents/SpeechInputGpu")
m = json.loads((A / "corpus/manifest.json").read_text())
by = collections.defaultdict(list)
for s in m["samples"]:
    x, r = sf.read(s["audio_path"], dtype="float32")
    x = x[:: r // 16000] if r != 16000 else x  # energy census only
    by[s["subset"]].append(round(worker.voiced_seconds(np, x), 2))
for subset, v in by.items():
    print(subset, "n", len(v), "min", min(v), "max", max(v),
          "clips >= 0.5 s voiced", sum(1 for x in v if x >= worker.MIN_VOICED_SECONDS))
