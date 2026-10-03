#!/usr/bin/env python3
"""Calibrate the frozen statistics used by streamed Kokoro decoding.

For every voice in the pinned pack, cuts a fixed calibration corpus
(disjoint from the read-aloud evaluation corpus) into the same utterances
the product decodes (kokoro_stream.segment_inputs), runs the whole decoder
on each, records the per-channel time mean and variance at every AdaIN
layer of the generator's last stage, averages them per utterance, and
writes serve/mlx_omarchy_assistant/kokoro_gen_stats.npz.

Usage (Apple Silicon host with the pinned pack and the voice runtime):
  PYTHONPATH=serve python3 scripts/kokoro_gen_stats.py <pack_dir> [out.npz]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

CORPUS = [
    "Good morning.",
    "The coffee is ready.",
    "Turn left at the second light.",
    "I saved the draft and closed the editor.",
    "The train was late, so we waited on the platform for twenty minutes.",
    "Could you send me the notes from yesterday's planning call?",
    "Bring a warm jacket, because the evening gets cold by the lake.",
    "The printer on the third floor jams whenever someone uses the thick paper.",
    "We moved the release to Tuesday so the team could finish the last round of testing.",
    "Her garden grows tomatoes, peppers, beans, and a row of bright yellow sunflowers along the fence.",
    "After the storm passed, the neighbors gathered in the street to clear branches and check on each other.",
    "The museum opens at ten, and the guided tour of the new wing starts every hour on the half hour.",
    "Yes.",
    "Thanks, that helps a lot.",
    "Three apples, two pears, and a melon.",
    "Please lock the door when you leave tonight.",
    "The battery lasted almost two days on a single charge.",
    "Our flight was cancelled, and the airline rebooked us on the first departure tomorrow morning.",
    "He fixed the squeaky hinge with a drop of oil and a little patience.",
    "When the bell rang, the students packed their bags and hurried out into the sunny courtyard.",
    "The report covers sales, staffing, and the budget for the next quarter.",
    "If the soup is too thick, add a cup of water and let it simmer a little longer.",
    "The old bridge closes for repairs next month, so plan an extra ten minutes for the detour.",
    "Rain is expected after lunch.",
]


def main():
    pack = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).resolve().parents[1] / "serve" / "mlx_omarchy_assistant" / "kokoro_gen_stats.npz"
    from mlx_omarchy_assistant.kokoro_stream import last_stage_adains, segment_inputs
    from mlx_omarchy_assistant.synthesis import KOKORO_PACK, _kokoro_runtime
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    pipe = _kokoro_runtime(str(pack))
    decoder = pipe.model.decoder
    layers = last_stage_adains(decoder)
    seen: dict = {}

    class Recorder(nn.Module):
        def __init__(self, name, inner):
            super().__init__()
            self._name = name
            self._inner = inner

        def __call__(self, x):
            seen[self._name] = (mx.mean(x.astype(mx.float32), axis=2).reshape(-1),
                                mx.var(x.astype(mx.float32), axis=2).reshape(-1))
            return self._inner(x)

    originals = {name: adain.norm for name, adain in layers.items()}
    for name, adain in layers.items():
        adain.norm = Recorder(name, originals[name])

    arrays = {}
    t0 = time.perf_counter()
    for voice in KOKORO_PACK["voices"]:
        sums = {name: [0.0, 0.0] for name in layers}
        count = 0
        for text in CORPUS:
            for asr, F0, N, s, _keep in segment_inputs(pipe, text, voice):
                mx.eval(decoder(asr, F0, N, s))
                count += 1
                missing = set(layers) - set(seen)
                if missing:
                    raise SystemExit(f"layers not reached: {sorted(missing)[:3]}")
                mx.eval(*[t for pair in seen.values() for t in pair])
                for name, (m, v) in seen.items():
                    sums[name][0] = sums[name][0] + np.asarray(m)
                    sums[name][1] = sums[name][1] + np.asarray(v)
                seen.clear()
        for name, (m, v) in sums.items():
            arrays[f"{voice}|{name}|mean"] = (m / count).astype(np.float32)
            arrays[f"{voice}|{name}|var"] = (v / count).astype(np.float32)
        print(json.dumps({"voice": voice, "layers": len(sums), "utterances": count}), flush=True)

    corpus_sha = hashlib.sha256("\n".join(CORPUS).encode()).hexdigest()
    meta = {"pack_id": KOKORO_PACK["id"], "pack_revision": KOKORO_PACK["revision"],
            "mlx_audio": KOKORO_PACK["runtime"]["mlx_audio"]["version"],
            "corpus_sha256": corpus_sha, "sentences": len(CORPUS),
            "voices": KOKORO_PACK["voices"]}
    np.savez(out, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({"wrote": str(out), "bytes": out.stat().st_size, "sha256": digest,
                      "seconds": round(time.perf_counter() - t0, 1), **meta}), flush=True)


if __name__ == "__main__":
    main()
