"""Step 2: write the synthetic silence and pure pink-noise clips named in the manifest.

Speech-in-noise mixes come from build_extend.py (pinned public-domain babble).
"""
import json, hashlib, wave
from pathlib import Path
import numpy as np

CORPUS = Path(__file__).resolve().parent
MANIFEST = CORPUS / "manifest.json"
OUT_DIR = CORPUS / "synthetic"
OUT_DIR.mkdir(exist_ok=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pink_noise(n, rng):
    """Voss-McCartney pink noise; deterministic per rng seed."""
    rows = 16
    state = rng.standard_normal(rows)
    out = np.zeros(n, dtype=np.float32)
    counter = 0
    for i in range(n):
        counter += 1
        out[i] = state[counter & (rows - 1)] * 0.5
        # update one row randomly
        r = int(rng.integers(0, rows))
        state[r] = rng.standard_normal()
    return out / (np.abs(out).max() + 1e-9)


def write_wav(path, samples, rate=16000):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def main():
    manifest = json.loads(MANIFEST.read_text())
    rng = np.random.default_rng(20260929)
    for s in manifest["samples"]:
        if not s["audio_path"].startswith("<generated"):
            continue
        out = OUT_DIR / f"{s['utt_id']}.wav"
        duration = int(s["duration_s"] * s["sample_rate"])
        samples = (np.zeros(duration, dtype=np.float32) if s["subset"] == "silence"
                   else pink_noise(duration, rng))
        write_wav(out, samples, s["sample_rate"])
        s["audio_path"] = str(out)
        s["sha256"] = sha256_file(out)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"total samples: {len(manifest['samples'])}")


if __name__ == "__main__":
    main()