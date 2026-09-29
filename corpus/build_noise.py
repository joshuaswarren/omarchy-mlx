"""Generate synthetic silence and noise clips + noise-mixed speech at 10 dB and 0 dB SNR.

Reads from the manifest and overwrites the placeholder entries with real audio files,
mixing pinned noise (synthetic pink) at the requested SNR.
"""
import json, hashlib, struct, wave
from pathlib import Path
import numpy as np
import soundfile as sf

CORPUS = Path("<project-m2>/agents/SpeechInputGpu/corpus")
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
        if s["audio_path"].startswith("<generated"):
            kind = s["subset"]
            idx = int(s["utt_id"].rsplit("_", 1)[1])
            out = OUT_DIR / f"{s['utt_id']}.wav"
            duration = int(s["duration_s"] * s["sample_rate"])
            if kind == "silence":
                samples = np.zeros(duration, dtype=np.float32)
                write_wav(out, samples, s["sample_rate"])
            elif kind == "noise":
                samples = pink_noise(duration, rng)
                write_wav(out, samples, s["sample_rate"])
            s["audio_path"] = str(out)
        elif s.get("subset") == "short":
            # Already resolved; nothing to do.
            pass

    # Generate noise-mixed clips: pick first 10 test-clean clips, mix with pink noise
    # at 10 dB SNR (5 clips) and 0 dB SNR (5 clips).
    test_clean_pool = [s for s in manifest["samples"] if s.get("source") == "librispeech" and s.get("subset") == "test-clean"]
    rng2 = np.random.default_rng(20260930)
    mixed_entries = []
    for i, clip in enumerate(test_clean_pool[:10]):
        snr_db = 10 if i < 5 else 0
        clean, rate = sf.read(clip["audio_path"], dtype="float32")
        if rate != 16000:
            # quick resample via numpy linear (good enough for noise mixing)
            new_len = int(len(clean) * 16000 / rate)
            idx = np.linspace(0, len(clean) - 1, new_len)
            left = np.floor(idx).astype(int)
            right = np.minimum(left + 1, len(clean) - 1)
            t = idx - left
            clean = clean[left] * (1 - t) + clean[right] * t
            rate = 16000
        # RMS normalize to a level
        clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-9)
        target_clean_rms = 0.1
        clean = clean * (target_clean_rms / clean_rms)
        # Generate pink noise to match length
        noise = pink_noise(len(clean), rng2)
        noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-9)
        target_noise_rms = target_clean_rms / (10 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / noise_rms)
        mixed = clean + noise
        out = OUT_DIR / f"mixed_{snr_db}dB_{i:02d}.wav"
        write_wav(out, mixed, rate)
        mixed_entries.append({
            "utt_id": f"mixed_{snr_db}dB_{i:02d}",
            "audio_path": str(out),
            "reference": clip["reference"],
            "duration_s": round(len(mixed) / rate, 3),
            "sample_rate": rate,
            "channels": 1,
            "source": f"synthetic_noise_mix_{snr_db}dB",
            "subset": f"mixed_{snr_db}dB",
            "base_utt_id": clip["utt_id"],
            "snr_db": snr_db,
            "sha256": None,  # filled below
        })

    manifest["samples"].extend(mixed_entries)
    # Hash everything that was newly written
    for s in manifest["samples"]:
        if s.get("sha256") is None:
            p = Path(s["audio_path"])
            if p.is_file():
                s["sha256"] = sha256_file(p)
    # Update thresholds metadata
    manifest["noise_mix"] = {
        "kind": "voss_mccartney_pink",
        "snr_targets_db": [0, 10],
        "clean_rms_target": 0.1,
        "rng_seed": 20260930,
        "code": "stt-noise.py (entries/SpeechInputGpu)",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"manifest updated with {len(mixed_entries)} mixed clips + {sum(1 for s in manifest['samples'] if s['audio_path'].startswith(str(OUT_DIR)))} synthetic clips")
    print(f"total samples: {len(manifest['samples'])}")


if __name__ == "__main__":
    main()