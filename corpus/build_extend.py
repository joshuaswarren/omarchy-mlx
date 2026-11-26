"""Second corpus pass: >=60 test-clean, >=10 natural short clips, final-chunk
clips, and 10 dB / 0 dB mixes with a pinned public-domain babble recording.

Mixing: speech RMS and noise RMS are measured over the whole clip and the
noise segment (seeded random offset into the recording) is scaled so that
20*log10(rms_speech / rms_noise) equals the target SNR; if the sum would
clip, speech and noise are scaled down together, preserving the SNR.
"""
import hashlib, json, random, urllib.request, wave
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
OUT = ROOT / "synthetic"
NOISE = {"url": "https://upload.wikimedia.org/wikipedia/commons/d/df/High_school_cafeteria.ogg",
         "page": "https://commons.wikimedia.org/wiki/File:High_school_cafeteria.ogg",
         "licence": "Public domain (Wikimedia Commons; author aradlaw via pdsounds.org)",
         "commons_sha1": "5b92ff5f3b82a9984d2def1eb0300d12f29e37a2"}
rng = random.Random(20260930)
nrng = np.random.default_rng(20260930)


def sha(path, algo="sha256"):
    return hashlib.new(algo, Path(path).read_bytes()).hexdigest()


def write_wav(path, x, rate=16_000):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())


def libri_pool(subset):
    for trans in sorted((ROOT / "LibriSpeech" / subset).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            path = trans.parent / f"{utt}.flac"
            info = sf.info(str(path))
            yield {"utt_id": utt, "audio_path": str(path), "reference": text,
                   "duration_s": round(info.frames / info.samplerate, 3), "sample_rate": info.samplerate,
                   "channels": 1, "source": "librispeech"}


m = json.loads(MANIFEST.read_text())
used = {s["utt_id"] for s in m["samples"]}
# The one "short" row was a relabelled test-clean clip; keep it and top test-clean up to 60.
clean_pool = [s for s in libri_pool("test-clean") if s["utt_id"] not in used]
other_pool = [s for s in libri_pool("test-other") if s["utt_id"] not in used]
rng.shuffle(clean_pool)
have = sum(s["subset"] == "test-clean" for s in m["samples"])
for s in [s for s in clean_pool if s["duration_s"] >= 2.0][: 60 - have]:
    m["samples"].append(dict(s, subset="test-clean")); used.add(s["utt_id"])

shorts = [s for s in clean_pool + other_pool if s["duration_s"] < 2.0 and s["utt_id"] not in used]
have = sum(s["subset"] == "short" for s in m["samples"])
for s in shorts[: 12 - have]:
    m["samples"].append(dict(s, subset="short")); used.add(s["utt_id"])

finals = [s for s in clean_pool if s["utt_id"] not in used and 3.0 <= s["duration_s"] <= 12.0][:5]
for s in finals:
    audio, rate = sf.read(s["audio_path"], dtype="float32")
    frame = rate // 50
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(frame) / frame, mode="same"))
    last = int(np.nonzero(rms > 0.1 * rms.max())[0][-1])
    cut = audio[: last + 1]
    path = OUT / f"final_{s['utt_id']}.wav"
    write_wav(path, cut, rate)
    m["samples"].append(dict(s, utt_id=f"final_{s['utt_id']}", audio_path=str(path), subset="final-chunk",
                             duration_s=round(len(cut) / rate, 3), base_utt_id=s["utt_id"],
                             trimmed_s=round((len(audio) - len(cut)) / rate, 3)))
    used.add(s["utt_id"])

noise_path = ROOT / "High_school_cafeteria.ogg"
if not noise_path.exists():
    req = urllib.request.Request(NOISE["url"], headers={"User-Agent": "mlx-omarchy-corpus/1 (research)"})
    noise_path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
assert sha(noise_path, "sha1") == NOISE["commons_sha1"], "noise file does not match Commons sha1"
NOISE["sha256"] = sha(noise_path)
decoded = ROOT / "High_school_cafeteria.16k.wav"
if not decoded.exists():
    import subprocess
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(noise_path),
                    "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(decoded)],
                   check=True)
NOISE["decoded"] = {"file": decoded.name, "sha256": sha(decoded),
                    "command": "ffmpeg -i High_school_cafeteria.ogg -map 0:a:0 -ac 1 -ar 16000 -c:a pcm_s16le"}
noise, nrate = sf.read(str(decoded), dtype="float32")
assert nrate == 16_000 and noise.ndim == 1

m["samples"] = [s for s in m["samples"] if not s["subset"].startswith("mixed_")]
bases = [s for s in m["samples"] if s["subset"] == "test-clean" and s["duration_s"] < 20][:20]
for i, base in enumerate(bases):
    snr = 10 if i < 10 else 0
    speech, rate = sf.read(base["audio_path"], dtype="float32")
    start = int(nrng.integers(0, len(noise) - len(speech)))
    seg = noise[start: start + len(speech)]
    seg = seg * (np.sqrt(np.mean(speech ** 2)) / np.sqrt(np.mean(seg ** 2))) / 10 ** (snr / 20)
    mix = speech + seg
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    path = OUT / f"mixed_{snr}dB_{base['utt_id']}.wav"
    write_wav(path, mix, rate)
    m["samples"].append({"utt_id": f"mixed_{snr}dB_{base['utt_id']}", "audio_path": str(path),
                         "reference": base["reference"], "duration_s": base["duration_s"],
                         "sample_rate": rate, "channels": 1, "source": "librispeech+cafeteria",
                         "subset": f"mixed_{snr}dB", "base_utt_id": base["utt_id"], "snr_db": snr,
                         "noise_offset_samples": start})

for s in m["samples"]:
    s["sha256"] = sha(s["audio_path"])
m["noise_mix"] = dict(NOISE, method=__doc__.split("Mixing:")[1].strip(), rng_seed=20260930,
                      script="corpus/build_extend.py")
m["license"]["noise"] = NOISE["licence"]
m["source_urls"]["noise"] = NOISE["page"]
MANIFEST.write_text(json.dumps(m, indent=2))
counts = {}
for s in m["samples"]:
    counts[s["subset"]] = counts.get(s["subset"], 0) + 1
print(json.dumps({"total": len(m["samples"]), "counts": counts,
                  "manifest_sha256": sha(MANIFEST), "noise_sha256": NOISE["sha256"]}, indent=1))
