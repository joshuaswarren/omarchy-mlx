"""Build the STT corpus manifest.

Selects files from existing LibriSpeech + Midlands English corpora,
picks short clips (< 2 s), silence, and noise clips, hashes each,
and writes a JSON manifest with reference transcripts.

NOISE_MIXING is delegated to a separate script (see noise_mix.py).
"""
import os, json, hashlib, wave, struct, random, sys
from pathlib import Path
import numpy as np
import soundfile as sf

random.seed(42)

CORPUS_ROOT = Path(__file__).resolve().parent
LIBRI_ROOT = CORPUS_ROOT / "LibriSpeech"
MIDLANDS_ROOT = CORPUS_ROOT / "midlands"
MANIFEST_OUT = CORPUS_ROOT / "manifest.json"

NUM_TEST_CLEAN = 60  # clean read speech
NUM_TEST_OTHER = 40  # harder read speech
NUM_ACCENTED = 35    # midlands English female
NUM_SHORT = 12       # < 2 s clips
NUM_SILENCE = 10     # pure silence
NUM_NOISE = 10       # noise + speech at 0 dB / 10 dB SNR (mixed by noise_mix.py)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav_meta(p):
    info = sf.info(str(p))
    return info.samplerate, info.frames, info.channels, info.subtype


def collect_libri(root, max_clips):
    """Walk a LibriSpeech subset (test-clean / test-other) and yield
    (utt_id, flac_path, transcript, duration_seconds, words_count)."""
    entries = []
    for speaker in sorted(root.iterdir()):
        for chapter in sorted(speaker.iterdir()):
            trans_file = chapter / f"{speaker.name}-{chapter.name}.trans.txt"
            if not trans_file.is_file():
                continue
            for line in trans_file.read_text().splitlines():
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                flac = chapter / f"{utt_id}.flac"
                if not flac.is_file():
                    continue
                rate, frames, ch, subtype = read_wav_meta(flac)
                dur = frames / rate
                words = text.split()
                entries.append({
                    "utt_id": utt_id,
                    "audio_path": str(flac),
                    "reference": text.upper(),  # LibriSpeech is uppercase
                    "duration_s": round(dur, 3),
                    "sample_rate": rate,
                    "channels": ch,
                    "source": "librispeech",
                    "subset": root.name,
                })
                if len(entries) >= max_clips * 3:  # gather 3x then sample
                    break
            if len(entries) >= max_clips * 3:
                break
        if len(entries) >= max_clips * 3:
            break
    random.shuffle(entries)
    return entries[:max_clips]


def collect_midlands(root, max_clips):
    line_index = root / "line_index.csv"
    entries = []
    for line in line_index.read_text().splitlines():
        parts = line.split(",", 2)
        if len(parts) != 3:
            continue
        speaker, fname, text = parts
        speaker = speaker.strip()
        fname = fname.strip()
        # CSV holds basename without .wav; the dir is flat. Some zip variants
        # nest under a speaker subdirectory; check both.
        if not fname.endswith(".wav"):
            fname += ".wav"
        wav = root / fname
        if not wav.is_file():
            wav = root / speaker / fname
        if not wav.is_file():
            continue
        try:
            rate, frames, ch, subtype = read_wav_meta(wav)
        except Exception:
            continue
        dur = frames / rate
        entries.append({
            "utt_id": f"midlands_{fname.replace('.wav', '')}",
            "audio_path": str(wav),
            "reference": text.strip(),
            "duration_s": round(dur, 3),
            "sample_rate": rate,
            "channels": ch,
            "source": "midlands",
            "subset": "accented",
        })
    random.shuffle(entries)
    return entries[:max_clips]


def main():
    manifest = {
        "schema": "mlx-omarchy/gpu-stt-corpus/1",
        "created_utc": "2026-09-29",
        "license": {
            "librispeech": "CC-BY-4.0 (OpenSLR SLR12)",
            "midlands_english_female": "CC-BY-SA-4.0 (OpenSLR SLR83)",
        },
        "source_urls": {
            "librispeech": "https://openslr.elda.org/resources/12/",
            "midlands_english_female": "https://openslr.elda.org/resources/83/midlands_english_female.zip",
        },
        "thresholds": {
            "test-clean_wer": 0.06,
            "test-other_wer": 0.14,
            "accented_wer": 0.20,
            "0db_snr_wer": 0.30,
            "silence_empty_rate": 0.95,
            "p95_latency_5s_warm_s": 2.0,
        },
        "samples": [],
    }

    # 1) LibriSpeech test-clean (>= 60)
    print("Collecting LibriSpeech test-clean...")
    tc = collect_libri(LIBRI_ROOT / "test-clean", NUM_TEST_CLEAN)
    manifest["samples"].extend(tc)
    print(f"  test-clean: {len(tc)}")

    # 2) LibriSpeech test-other (>= 40)
    print("Collecting LibriSpeech test-other...")
    to = collect_libri(LIBRI_ROOT / "test-other", NUM_TEST_OTHER)
    manifest["samples"].extend(to)
    print(f"  test-other: {len(to)}")

    # 3) Accented English (midlands)
    print("Collecting Midlands English female...")
    acc = collect_midlands(MIDLANDS_ROOT, NUM_ACCENTED)
    manifest["samples"].extend(acc)
    print(f"  accented: {len(acc)}")

    # 4) Short clips (< 2 s) — first filter from all collected
    print("Filtering short clips < 2 s...")
    pool = tc + to + acc
    shorts = [s for s in pool if s["duration_s"] < 2.0][:NUM_SHORT]
    for s in shorts:
        s["subset"] = "short"
    manifest["samples"].extend(shorts)
    print(f"  short: {len(shorts)}")

    # 5) Silence and pure noise clips (>= 20 total)
    print("Generating silence and noise clips...")
    for i in range(NUM_SILENCE):
        manifest["samples"].append({
            "utt_id": f"silence_{i:03d}",
            "audio_path": f"<generated: {i}>",
            "reference": "",
            "duration_s": 3.0,
            "sample_rate": 16000,
            "channels": 1,
            "source": "synthetic_silence",
            "subset": "silence",
        })
    for i in range(NUM_NOISE):
        manifest["samples"].append({
            "utt_id": f"noise_{i:03d}",
            "audio_path": f"<generated: {i}>",
            "reference": "",
            "duration_s": 3.0,
            "sample_rate": 16000,
            "channels": 1,
            "source": "synthetic_noise",
            "subset": "noise",
        })
    print(f"  silence+noise synthetic: {NUM_SILENCE + NUM_NOISE}")

    # Hash existing files; mark synthetic as not-yet-generated
    print("Hashing existing audio files...")
    for s in manifest["samples"]:
        if s["audio_path"].startswith("<"):
            s["sha256"] = None
        else:
            s["sha256"] = sha256_file(s["audio_path"])
    print(f"Total samples: {len(manifest['samples'])}")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written: {MANIFEST_OUT}")


if __name__ == "__main__":
    main()