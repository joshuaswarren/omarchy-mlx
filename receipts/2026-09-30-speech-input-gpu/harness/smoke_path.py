"""Subprocess-path smoke: Recognition.transcribe -> owned GPU worker, on the M2."""
import json, os, struct, sys, threading, time
from pathlib import Path

import numpy as np
import soundfile as sf

from mlx_omarchy_assistant import gpu_stt, recognition

HOME = Path("<home>/agents/SpeechInputGpu/home")
CLIP = "<home>/agents/SpeechInputGpu/corpus/LibriSpeech/test-clean/1320/122617/1320-122617-0000.flac"
REF = "NOTWITHSTANDING THE HIGH RESOLUTION OF HAWKEYE HE FULLY COMPREHENDED ALL THE DIFFICULTIES AND DANGER HE WAS ABOUT TO INCUR"
out = {}


def wav(samples, rate, tag=1, bits=16):
    if tag == 3:
        data = np.asarray(samples, "<f4").tobytes()
    elif bits == 16:
        data = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    else:
        data = bytes(samples)
    width = bits // 8
    fmt = struct.pack("<HHIIHH", tag, 1, rate, rate * width, width, bits)
    body = b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def norm(t):
    return " ".join("".join(c if c.isalnum() else " " for c in t.upper()).split())


def expect_error(name, cls, payload):
    t0 = time.monotonic()
    try:
        rec.transcribe(payload)
        out[name] = {"pass": False, "error": "no refusal"}
    except cls as exc:
        out[name] = {"pass": True, "refusal": str(exc), "ms": round((time.monotonic() - t0) * 1000, 1)}


out["prepare_without_approval"] = gpu_stt.prepare(HOME, approve_download=False)
rec = recognition.Recognition(HOME)
out["status_before"] = {k: rec.status()[k] for k in ("state", "reasons", "model", "memory")}
audio, rate = sf.read(CLIP, dtype="float32")

t0 = time.monotonic()
first = rec.transcribe(wav(audio, rate))
out["first_call"] = {"ms": round((time.monotonic() - t0) * 1000, 1), "text": first,
                     "matches_reference": norm(first) == REF}
t = np.arange(len(audio) * 3) / 48_000
up48 = np.interp(np.arange(len(audio) * 3) / 3, np.arange(len(audio)), audio).astype(np.float32)
t0 = time.monotonic()
text48 = rec.transcribe(wav(up48, 48_000))
out["input_48k"] = {"ms": round((time.monotonic() - t0) * 1000, 1), "text": text48,
                    "matches_reference": norm(text48) == REF}
t0 = time.monotonic()
out["silence_3s"] = {"text": rec.transcribe(wav(np.zeros(48_000), 16_000)),
                     "ms": round((time.monotonic() - t0) * 1000, 1)}
at_limit = np.tile(audio, 4)[: 30 * 16_000]
t0 = time.monotonic()
text30 = rec.transcribe(wav(at_limit, 16_000))
out["exactly_30s"] = {"ms": round((time.monotonic() - t0) * 1000, 1), "words": len(text30.split())}
expect_error("over_30s", recognition.RecognitionInputError, wav(np.zeros(30 * 16_000 + 1600), 16_000))
expect_error("oversize_payload", recognition.RecognitionInputError, b"RIFF" + b"\0" * (49 * 1024 * 1024))
expect_error("nan_float", recognition.RecognitionInputError, wav(np.array([0.1, np.nan, 0.2] * 100, np.float32), 16_000, tag=3, bits=32))
expect_error("rate_zero", recognition.RecognitionInputError, wav(np.zeros(160), 0))
expect_error("rate_500k", recognition.RecognitionInputError, wav(np.zeros(1600), 500_000))
expect_error("pcm_8bit", recognition.RecognitionInputError, wav(np.zeros(160, np.uint8), 16_000, bits=8))

worker = rec._worker
pid = worker._process.pid
cancel = threading.Event()
threading.Timer(0.05, cancel.set).start()
t0 = time.monotonic()
try:
    rec.transcribe(wav(at_limit, 16_000), cancel)
    out["cancel_mid_request"] = {"pass": False, "error": "returned a transcript"}
except recognition.RecognitionCancelled as exc:
    time.sleep(0.2)
    out["cancel_mid_request"] = {"pass": True, "refusal": str(exc),
                                 "ms_to_raise": round((time.monotonic() - t0) * 1000, 1),
                                 "worker_pid": pid, "worker_alive": worker.alive,
                                 "process_group_alive": worker._group_alive()}
t0 = time.monotonic()
after = rec.transcribe(wav(audio, rate))
out["after_cancel"] = {"ms": round((time.monotonic() - t0) * 1000, 1),
                       "matches_reference": norm(after) == REF,
                       "new_worker_pid": rec._worker._process.pid}
rec.close()
out["closed_worker_exit_code"] = rec._worker is None
print(json.dumps(out, indent=1))
