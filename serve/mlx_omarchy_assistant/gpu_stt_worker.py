# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU STT worker subprocess.

Owns one mlx-audio STT model loaded on the live mlx Vulkan backend.
Reads framed requests on stdin (``<u64 header_bytes><json header><bytes>
payload``), runs ``model.generate`` on the GPU, writes a framed
response to stdout, and exits cleanly on ``close`` / cancel / deadline.
The model is resident across requests so warm latencies are tight
and the parent can confirm its exit before reporting a failure.

The worker NEVER does CPU tensor inference — every dispatch goes
through the live mlx binary, and the dispatch trace mode
(``MLX_OMARCHY_TRACE_DISPATCH``) records each GPU compute submission
so a zero-CPU-tensor-dispatch assertion is verifiable on a real run.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from typing import Any

import numpy as np


def _read_exact(stream, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf


def _send(stream, header: dict, payload: bytes = b"") -> None:
    header = dict(header)
    header["payload_bytes"] = len(payload)
    encoded = json.dumps(header).encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)) + encoded + payload)
    stream.flush()


def _load_model(model_dir: str, model_id: str):
    """Load the pinned model from its resolved HF cache directory.

    ``model_dir`` is the resolved snapshot path (used for files like
    tokenizer.model that live alongside the weights); ``model_id`` is
    the repo slug (``mlx-community/parakeet-tdt-0.6b-v3``) that
    ``load_model`` recognises for the type-detection table.
    """
    from mlx_audio.stt.utils import load_model
    return load_model(model_id)


def _run(model, samples: np.ndarray, deadline_ms: int,
         cancel: threading.Event) -> dict:
    import mlx.core as mx
    audio = mx.array(np.ascontiguousarray(samples, dtype=np.float32))
    # Parakeet-TDT expects a 1-D waveform at 16 kHz; passing a 2-D
    # (1, N) view silently returns "" on mlx-audio 0.5.6. Whisper
    # accepts both shapes, but feeding it 1-D works too. Stay 1-D.
    mx_audio = audio
    deadline = time.monotonic() + deadline_ms / 1000.0
    cancel_event = threading.Event()
    cancel_thread: threading.Thread | None = None
    if cancel is not None:
        def watch():
            cancel.wait()
            cancel_event.set()
        cancel_thread = threading.Thread(target=watch, daemon=True)
        cancel_thread.start()
    try:
        if cancel_event.is_set():
            return {"ok": False, "kind": "cancelled",
                    "error": "cancelled before transcription"}
        # Whisper accepts language; Parakeet ignores it (and emitting it
        # empty-stubs the output on the mlx-audio 0.5.6 STT path). Pass
        # only the kwargs the model's signature actually wants.
        kwargs: dict[str, Any] = {"verbose": False}
        try:
            import inspect
            params = inspect.signature(model.generate).parameters
            if "language" in params and "Whisper" in type(model).__name__:
                kwargs["language"] = "en"
        except Exception:
            pass
        out = model.generate(mx_audio, **kwargs)
        if cancel_event.is_set():
            return {"ok": False, "kind": "cancelled",
                    "error": "cancelled during transcription"}
        if time.monotonic() > deadline:
            return {"ok": False, "kind": "timeout",
                    "error": "transcription exceeded deadline"}
        # mlx-audio STT returns either a string or an STTOutput with a
        # .text attribute. Both code paths normalize to str here.
        if isinstance(out, str):
            text = out
        else:
            text = getattr(out, "text", "")
        if text is None:
            text = ""
        return {"ok": True, "transcript": str(text),
                "model": type(model).__name__,
                "segments": len(getattr(out, "segments", []) or [])
                              if not isinstance(out, str) else 0}
    except Exception as exc:
        return {"ok": False, "kind": "unavailable",
                "error": f"gpu-stt generate failed: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU STT worker")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args(argv)
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    try:
        model = _load_model(args.model_dir, args.model_id)
    except Exception as exc:
        _send(stdout, {"ok": False, "kind": "unavailable",
                       "error": f"failed to load model: {exc}"})
        return 2
    _send(stdout, {"ok": True, "event": "ready",
                   "model_id": args.model_id,
                   "model": type(model).__name__})

    cancel = threading.Event()

    for raw in iter(lambda: _read_exact(stdin, 8), b""):
        (header_size,) = struct.unpack("<Q", raw)
        header_bytes = _read_exact(stdin, header_size)
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except Exception as exc:
            _send(stdout, {"ok": False, "kind": "input",
                           "error": f"invalid header JSON: {exc}"})
            continue
        op = header.get("op")
        payload_size = int(header.get("payload_bytes", 0))
        payload = _read_exact(stdin, payload_size) if payload_size else b""
        request_id = header.get("id")
        if op == "close":
            _send(stdout, {"ok": True, "event": "closed", "id": request_id})
            break
        if op == "cancel":
            cancel.set()
            continue
        if op == "warmup":
            try:
                import mlx.core as mx
                _ = model.generate(
                    mx.zeros((16000 * 1,), dtype=mx.float32),
                    verbose=False,
                )
            except Exception as exc:
                _send(stdout, {"ok": False, "kind": "unavailable",
                               "error": f"warmup failed: {exc}",
                               "id": request_id})
                continue
            _send(stdout, {"ok": True, "event": "warmed", "id": request_id})
            continue
        if op == "transcribe":
            cancel.clear()
            try:
                samples = np.frombuffer(payload, dtype="<f4").astype(np.float32)
            except Exception as exc:
                _send(stdout, {"ok": False, "kind": "input",
                               "error": f"payload parse failed: {exc}",
                               "id": request_id})
                continue
            deadline_ms = int(header.get("deadline_ms", 90_000))
            response = _run(model, samples, deadline_ms, cancel)
            response["id"] = request_id
            _send(stdout, response)
            continue
        _send(stdout, {"ok": False, "kind": "input",
                       "error": f"unknown op {op!r}", "id": request_id})
    return 0


if __name__ == "__main__":
    sys.exit(main())