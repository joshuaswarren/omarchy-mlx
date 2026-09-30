# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU STT worker subprocess.

Owns one mlx-audio STT model resident on the live mlx backend. Framed
requests arrive on stdin (``<u64 header size><json header><payload>``)
and each gets exactly one framed response on stdout; ``close`` ends the
loop. The worker serves one request at a time. The parent cancels by
killing the process group, so there is no cancel frame. Resampling to
16 kHz runs on the mlx device inside the worker.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def _read_exact(stream, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf


def _send(stream, header: dict) -> None:
    encoded = json.dumps(dict(header, payload_bytes=0)).encode("utf-8")
    stream.write(struct.pack("<Q", len(encoded)) + encoded)
    stream.flush()


def _transcribe(model, payload: bytes, rate: int) -> dict:
    import mlx.core as mx

    from .gpu_stt import resample_to_16k

    try:
        samples = np.frombuffer(payload, dtype="<f4")
    except ValueError as exc:
        return {"ok": False, "kind": "input", "error": f"payload parse failed: {exc}"}
    try:
        # 1-D only: a (1, N) input makes Parakeet-TDT on mlx-audio 0.5.6
        # silently return "".
        audio = resample_to_16k(mx.array(samples), rate)
        out = model.generate(audio, verbose=False)
    except Exception as exc:
        return {"ok": False, "kind": "unavailable", "error": f"gpu-stt generate failed: {exc}"}
    return {"ok": True, "transcript": str(getattr(out, "text", out) or ""),
            "peak_memory_bytes": int(mx.get_peak_memory())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU STT worker")
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    try:
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model

        # A Path (not str) skips mlx-audio's hub resolution: the snapshot
        # directory is loaded as-is, with no network lookup.
        model = load_model(args.model_dir)
        # The first generate() after load returns "" on this stack; spend
        # it on one second of silence before declaring readiness.
        model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)
    except Exception as exc:
        _send(stdout, {"ok": False, "kind": "unavailable", "error": f"failed to load model: {exc}"})
        return 2
    _send(stdout, {"ok": True, "event": "ready", "model": type(model).__name__})

    while True:
        raw = _read_exact(stdin, 8)
        if len(raw) < 8:
            return 0
        (size,) = struct.unpack("<Q", raw)
        try:
            header = json.loads(_read_exact(stdin, size).decode("utf-8"))
        except ValueError as exc:
            _send(stdout, {"ok": False, "kind": "input", "error": f"invalid header JSON: {exc}"})
            continue
        payload_size = int(header.get("payload_bytes", 0))
        payload = _read_exact(stdin, payload_size) if payload_size else b""
        op, request_id = header.get("op"), header.get("id")
        if op == "close":
            _send(stdout, {"ok": True, "event": "closed", "id": request_id})
            return 0
        if op == "transcribe":
            response = _transcribe(model, payload, int(header.get("sample_rate", 16_000)))
        else:
            response = {"ok": False, "kind": "input", "error": f"unknown op {op!r}"}
        _send(stdout, dict(response, id=request_id))


if __name__ == "__main__":
    sys.exit(main())
