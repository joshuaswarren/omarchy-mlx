# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU STT worker subprocess.

Owns one mlx-audio STT model resident on the live mlx backend. Framed
requests arrive on stdin (``<u64 header size><json header><payload>``)
and each gets exactly one framed response on stdout; ``close`` ends the
loop. The worker serves one request at a time. The parent cancels by
killing the process group, so there is no cancel frame. Resampling to
16 kHz runs on the mlx device inside the worker.

Empty-transcript retry: Parakeet-TDT returned no transcript for 3 of 96
browser recordings of clear speech, and NeMo's own implementation does the
same on them. When a decode comes back empty and the clip holds at least
0.5 s of voiced frames, it is decoded once more with 1 s of fixed low-level
noise on both ends, which recovers those recordings. Padding every request
instead was measured and rejected: it emptied tiled 20-30 s clips that
decode fully without it. Silence and stationary noise have no voiced frames,
so they never retry.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
RETRY_PAD_SECONDS = 1.0
MIN_VOICED_SECONDS = 0.5
_FRAME = 320  # 20 ms
_VOICED_RMS_FLOOR = 0.005  # the browser recorder's no-speech level
_VOICED_OVER_BACKGROUND = 4.0  # 12 dB above the clip's 10th-percentile frame
_PAD_NOISE = (np.random.default_rng(0).standard_normal(int(RETRY_PAD_SECONDS * SAMPLE_RATE))
              .astype(np.float32) * 1e-4)  # about -80 dBFS


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


def _register_bare_package(name: str) -> None:
    """Put package ``name`` in sys.modules without running its __init__.

    Its submodules still import normally. mlx_audio.stt.models/__init__
    imports every model family, and granite_speech5_ctc builds a float64 mel
    filterbank on the CPU stream at import: 152 CPU-stream dispatches in
    every worker, none of them for Parakeet.
    """
    if name in sys.modules:
        return
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ModuleNotFoundError(name)
    sys.modules[name] = importlib.util.module_from_spec(spec)


STAGES = ("load", "resample", "mel", "encoder", "decoder", "joint", "post")


def _mx():
    import mlx.core as mx

    return mx


class StageClock:
    """Per-stage millisecond accumulators for one transcribe request.

    Plain floats and one dict, so the clocks add nothing to the mlx graph
    and cost nothing when the response is built.
    """

    def __init__(self):
        self.ms = {name: 0.0 for name in STAGES}

    def reset(self):
        for name in self.ms:
            self.ms[name] = 0.0

    def add(self, stage: str, seconds: float):
        self.ms[stage] += seconds * 1000.0


class _StageProxy:
    """Route calls to a model seam through the clock; delegate the rest.

    Assigned with ``object.__setattr__`` so mlx's ``Module.__setattr__``
    never registers the proxy as a child module. Shortcut: the proxies sit
    in the instance dict, so mlx module walks no longer see encoder or
    joint; fine while this worker only infers.
    """

    def __init__(self, clock: StageClock, stage: str, target):
        self._clock = clock
        self._stage = stage
        self._target = target

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            out = self._target(*args, **kwargs)
            if self._stage == "encoder":
                # The conformer call is lazy; its compute lands in the
                # mx.eval decode() issues right after, so the eval happens
                # inside the timed window instead.
                mx = _mx()
                outputs = out if isinstance(out, tuple) else (out,)
                mx.eval(*[a for a in outputs if isinstance(a, mx.array)])
            return out
        finally:
            self._clock.add(self._stage, time.perf_counter() - started)

    def __getattr__(self, name):
        return getattr(self._target, name)


def instrument(model, clock: StageClock):
    """Wrap the pipeline seams once so each stage accumulates ms.

    Seams: ``_log_mel_spectrogram`` (mel), ``encoder`` (the conformer),
    ``_tdt_step`` (the decoder loop), ``joint``. A model without a seam
    keeps the stages it has; resample and post are timed in ``_transcribe``
    either way.
    """
    for attr, stage in (("_log_mel_spectrogram", "mel"), ("encoder", "encoder"),
                        ("_tdt_step", "decoder"), ("joint", "joint")):
        target = getattr(model, attr, None)
        if target is None or isinstance(target, _StageProxy):
            continue
        object.__setattr__(model, attr, _StageProxy(clock, stage, target))
    object.__setattr__(model, "_stage_clock", clock)
    return model


def voiced_seconds(xp, audio) -> float:
    """Seconds of 20 ms frames louder than both the recorder's no-speech
    level and 12 dB over the clip's own background (10th-percentile frame).
    ``xp`` is numpy or mlx.core; with mlx the work stays on the device."""
    count = audio.shape[0] // _FRAME
    if count == 0:
        return 0.0
    frames = audio[: count * _FRAME].reshape(count, _FRAME)
    rms = xp.sqrt((frames * frames).mean(axis=1))
    background = xp.sort(rms)[int(0.1 * (count - 1))]
    threshold = xp.maximum(background * _VOICED_OVER_BACKGROUND, _VOICED_RMS_FLOOR)
    return float((rms > threshold).sum()) * _FRAME / SAMPLE_RATE


def decode_with_retry(generate, xp, audio, noise) -> str:
    """``generate(audio) -> str``, retried once with padded edges when it is
    empty but voiced (see the module docstring)."""
    text = generate(audio)
    if text or voiced_seconds(xp, audio) < MIN_VOICED_SECONDS:
        return text
    n = int(RETRY_PAD_SECONDS * SAMPLE_RATE)
    return generate(xp.concatenate([noise[:n], audio, noise[-n:]]))


def _transcribe(model, payload: bytes, rate: int, noise,
                load_ms: float | None = None) -> dict:
    import mlx.core as mx

    from .gpu_stt import resample_to_16k

    try:
        samples = np.frombuffer(payload, dtype="<f4")
    except ValueError as exc:
        return {"ok": False, "kind": "input", "error": f"payload parse failed: {exc}"}

    # 1-D only: a (1, N) input makes Parakeet-TDT on mlx-audio 0.5.6
    # silently return "".
    def generate(audio) -> str:
        return str(getattr(model.generate(audio, verbose=False), "text", "") or "")

    clock = getattr(model, "_stage_clock", None)
    if clock is None or getattr(clock, "ms", None) is None:
        instrument(model, StageClock())
        clock = model._stage_clock
    clock.reset()
    try:
        started = time.perf_counter()
        audio = resample_to_16k(mx.array(samples), rate)
        clock.add("resample", time.perf_counter() - started)
        text = decode_with_retry(generate, mx, audio, noise)
        started = time.perf_counter()
        text = str(text)
        clock.add("post", time.perf_counter() - started)
    except Exception as exc:
        return {"ok": False, "kind": "unavailable", "error": f"gpu-stt generate failed: {exc}"}
    response = {"ok": True, "transcript": text,
                "peak_memory_bytes": int(mx.get_peak_memory())}
    if load_ms is not None:
        response["load_ms"] = float(load_ms)
    response["stage_ms"] = {name: round(ms, 3) for name, ms in clock.ms.items()}
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU STT worker")
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    try:
        import mlx.core as mx
        _register_bare_package("mlx_audio.stt.models")
        from mlx_audio.stt.utils import load_model

        # A Path (not str) skips mlx-audio's hub resolution: the snapshot
        # directory is loaded as-is, with no network lookup.
        load_started = time.perf_counter()
        model = load_model(args.model_dir)
        load_ms = (time.perf_counter() - load_started) * 1000.0
        noise = mx.array(_PAD_NOISE)
        # The first generate() after load returns "" on this stack; spend
        # it on one second of silence before declaring readiness (untimed:
        # a workaround, not a pipeline stage).
        model.generate(mx.zeros((16_000,), dtype=mx.float32), verbose=False)
    except Exception as exc:
        _send(stdout, {"ok": False, "kind": "unavailable", "error": f"failed to load model: {exc}"})
        return 2
    instrument(model, StageClock())
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
            response = _transcribe(model, payload, int(header.get("sample_rate", 16_000)), noise)
        else:
            response = {"ok": False, "kind": "input", "error": f"unknown op {op!r}"}
        _send(stdout, dict(response, id=request_id))


if __name__ == "__main__":
    sys.exit(main())
