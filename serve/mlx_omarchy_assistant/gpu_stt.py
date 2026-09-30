# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU speech recognition for the offline assistant (mlx Vulkan backend).

The recognition backend for Apple Silicon Linux hosts without a live
ANE. An owned worker subprocess (``gpu_stt_worker``) keeps the pinned
Parakeet-TDT model resident on the mlx device, resamples to 16 kHz there
and runs mlx-audio's ``generate``; cancellation and wedges end with a
confirmed process-group kill.

The model is a pinned file set downloaded approve-first into
``<home>/voice/<id>/`` and sha256-verified, like the TTS pack.
``ready`` needs an acceptance receipt that names this process's mlx
backend binary and the verified model hash, and records every frozen
threshold in ACCEPTANCE_THRESHOLDS as passed.
"""

from __future__ import annotations

import functools
import importlib.metadata
import json
import os
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .recognition import (
    RecognitionCancelled,
    RecognitionInputError,
    RecognitionTimedOut,
    RecognitionUnavailable,
)

TARGET_SAMPLE_RATE = 16_000
CANCEL_GRACE_SECONDS = 5.0
BACKEND_KIND = "gpu"
MLX_AUDIO_VERSION = "0.5.6"

GPU_STT_MODEL = {
    "id": "parakeet-tdt-0.6b-v3",
    "repo": "mlx-community/parakeet-tdt-0.6b-v3",
    "revision": "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    "license": "cc-by-4.0",
    "weights_bytes": 2_508_288_736,
    "files": [
        {"name": "config.json", "bytes": 244_093,
         "sha256": "f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c"},
        {"name": "model.safetensors", "bytes": 2_508_288_736,
         "sha256": "05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592"},
        {"name": "tokenizer.model", "bytes": 360_916,
         "sha256": "eacec2b0a77f336d4a2ca4a25a7047575d3c2b74de47e997f4c205126ed3135e"},
        {"name": "tokenizer.vocab", "bytes": 101_024,
         "sha256": "41130ff456706304a1adec782ccc9e003c4d417e8e324353d281be958cac4e17"},
        {"name": "vocab.txt", "bytes": 46_772,
         "sha256": "3cde1409fd78783a79b29ed4d32da57c746993856f7c8263bcb905d2e5839db7"},
    ],
}


def model_dir(home: Path) -> Path:
    return Path(home) / "voice" / GPU_STT_MODEL["id"]


def _file_status(directory: Path) -> tuple[bool, list[str]]:
    """(all present with pinned sizes, names missing or wrong-sized)."""
    bad = [e["name"] for e in GPU_STT_MODEL["files"]
           if not (directory / e["name"]).is_file()
           or (directory / e["name"]).stat().st_size != e["bytes"]]
    return not bad, bad


@functools.cache
def _verified(directory: Path, identities: tuple) -> bool:
    """sha256 of every pinned file; cached per (path, size/mtime/ctime/inode)."""
    from .synthesis import _sha256_file
    return all(_sha256_file(directory / e["name"]) == e["sha256"]
               for e in GPU_STT_MODEL["files"])


def model_verified(home: Path) -> bool:
    directory = model_dir(home)
    present, _ = _file_status(directory)
    if not present:
        return False
    identities = tuple(
        (s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)
        for s in ((directory / e["name"]).stat() for e in GPU_STT_MODEL["files"]))
    return _verified(directory, identities)


def prepare(home: Path, approve_download: bool, *, fetch=None) -> dict:
    """Download the pinned model with explicit approval; verify every file."""
    from .synthesis import _URL_TEMPLATE, _default_fetch, _sha256_file

    directory = model_dir(home)
    if model_verified(home):
        return {"downloaded": True, "verified": True, "path": str(directory)}
    if not approve_download:
        return {"downloaded": False, "verified": False, "path": str(directory),
                "reason": ("speech recognition model download requires explicit "
                           f"approval ({GPU_STT_MODEL['weights_bytes']} bytes, "
                           f"{GPU_STT_MODEL['license']} license, "
                           f"{GPU_STT_MODEL['repo']} revision {GPU_STT_MODEL['revision']})")}
    directory.mkdir(parents=True, exist_ok=True)
    for entry in sorted(GPU_STT_MODEL["files"], key=lambda e: e["bytes"]):
        dest = directory / entry["name"]
        if dest.is_file() and dest.stat().st_size == entry["bytes"] \
                and _sha256_file(dest) == entry["sha256"]:
            continue
        tmp = dest.with_name(dest.name + ".part")
        try:
            (fetch or _default_fetch)(_URL_TEMPLATE.format(
                repo=GPU_STT_MODEL["repo"], revision=GPU_STT_MODEL["revision"],
                name=entry["name"]), tmp)
            actual = _sha256_file(tmp)
            if actual != entry["sha256"]:
                raise RecognitionUnavailable(
                    f"speech model file {entry['name']}: sha256 mismatch "
                    f"(expected {entry['sha256']}, got {actual})")
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
    if not model_verified(home):
        raise RecognitionUnavailable("speech recognition model failed verification")
    return {"downloaded": True, "verified": True, "path": str(directory)}


def probe_runtime(home: Path, verify_cache: bool = False) -> dict:
    """Named preflight facts. Never promotes to ``ready``: only a receipt does."""
    reasons: list[str] = []
    facts: dict = {"kind": "gpu_stt",
                   "model": {"repo": GPU_STT_MODEL["repo"],
                             "revision": GPU_STT_MODEL["revision"]},
                   "asset_bytes": {"weights_bytes": GPU_STT_MODEL["weights_bytes"]}}
    try:
        version = importlib.metadata.version("mlx-audio")
    except importlib.metadata.PackageNotFoundError:
        version = None
    facts["mlx_audio"] = version
    if version != MLX_AUDIO_VERSION:
        reasons.append(f"mlx-audio {MLX_AUDIO_VERSION} is required (found {version}); "
                       "install with bash install.sh --voice")
    try:
        import mlx.core as mx
        device = str(mx.default_device())
    except Exception as exc:
        device = None
        reasons.append(f"mlx is not importable: {exc}")
    facts["device"] = device
    if device is not None and "gpu" not in device:
        reasons.append(f"GPU unavailable: default device is {device}")
    present, missing = _file_status(model_dir(home))
    if not present:
        facts["cache"] = "missing"
        reasons.append("speech recognition model not downloaded (missing: "
                       + ", ".join(missing) + "); enable voice in model setup")
    elif verify_cache and not model_verified(home):
        facts["cache"] = "invalid"
        reasons.append("speech recognition model failed sha256 verification")
    else:
        facts["cache"] = "verified" if verify_cache else "present"
    return {"ok": not reasons, "reasons": reasons, "facts": facts}


def resample_to_16k(samples, rate: int):
    """Resample a mono mx.array to 16 kHz on the GPU; returns an mx.array.

    Kaiser-windowed sinc polyphase filter (the same design the ANE path
    uses). The taps are host-designed constants; every sample operation
    runs on the mlx device.
    """
    import math

    import mlx.core as mx

    if rate == TARGET_SAMPLE_RATE:
        return samples
    if rate <= 0:
        raise RecognitionInputError(f"invalid source rate {rate}")
    divisor = math.gcd(int(rate), TARGET_SAMPLE_RATE)
    up, down = TARGET_SAMPLE_RATE // divisor, int(rate) // divisor
    half = 16 * up
    n = np.arange(-half, half + 1, dtype=np.float64)
    taps = np.sinc(n / max(up, down)) * np.kaiser(n.size, 8.0)
    taps = (taps / taps.sum()).astype(np.float32)
    rows = -(-taps.size // up)
    padded = np.zeros(rows * up, dtype=np.float32)
    padded[: taps.size] = taps
    centre = (taps.size - 1) // 2
    count = samples.shape[0]
    n_out = -(-count * up // down)
    signal = mx.pad(samples, [(rows, rows)])
    m = mx.arange(n_out) * down + centre
    phase, base = m % up, m // up
    offsets = mx.arange(rows)
    gathered = mx.take(signal, (base[:, None] - offsets[None, :] + rows).reshape(-1))
    weights = mx.take(mx.array(padded), (offsets[None, :] * up + phase[:, None]).reshape(-1))
    return (gathered.reshape(n_out, rows) * weights.reshape(n_out, rows)).sum(axis=1)


def _receipt_path(home: Path) -> Path:
    return model_dir(home) / "qualification.json"


ACCEPTANCE_THRESHOLDS = {
    "wer": {"test-clean": 0.06, "test-other": 0.14, "accented": 0.20,
            "mixed_0dB": 0.30},
    "empty_rate_min": {"silence": 0.95, "noise": 0.95},
    "latency_p95_ms_max": 2000.0,
}


@functools.cache
def _backend_identity() -> tuple[str | None, str | None]:
    """(identity, detail) of the loaded mlx extension + libmlx.so; fixed per process."""
    from .synthesis import _backend_provenance
    backend = _backend_provenance()
    return (backend.get("identity") if backend.get("verified") == "match" else None,
            backend.get("detail"))


def _runtime_identity(home: Path) -> dict:
    backend, detail = _backend_identity()
    weights = next(e for e in GPU_STT_MODEL["files"] if e["name"] == "model.safetensors")
    return {"mlx_backend": backend, "mlx_backend_detail": detail,
            "model_sha256": weights["sha256"] if model_verified(home) else None}


def read_acceptance_receipt(home: Path) -> dict | None:
    """The receipt, only while it names THIS runtime's backend and model."""
    try:
        receipt = json.loads(_receipt_path(home).read_text())
    except (OSError, ValueError):
        return None
    identity = _runtime_identity(home)
    if (not isinstance(receipt, dict)
            or identity["mlx_backend"] is None
            or identity["model_sha256"] is None
            or receipt.get("mlx_backend") != identity["mlx_backend"]
            or receipt.get("model_sha256") != identity["model_sha256"]):
        return None
    return receipt


def write_acceptance_receipt(home: Path, *, wer_by_subset: dict,
                             empty_rate_by_subset: dict, latency_ms: dict,
                             cpu_tensor_events: int, receipt_source: str,
                             host: str) -> dict:
    """Record a passed hardware acceptance run for the running runtime.

    Refuses unless every frozen threshold in ACCEPTANCE_THRESHOLDS passes
    and the CPU tensor event count is zero. The backend identity and model
    hash are computed here, never taken from the caller.
    """
    if cpu_tensor_events != 0:
        raise ValueError(f"acceptance refused: cpu_tensor_events={cpu_tensor_events}")
    failures = []
    for subset, limit in ACCEPTANCE_THRESHOLDS["wer"].items():
        value = wer_by_subset.get(subset)
        if value is None or value > limit:
            failures.append(f"{subset} WER {value} > {limit}")
    for subset, floor in ACCEPTANCE_THRESHOLDS["empty_rate_min"].items():
        value = empty_rate_by_subset.get(subset)
        if value is None or value < floor:
            failures.append(f"{subset} empty rate {value} < {floor}")
    p95 = latency_ms.get("p95")
    if p95 is None or p95 > ACCEPTANCE_THRESHOLDS["latency_p95_ms_max"]:
        failures.append(f"latency p95 {p95} ms > {ACCEPTANCE_THRESHOLDS['latency_p95_ms_max']}")
    if failures:
        raise ValueError("acceptance refused: " + "; ".join(failures))
    identity = _runtime_identity(home)
    if identity["mlx_backend"] is None or identity["model_sha256"] is None:
        raise ValueError("acceptance refused: runtime identity unverified "
                         f"({identity['mlx_backend_detail']})")
    receipt = {
        "schema": "mlx-omarchy/gpu-stt-acceptance/2",
        "recorded_at": time.time(),
        "host": host,
        "model": GPU_STT_MODEL["id"],
        "model_sha256": identity["model_sha256"],
        "mlx_backend": identity["mlx_backend"],
        "wer_by_subset": wer_by_subset,
        "empty_rate_by_subset": empty_rate_by_subset,
        "latency_ms": latency_ms,
        "cpu_tensor_events": cpu_tensor_events,
        "receipt_source": receipt_source,
    }
    path = _receipt_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return receipt


def _worker_error(response: dict) -> RuntimeError:
    message = response.get("error", "gpu-stt worker failed without a reason")
    if response.get("kind") == "input":
        return RecognitionInputError(message)
    return RecognitionUnavailable(message)


class _WorkerHandle:
    """Owned GPU STT subprocess; model resident, IPC-framed, cancel confirms."""

    def __init__(self, model_dir: Path):
        env = dict(os.environ, HF_HUB_OFFLINE="1")
        # ``python -m mlx_omarchy_assistant.gpu_stt_worker`` needs the directory
        # that holds this package, wherever the assistant was installed.
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (package_root, env.get("PYTHONPATH", "")) if p)
        self._process = subprocess.Popen(
            [sys.executable, "-m", "mlx_omarchy_assistant.gpu_stt_worker",
             "--model-dir", str(model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._pgid = os.getpgid(self._process.pid)
        self._send_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._ready_lock = threading.Lock()
        self._ready_seen = False
        self._request_id = 0

    def _drain_stderr(self) -> None:
        proc = self._process
        assert proc.stderr is not None
        try:
            for line in iter(proc.stderr.readline, b""):
                sys.stderr.write("[gpu-stt] " + line.decode("utf-8", "replace"))
        except Exception:
            return

    @property
    def alive(self) -> bool:
        return self._process.poll() is None

    def _send(self, header: dict, payload: bytes = b"") -> None:
        header = dict(header)
        header["payload_bytes"] = len(payload)
        encoded = json.dumps(header).encode("utf-8")
        with self._send_lock:
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(
                    struct.pack("<Q", len(encoded)) + encoded + payload
                )
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                raise RecognitionUnavailable(
                    "the gpu-stt worker exited unexpectedly (exit code "
                    f"{self._process.poll()}; its stderr is prefixed [gpu-stt])"
                ) from exc

    def request(self, header: dict, payload: bytes = b"", *,
                timeout: float, cancel=None) -> dict:
        """One framed round trip. Any cancel outcome raises RecognitionCancelled.

        generate() cannot be interrupted, so a cancel kills the worker's
        process group at once and confirms it is gone; the next request
        starts a fresh worker.
        """
        self._request_id += 1
        done = threading.Event()

        def watch_cancel():
            while not done.is_set():
                if cancel.wait(0.05):
                    if not done.is_set():
                        self.confirm_stopped()
                    return

        watcher = None
        if cancel is not None:
            watcher = threading.Thread(target=watch_cancel, daemon=True)
            watcher.start()
        try:
            with self._ready_lock:
                # The boot "ready" frame must be consumed before the first
                # request; reading it as a response shifted every transcript.
                if not self._ready_seen:
                    ready = self._read_response(max(timeout, 120.0))
                    if ready.get("event") != "ready":
                        raise _worker_error(ready)
                    self._ready_seen = True
            self._send(dict(header, id=self._request_id), payload)
            response = self._read_response(timeout)
        except (RecognitionUnavailable, RecognitionTimedOut) as exc:
            if cancel is not None and cancel.is_set():
                raise RecognitionCancelled("cancelled during transcription") from exc
            raise
        finally:
            done.set()
            if watcher is not None:
                watcher.join(timeout=CANCEL_GRACE_SECONDS + 1.0)
        if cancel is not None and cancel.is_set():
            raise RecognitionCancelled("cancelled during transcription")
        if not response.get("ok"):
            raise _worker_error(response)
        return response

    def _read_response(self, timeout: float) -> dict:
        process = self._process
        assert process.stdout is not None
        deadline = time.monotonic() + timeout
        raw_header = b""
        while len(raw_header) < 8:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.alive:
                break
            byte = process.stdout.read(8 - len(raw_header))
            if byte:
                raw_header += byte
        if len(raw_header) < 8:
            self.confirm_stopped()
            if process.poll() not in (None, 0):
                raise RecognitionUnavailable(
                    "the gpu-stt worker exited while a request was in "
                    f"flight (exit code {process.poll()})"
                )
            raise RecognitionTimedOut(
                f"the gpu-stt worker did not answer within {timeout:.0f} s "
                "and was stopped (exit confirmed)"
            )
        (header_size,) = struct.unpack("<Q", raw_header)
        header_bytes = process.stdout.read(header_size)
        response = json.loads(header_bytes.decode("utf-8"))
        payload_size = int(response.get("payload_bytes", 0))
        if payload_size:
            process.stdout.read(payload_size)
        return response

    def _group_alive(self) -> bool:
        try:
            os.killpg(self._pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _wait_group(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._process.poll()
            if not self._group_alive():
                return True
            time.sleep(0.05)
        self._process.poll()
        return not self._group_alive()

    def confirm_stopped(self, grace: float = 0.0) -> None:
        with self._stop_lock:
            process = self._process
            if process.poll() is None and grace > 0:
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
            if self._group_alive():
                try:
                    os.killpg(self._pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                self._wait_group(2.0)
            if self._group_alive():
                try:
                    os.killpg(self._pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._wait_group(5.0)
            if process.poll() is None:
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None or self._group_alive():
                raise RecognitionUnavailable(
                    "the gpu-stt worker process group could not be "
                    "confirmed stopped"
                )
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def shutdown(self) -> None:
        if self.alive:
            try:
                self._send({"op": "close"})
            except RecognitionUnavailable:
                pass
        self.confirm_stopped(grace=CANCEL_GRACE_SECONDS)

    def kill(self) -> None:
        self.confirm_stopped()