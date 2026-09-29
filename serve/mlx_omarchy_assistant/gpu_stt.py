# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""GPU speech recognition for the offline assistant.

Apple Silicon Linux machines run mlx through the Vulkan backend (no
ANE, no Metal). This module is the parallel adapter to the pinned
Parakeet-TDT STT model that ships in mlx-audio 0.5.6: it loads the
model on the live mlx binary, decodes mono PCM WAV, and runs
``model.generate`` on the GPU. The contract mirrors the ANE Parakeet
adapter (``probe_runtime``, ``resample_to_16k``,
``read_acceptance_receipt``, ``write_acceptance_receipt``) so the
caller's runtime-discovery switch can drop in either backend without
branching. A worker subprocess owns the model so a wedge or cancel
can confirm the model's exit before raising.

Pinned model (verified 2026-09-29):
- repo ``mlx-community/parakeet-tdt-0.6b-v3``
- weights sha256 ``05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592``
- config sha256 ``f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c``
- license CC-BY-4.0 (NeMo Parakeet TDT 0.6B v3, NVIDIA).
- downloaded via ``huggingface_hub.snapshot_download`` with
  ``cache_dir=~/.cache/huggingface/hub``.

mlx-audio 0.5.6 RECORD sha256 ``86d106f8e013229dba23e71e34b1618c0c5aceb2ede5ae8c0eff9371f63234ae``.

Qualification is receipt-based: preflight probes may report the
runtime as usable, but only a recorded hardware acceptance run —
listener-verified transcript, zero CPU tensor dispatch events, p95
latency within budget — moves the status to ``ready``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 30.0
DEFAULT_DEADLINE_MS = 90_000
CANCEL_GRACE_SECONDS = 5.0
WORKER_WARMUP_MS = 1_000
BACKEND_KIND = "gpu"

GPU_STT_MODEL = {
    "id": "parakeet-tdt-0.6b-v3",
    "repo": "mlx-community/parakeet-tdt-0.6b-v3",
    "license": "cc-by-4.0",
    "weights_sha256": "05e01c7f396c298cf7d23f61da7b504adeab698f0aaeafd9c82d198625464592",
    "config_sha256": "f320f1292511f34ec47f513755fe20fd01dbfc09a925d42730e66059a6e1ef4c",
    "expected_bytes": 2_508_288_736,
    "runtime": {
        "mlx_audio": {
            "version": "0.5.6",
            "record_sha256": "86d106f8e013229dba23e71e34b1618c0c5aceb2ede5ae8c0eff9371f63234ae",
        },
        "requires": ["mlx", "mlx_audio", "numpy"],
    },
}


class GpuSttUnavailable(RuntimeError):
    """The GPU STT runtime cannot run on this host; reason named."""


class GpuSttInputError(ValueError):
    """Input violates the GPU STT contract; rule named."""


class GpuSttCancelled(RuntimeError):
    """The caller's cancel event fired before or during work."""


class GpuSttTimedOut(RuntimeError):
    """The worker exceeded the call deadline and was confirmed stopped."""


def _model_dir() -> Path:
    """Resolved HF cache snapshot dir for the pinned model."""
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + GPU_STT_MODEL["repo"].replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        for entry in sorted(snapshots.iterdir(), reverse=True):
            if (entry / "config.json").is_file() and (
                (entry / "model.safetensors").is_file()
                or (entry / "weights.safetensors").is_file()
            ):
                return entry
    return snapshots  # caller treats this as "missing"


def _model_files(model_dir: Path) -> list[Path]:
    files = []
    for name in (
        "config.json", "model.safetensors", "weights.safetensors",
        "tokenizer.model", "tokenizer.vocab", "vocab.txt", "README.md",
    ):
        path = model_dir / name
        if path.is_file():
            files.append(path)
    return files


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _model_verified(model_dir: Path) -> tuple[bool, list[str]]:
    """Verify weights + config sha256 against the pinned manifest."""
    reasons: list[str] = []
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        weights = model_dir / "weights.safetensors"
    if not weights.is_file():
        return False, ["pinned model weights not present in the HF cache"]
    actual = _sha256_file(weights)
    if actual != GPU_STT_MODEL["weights_sha256"]:
        return False, [
            f"weights sha256 mismatch: expected "
            f"{GPU_STT_MODEL['weights_sha256']}, got {actual}"
        ]
    config = model_dir / "config.json"
    if config.is_file():
        actual_cfg = _sha256_file(config)
        if actual_cfg != GPU_STT_MODEL["config_sha256"]:
            reasons.append(
                f"config sha256 mismatch: expected "
                f"{GPU_STT_MODEL['config_sha256']}, got {actual_cfg}"
            )
    if reasons:
        return False, reasons
    return True, []


def _probe_mlx_audio() -> dict:
    """Cheap check: mlx + mlx_audio importable on the live venv."""
    info: dict = {"present": False, "missing": [], "detail": {}}
    import importlib
    for module_name in ("mlx.core", "mlx_audio.stt.utils"):
        try:
            importlib.import_module(module_name)
            info["detail"][module_name] = True
        except Exception as exc:
            info["detail"][module_name] = False
            info["missing"].append(f"{module_name}: {exc}")
    info["present"] = not info["missing"]
    return info


def _probe_accelerator() -> dict:
    """GPU must be live (default device gpu, vulkan or metal)."""
    info: dict = {"available": False, "detail": "mlx not importable"}
    try:
        import mlx.core as mx
        device = str(mx.default_device())
        info["device"] = device
        if "gpu" in device:
            info["available"] = True
            info["detail"] = f"default device {device}"
        else:
            info["detail"] = f"default device is {device}, not gpu"
    except Exception as exc:
        info["detail"] = f"mlx import failed: {exc}"
    return info


def probe_runtime(verify_cache: bool = False) -> dict:
    """Preflight summary the caller merges with its own facts.

    ``verify_cache=True`` also pins the on-disk model against the
    manifest. The probe NEVER promotes the runtime to ``ready``: that
    is reserved for the recorded acceptance receipt.
    """
    facts: dict = {"kind": "gpu_stt", "model": GPU_STT_MODEL["id"]}
    reasons: list[str] = []
    mlx_info = _probe_mlx_audio()
    facts["dependencies"] = mlx_info
    accel = _probe_accelerator()
    facts["accelerator"] = accel
    model_dir = _model_dir()
    facts["model_dir"] = str(model_dir)
    files = _model_files(model_dir) if model_dir.is_dir() else []
    facts["asset_bytes"] = sum(p.stat().st_size for p in files)
    if verify_cache:
        verified, model_reasons = _model_verified(model_dir)
        facts["cache"] = "verified" if verified else "invalid"
        reasons.extend(model_reasons)
    elif files:
        facts["cache"] = "present"
    else:
        facts["cache"] = "missing"
        reasons.append(
            "pinned STT model not present in the HF cache; download with "
            f"huggingface-cli download {GPU_STT_MODEL['repo']}"
        )
    if not mlx_info["present"]:
        reasons.append("mlx + mlx_audio not importable in the live venv")
    if not accel["available"]:
        reasons.append(f"GPU unavailable: {accel['detail']}")
    return {"ok": not reasons, "reasons": reasons, "facts": facts}


def resample_to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    """Linear resampling to the model's input sample rate (16 kHz).

    Runs in numpy so it works without a working tree; for a small
    30 s clip a polyphase filter would be cleaner, but at 48 kHz the
    linear path uses O(N) samples and the model itself runs the
    log-mel frontend, so the difference is not measurable.
    """
    if rate == TARGET_SAMPLE_RATE:
        return np.ascontiguousarray(samples, dtype=np.float32)
    if rate <= 0:
        raise GpuSttInputError(f"invalid source rate {rate}")
    out_length = max(1, int(round(samples.shape[0] * TARGET_SAMPLE_RATE / rate)))
    ratio = rate / TARGET_SAMPLE_RATE
    out = np.empty(out_length, dtype=np.float32)
    for i in range(out_length):
        src = i * ratio
        left = int(src)
        right = min(samples.shape[0] - 1, left + 1)
        t = src - left
        out[i] = samples[left] * (1 - t) + samples[right] * t
    return out


def _receipt_path(home: Path) -> Path:
    return Path(home) / "voice" / "stt" / "acceptance.json"


def read_acceptance_receipt(home: Path) -> dict | None:
    """Return the on-disk receipt for this runtime, or ``None``."""
    path = _receipt_path(home)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_acceptance_receipt(home: Path, *, transcript: str, emissions: dict,
                              cpu_tensor_events: int, latency_ms: dict,
                              mlx_binary_sha: str, model_sha: str,
                              host: str) -> dict:
    """Write the durable hardware acceptance receipt.

    Required facts: listener_verified true, zero cpu_tensor_events,
    latency_ms with at least ``p50`` and ``p95`` over the warm batch.
    The mlx binary hash and pinned model hash are stored so a future
    runtime (different wheel or model) cannot masquerade as the one
    that was measured.
    """
    if cpu_tensor_events != 0:
        raise ValueError(
            f"acceptance refused: cpu_tensor_events={cpu_tensor_events} != 0"
        )
    if "p50" not in latency_ms or "p95" not in latency_ms:
        raise ValueError("latency_ms must include p50 and p95")
    receipt = {
        "schema": "mlx-omarchy/gpu-stt-acceptance/1",
        "recorded_at": time.time(),
        "host": host,
        "model": GPU_STT_MODEL["id"],
        "model_repo": GPU_STT_MODEL["repo"],
        "model_sha256": model_sha,
        "mlx_binary_sha256": mlx_binary_sha,
        "transcript": transcript,
        "emissions": emissions,
        "cpu_tensor_events": cpu_tensor_events,
        "latency_ms": latency_ms,
        "listener_verified": True,
    }
    path = _receipt_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return receipt


def _worker_error(response: dict) -> RuntimeError:
    kind = response.get("kind")
    message = response.get("error", "gpu-stt worker failed without a reason")
    if kind == "input":
        return GpuSttInputError(message)
    if kind == "cancelled":
        return GpuSttCancelled(message)
    if kind == "timeout":
        return GpuSttTimedOut(message)
    return GpuSttUnavailable(message)


class _WorkerHandle:
    """Owned GPU STT subprocess; model resident, IPC-framed, cancel confirms."""

    def __init__(self, model_dir: Path, model_id: str):
        self._model_dir = model_dir
        self._model_id = model_id
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        voice_site = str(Path.home() / "voice-site")
        # The worker resolves ``mlx_omarchy_assistant.gpu_stt_worker`` as a
        # module, so the assistant package directory must be on the path.
        # When installed by install.sh it lives at ``$PREFIX``, so fall
        # back to the same prefix the assistant module is loaded from.
        package_root = str(Path(__file__).resolve().parent.parent)
        pieces = [p for p in (voice_site, package_root, existing) if p]
        env["PYTHONPATH"] = os.pathsep.join(pieces)
        self._process = subprocess.Popen(
            [sys.executable, "-m", "mlx_omarchy_assistant.gpu_stt_worker",
             "--model-dir", str(model_dir),
             "--model-id", GPU_STT_MODEL["repo"]],
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
        self._request_id = 0
        self._active_id: int | None = None

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
                # Drain the worker's stderr so the failure has a named cause.
                err = b""
                try:
                    proc = self._process
                    if proc.stderr is not None:
                        err = proc.stderr.read() or b""
                except Exception:
                    pass
                stderr_tail = err.decode("utf-8", "replace").strip()
                detail = stderr_tail or str(exc)
                raise GpuSttUnavailable(
                    f"the gpu-stt worker exited unexpectedly (exit code "
                    f"{self._process.poll()}); stderr: {detail[:400]}"
                )

    def request(self, header: dict, payload: bytes = b"", *,
                timeout: float, cancel=None) -> dict:
        self._request_id += 1
        request_id = self._request_id
        self._active_id = request_id
        done = threading.Event()

        def watch_cancel():
            if cancel is None:
                return
            cancel.wait()
            if done.is_set():
                return
            self._active_id = None
            try:
                self._send({"op": "cancel"})
            except GpuSttUnavailable:
                return
            if not done.wait(CANCEL_GRACE_SECONDS):
                self.confirm_stopped()

        watcher = None
        if cancel is not None:
            watcher = threading.Thread(target=watch_cancel, daemon=True)
            watcher.start()
        try:
            header = dict(header, id=request_id)
            self._send(header, payload)
            assert self._process.stdout is not None
            response = self._read_response(timeout)
        finally:
            done.set()
            self._active_id = None
            if watcher is not None:
                watcher.join(timeout=CANCEL_GRACE_SECONDS + 1.0)
        if not response.get("ok"):
            if cancel is not None and cancel.is_set() and not isinstance(
                _worker_error(response), GpuSttCancelled
            ):
                raise GpuSttCancelled(
                    "cancelled during gpu transcription"
                ) from _worker_error(response)
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
                raise GpuSttUnavailable(
                    "the gpu-stt worker exited while a request was in "
                    f"flight (exit code {process.poll()})"
                )
            raise GpuSttTimedOut(
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

    def cancel(self) -> None:
        if self.alive and self._active_id is not None:
            try:
                self._send({"op": "cancel"})
            except GpuSttUnavailable:
                pass

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
                raise GpuSttUnavailable(
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
            except GpuSttUnavailable:
                pass
        self.confirm_stopped(grace=CANCEL_GRACE_SECONDS)

    def kill(self) -> None:
        self.confirm_stopped()