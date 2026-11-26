# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Assistant speech recognition over the pinned Parakeet runtime.

Contract (offline assistant, voice adapter): ``Recognition(home)`` with
``status()``, ``transcribe(wav_bytes, cancel)``, ``close()``. Input is
mono PCM WAV (16-bit/32-bit integer or 32-bit IEEE float, at any
reasonable rate); samples are decoded to the Parakeet input contract —
16 kHz mono float32 — with resampling executed on the accelerator path
(``coreml.parakeet_dictation.resample_to_16k``). There is no CPU tensor
inference and no remote service: missing tools, assets, or accelerator
produce a named ``RecognitionUnavailable``, never a transcript.

Transcription runs in an owned subprocess driven by a framing
protocol. Every call is bounded by a deadline; a cancellation or
shutdown first asks the worker to stop and then escalates, and the
parent always confirms the child's exit before releasing its handle.
Silence decodes to an empty transcript — the caller renders it as
"No speech detected", it is never fabricated here.

Qualification is receipt-based: preflight probes can report a runtime
as usable, but only a recorded hardware acceptance run (see
``coreml.parakeet_dictation.write_acceptance_receipt``) moves the
status to ``ready``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

import mlx_omarchy_paths

TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 30.0
MAX_INPUT_RATE = 192_000
MAX_PAYLOAD_BYTES = 48 * 1024 * 1024
DEFAULT_DEADLINE_MS = 90_000
CANCEL_GRACE_SECONDS = 5.0

_DUR_EPSILON = 1e-6
_PCM_FORMAT = 0x0001
_FLOAT_FORMAT = 0x0003
_SUPPORTED_FORMATS = {_PCM_FORMAT: "PCM", _FLOAT_FORMAT: "IEEE float"}


class RecognitionUnavailable(RuntimeError):
    """Recognition cannot run on this host; the reason is named."""


class RecognitionInputError(ValueError):
    """The submitted audio violates the input contract; the rule is named."""


class RecognitionCancelled(RuntimeError):
    """The caller's cancellation event fired before or during work."""


class RecognitionTimedOut(RuntimeError):
    """The worker exceeded the call deadline and was confirmed stopped."""


def _installed_roots() -> list[Path]:
    """The mlx-omarchy install venv package root, independent of PATH."""
    roots: list[Path] = []
    for venv in mlx_omarchy_paths.venv_roots():
        roots.extend(venv.glob("lib/python3.*/site-packages/mlx"))
    return roots


def _locate_tools_root() -> Path | None:
    """The directory containing the coreml/ package tree, or None."""
    candidates: list[Path] = []
    env = os.environ.get("MLX_OMARCHY_TOOLS", "")
    if env:
        candidates.append(Path(env))
    candidates.extend(_installed_roots())
    candidates.append(Path(__file__).resolve().parent.parent)
    for base in Path(__file__).resolve().parents:
        candidates.append(base / "overlay" / "tools")
    found = shutil.which("mlx-omarchy-parakeet")
    if found:
        candidates.append(Path(found).resolve().parents[1])
    for candidate in candidates:
        if (candidate / "coreml" / "parakeet_dictation.py").is_file():
            return candidate
    return None


def _backend_kind(module) -> str:
    """``gpu`` for gpu_stt; the ANE dictation module carries no tag."""
    return getattr(module, "BACKEND_KIND", "ane")


def _load_dictation_module():
    """Return the active STT backend module or raise ``RecognitionUnavailable``.

    Runtime discovery picks the backend: ANE Parakeet when its tools
    tree is present AND its probe succeeds, otherwise the GPU STT
    path that ships with this package. ``MLX_OMARCHY_RECOGNITION_BACKEND``
    can pin ``ane`` or ``gpu`` for tests; default ``auto`` probes both.
    Both expose ``probe_runtime`` and ``read_acceptance_receipt``; each
    backend's worker receives the original samples and rate and resamples
    on its own accelerator. The ANE module has no ``BACKEND_KIND`` tag.
    """
    pin = os.environ.get("MLX_OMARCHY_RECOGNITION_BACKEND", "auto").strip().lower()
    if pin not in ("auto", "ane", "gpu"):
        pin = "auto"

    last_error: Exception | None = None

    def _try_ane():
        root = _locate_tools_root()
        if root is None:
            return None
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            from coreml import parakeet_dictation  # type: ignore
        except ImportError as exc:
            raise RecognitionUnavailable(
                "Parakeet dictation is not importable from "
                f"{root}: {exc}"
            ) from exc
        return parakeet_dictation

    def _try_gpu():
        try:
            from . import gpu_stt  # type: ignore
        except ImportError as exc:
            raise RecognitionUnavailable(
                f"the GPU STT backend is not importable: {exc}"
            ) from exc
        return gpu_stt

    if pin in ("auto", "ane"):
        try:
            module = _try_ane()
            if module is not None:
                probe = module.probe_runtime()
                if probe["ok"]:
                    return module
                last_error = RecognitionUnavailable(
                    "ANE Parakeet probe failed: " + "; ".join(probe["reasons"])
                )
            else:
                last_error = RecognitionUnavailable(
                    "the mlx-omarchy Parakeet tools tree (coreml package) "
                    "was not found; set MLX_OMARCHY_TOOLS or install the "
                    "mlx-omarchy-parakeet product"
                )
        except RecognitionUnavailable as exc:
            last_error = exc

    if pin in ("auto", "gpu"):
        try:
            return _try_gpu()
        except RecognitionUnavailable as exc:
            if pin == "gpu":
                raise
            last_error = exc

    assert last_error is not None
    raise last_error


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Parse mono PCM WAV into (float32 samples in [-1, 1), sample_rate).

    Header facts are checked strictly — RIFF size, duplicate chunks,
    block align, whole samples, odd chunk padding — before sample
    conversion allocates anything.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise RecognitionInputError("audio payload must be raw WAV bytes")
    data = bytes(data)
    if len(data) > MAX_PAYLOAD_BYTES:
        raise RecognitionInputError(
            f"audio payload exceeds {MAX_PAYLOAD_BYTES} bytes "
            f"({len(data)} received)"
        )
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RecognitionInputError("audio payload is not a RIFF/WAVE file")
    (riff_size,) = struct.unpack("<I", data[4:8])
    if riff_size != len(data) - 8:
        raise RecognitionInputError(
            f"RIFF header declares {riff_size} payload bytes but the "
            f"payload has {len(data) - 8}"
        )

    fmt = None
    data_chunk = None
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position:position + 4]
        chunk_size = struct.unpack("<I", data[position + 4:position + 8])[0]
        body_start = position + 8
        body_end = body_start + chunk_size
        padded_end = body_end + (chunk_size & 1)
        if body_end > len(data):
            raise RecognitionInputError(
                f"WAV chunk {chunk_id!r} declares {chunk_size} bytes but the "
                f"payload ends after {len(data) - body_start}; truncated audio"
            )
        if chunk_id == b"fmt ":
            if fmt is not None:
                raise RecognitionInputError("WAV payload has a duplicate fmt chunk")
            if chunk_size < 16:
                raise RecognitionInputError(
                    f"fmt chunk is {chunk_size} bytes; at least 16 required"
                )
            fmt = struct.unpack("<HHIIHH", data[body_start:body_start + 16])
        elif chunk_id == b"data":
            if data_chunk is not None:
                raise RecognitionInputError(
                    "WAV payload has a duplicate data chunk"
                )
            data_chunk = data[body_start:body_end]
        position = padded_end

    if fmt is None:
        raise RecognitionInputError("WAV payload has no fmt chunk")
    if data_chunk is None:
        raise RecognitionInputError("WAV payload has no data chunk")

    tag, channels, rate, _byte_rate, block_align, bits = fmt
    name = _SUPPORTED_FORMATS.get(tag)
    if name is None:
        raise RecognitionInputError(
            f"unsupported WAV format tag {tag}; supported: mono PCM (s16/s32) "
            f"or IEEE float (f32)"
        )
    if channels != 1:
        raise RecognitionInputError(
            f"WAV payload has {channels} channels; the recognizer accepts "
            f"mono capture only"
        )
    width = bits // 8
    if block_align != channels * width:
        raise RecognitionInputError(
            f"fmt block align is {block_align}; a mono {bits}-bit stream "
            f"requires {channels * width}"
        )
    if not 1 <= rate <= MAX_INPUT_RATE:
        raise RecognitionInputError(
            f"invalid sample rate {rate}; supported range is 1-{MAX_INPUT_RATE} Hz"
        )
    if tag == _PCM_FORMAT and bits not in (16, 32):
        raise RecognitionInputError(
            f"unsupported PCM sample width {bits} bits; supported: 16 or 32"
        )
    if tag == _FLOAT_FORMAT and bits != 32:
        raise RecognitionInputError(
            f"unsupported float sample width {bits} bits; supported: 32"
        )
    if len(data_chunk) % width:
        raise RecognitionInputError(
            f"data chunk holds {len(data_chunk)} bytes, not a whole number "
            f"of {width}-byte samples"
        )

    sample_count = len(data_chunk) // width
    duration = sample_count / rate
    if duration > MAX_DURATION_SECONDS + _DUR_EPSILON:
        raise RecognitionInputError(
            f"audio is {duration:.2f} s; the recognizer accepts at most "
            f"{MAX_DURATION_SECONDS:.0f} s per submission"
        )
    if sample_count == 0:
        raise RecognitionInputError("WAV payload contains no audio samples")

    if tag == _FLOAT_FORMAT:
        samples = np.frombuffer(data_chunk, dtype="<f4")
        if not np.isfinite(samples).all():
            raise RecognitionInputError(
                "audio contains non-finite samples (NaN or infinity)"
            )
        return np.clip(samples, -1.0, 1.0).astype(np.float32), rate
    if bits == 16:
        scaled = np.frombuffer(data_chunk, dtype="<i2").astype(np.float32)
        return scaled / 32768.0, rate
    scaled = np.frombuffer(data_chunk, dtype="<i4").astype(np.float32)
    return scaled / 2147483648.0, rate


class _WorkerHandle:
    """Owned dictation subprocess with bounded, confirmable requests."""

    def __init__(self, tools_root: Path):
        self._tools_root = tools_root
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{tools_root}{os.pathsep}{existing}" if existing else str(tools_root)
        )
        self._process = subprocess.Popen(
            [sys.executable, "-m", "coreml.parakeet_dictation", "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            cwd=str(tools_root),
            start_new_session=True,
        )
        self._pgid = os.getpgid(self._process.pid)
        self._send_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._request_id = 0
        self._active_id: int | None = None

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
            except (BrokenPipeError, OSError, ValueError):
                raise RecognitionUnavailable(
                    "the recognition worker exited unexpectedly; "
                    f"exit code {self._process.poll()}"
                )

    def request(self, header: dict, payload: bytes = b"", *,
                timeout: float, cancel=None) -> dict:
        """One bounded request; a caller cancel interrupts and escalates."""
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
            except RecognitionUnavailable:
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
                _worker_error(response), RecognitionCancelled
            ):
                raise RecognitionCancelled(
                    "cancelled during transcription"
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
                raise RecognitionUnavailable(
                    "the recognition worker exited while a request was in "
                    f"flight (exit code {process.poll()})"
                )
            raise RecognitionTimedOut(
                f"the recognition worker did not answer within {timeout:.0f} s "
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
            except RecognitionUnavailable:
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
            # Reap the direct child first: a zombie still answers
            # killpg(0) and would mask an otherwise empty group.
            self._process.poll()
            if not self._group_alive():
                return True
            time.sleep(0.05)
        self._process.poll()
        return not self._group_alive()

    def confirm_stopped(self, grace: float = 0.0) -> None:
        """Stop the whole worker session and confirm nothing survives.

        Escalation: graceful grace window, group SIGTERM, group SIGKILL.
        Confirmation requires the direct child reaped AND the process
        group empty — the ANE worker is a group member, so a parent-only
        poll would not prove it stopped.
        """
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
                    "the recognition worker process group could not be "
                    "confirmed stopped"
                )
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def shutdown(self) -> None:
        """Ask the worker to finish, then confirm exit before returning."""
        if self.alive:
            try:
                self._send({"op": "close"})
            except RecognitionUnavailable:
                pass
        self.confirm_stopped(grace=CANCEL_GRACE_SECONDS)

    def kill(self) -> None:
        self.confirm_stopped()


def _worker_error(response: dict) -> Exception:
    kind = response.get("kind")
    message = response.get("error", "worker failed without a reason")
    if kind == "input":
        return RecognitionInputError(message)
    if kind == "cancelled":
        return RecognitionCancelled(message)
    if kind == "timeout":
        return RecognitionTimedOut(message)
    return RecognitionUnavailable(message)


class Recognition:
    """Voice adapter: assistant WAV bytes in, transcript text out."""

    def __init__(self, home: Path):
        self._home = Path(home)
        self._lock = threading.Lock()
        self._worker: _WorkerHandle | None = None
        self._probe = None
        self._usable = False

    def status(self) -> dict:
        """Truthful readiness. ``ready`` requires an acceptance receipt."""
        with self._lock:
            probe = self._probe_cached()
            module = self._module_cached()
            receipt = None
            if module is not None:
                receipt = module.read_acceptance_receipt(self._home)
            usable = bool(self._usable or (probe["ok"] and receipt))
            if usable and receipt:
                state = "ready"
                reasons: list[str] = []
            elif probe["ok"]:
                state = "usable"
                reasons = [
                    "runtime is loadable and its pipeline may run, but no "
                    "hardware acceptance receipt is recorded yet"
                ]
            elif self._classify(probe) == "unqualified":
                state = "unqualified"
                reasons = list(probe["reasons"])
            else:
                state = "missing"
                reasons = list(probe["reasons"])
            facts = probe["facts"]
            facts["weight_bytes"] = (
                self._cache_weight_bytes(module) if module is not None else 0
            )
            status = {
                "state": state,
                "ready": state == "ready",
                "usable": usable,
                "qualified": receipt is not None,
                "reasons": reasons,
                "qualification": facts,
                "model": facts.get("model"),
                "memory": self._memory_estimate(probe),
                "max_duration_seconds": MAX_DURATION_SECONDS,
                "sample_rate": TARGET_SAMPLE_RATE,
                "encoding": "mono PCM WAV (s16/s32/f32), float32 to 16 kHz",
            }
            if receipt is not None:
                status["acceptance"] = {
                    "recorded_at": receipt.get("recorded_at"),
                    "emissions": receipt.get("emissions"),
                    "latency_ms": receipt.get("latency_ms"),
                }
            return status

    @staticmethod
    def _classify(probe: dict) -> str:
        if any(reason.startswith("platform:") for reason in probe["reasons"]):
            return "unqualified"
        return "missing"

    def _probe_cached(self) -> dict:
        if self._probe is None:
            try:
                module = _load_dictation_module()
            except RecognitionUnavailable as error:
                self._probe = {"ok": False, "reasons": [str(error)], "facts": {}}
            else:
                if _backend_kind(module) == "gpu":
                    self._probe = module.probe_runtime(self._home)
                else:
                    self._probe = module.probe_runtime()
                    facts = self._probe["facts"]
                    facts["model"] = self._model_identity(module)
                    facts["asset_bytes"] = self._asset_footprint(module)
        return self._probe

    def _module_cached(self):
        try:
            return _load_dictation_module()
        except RecognitionUnavailable:
            return None

    @staticmethod
    def _model_identity(module) -> dict | None:
        try:
            reference = module.load_cli_module().ReferenceLock.load()
            return {
                "repo": reference.model_repo,
                "revision": reference.model_revision,
            }
        except Exception:
            return None

    @staticmethod
    def _asset_footprint(module) -> dict:
        footprint: dict[str, int] = {}
        try:
            cli = module.load_cli_module()
            share = cli._share_dir()
            libane = share / "libane" / "libane-strict.so"
            if libane.is_file():
                footprint["libane_bytes"] = libane.stat().st_size
            bundles = share / "bundles"
            if bundles.is_dir():
                footprint["bundle_bytes"] = sum(
                    path.stat().st_size
                    for path in bundles.rglob("*") if path.is_file()
                )
            footprint["worker_bytes"] = cli._worker_path().stat().st_size
        except Exception:
            pass
        return footprint

    _MARGIN_BYTES = 256 * 1024 * 1024

    @classmethod
    def _memory_estimate(cls, probe: dict) -> dict:
        """Conservative named estimate; always positive for admission."""
        facts = probe["facts"]
        basis: dict[str, int] = {}
        for key in ("bundle_bytes", "libane_bytes", "worker_bytes", "weights_bytes"):
            value = int(facts.get("asset_bytes", {}).get(key, 0))
            if value:
                basis[key] = value
        weights = int(facts.get("weight_bytes", 0))
        if weights:
            basis["decoder_joint_weights_bytes"] = weights
        total = sum(basis.values()) + cls._MARGIN_BYTES
        basis["conservative_margin_bytes"] = cls._MARGIN_BYTES
        return {
            "runtime_estimate_bytes": total,
            "estimate": True,
            "estimate_basis": basis,
        }

    @staticmethod
    def _cache_weight_bytes(module) -> int:
        """On-disk decoder/joint weight bytes from the pinned cache."""
        total = 0
        try:
            cli = module.load_cli_module()
            lock = cli.ReferenceLock.load()
            cache_dir = cli._cache_dir(lock)
            for package in ("decoder.mlpackage", "joint.mlpackage"):
                package_dir = cache_dir / package
                if package_dir.is_dir():
                    total += sum(
                        path.stat().st_size
                        for path in package_dir.rglob("*") if path.is_file()
                    )
        except Exception:
            return 0
        return total

    def _ensure_worker(self):
        """Return a live worker handle for the active backend.

        ``_WorkerHandle`` (ANE subprocess) and ``gpu_stt._WorkerHandle``
        share the same ``request(header, payload, timeout, cancel)``
        contract, so the rest of ``transcribe`` is backend-agnostic.
        """
        with self._lock:
            if self._worker is not None and self._worker.alive:
                return self._worker
            module = _load_dictation_module()
            if _backend_kind(module) == "gpu":
                if not module.model_verified(self._home):
                    raise RecognitionUnavailable(
                        "the speech recognition model is not downloaded and "
                        "verified; enable voice in model setup")
                self._worker = module._WorkerHandle(module.model_dir(self._home))
            else:
                root = _locate_tools_root()
                if root is None:
                    raise RecognitionUnavailable(
                        "the mlx-omarchy Parakeet tools tree was not found"
                    )
                self._worker = _WorkerHandle(root)
            return self._worker

    def prepare(self, approve_download: bool) -> dict:
        """Download and verify the GPU backend's pinned model (approve-first).

        The ANE backend's assets come from ``mlx-omarchy-parakeet download``,
        so it has nothing to fetch here.
        """
        module = _load_dictation_module()
        if _backend_kind(module) != "gpu":
            return {"verified": True, "backend": "ane"}
        result = module.prepare(self._home, approve_download)
        with self._lock:
            self._probe = None
        return result

    def transcribe(self, wav_bytes: bytes, cancel=None) -> str:
        """Transcribe mono PCM WAV bytes; empty string means silence."""
        if cancel is not None and cancel.is_set():
            raise RecognitionCancelled("cancelled before transcription started")
        samples, rate = decode_wav(wav_bytes)
        worker = self._ensure_worker()
        payload = np.ascontiguousarray(samples, dtype="<f4").tobytes()
        deadline = DEFAULT_DEADLINE_MS / 1000.0 + CANCEL_GRACE_SECONDS
        try:
            response = worker.request(
                {
                    "op": "transcribe",
                    "sample_rate": rate,
                    "deadline_ms": DEFAULT_DEADLINE_MS,
                },
                payload,
                timeout=deadline,
                cancel=cancel,
            )
        except (RecognitionTimedOut, RecognitionUnavailable):
            with self._lock:
                if self._worker is worker:
                    self._worker = None
            worker.kill()
            raise
        self._usable = True
        return str(response.get("transcript", ""))

    def close(self) -> None:
        with self._lock:
            worker, self._worker = self._worker, None
        if worker is not None:
            worker.shutdown()
