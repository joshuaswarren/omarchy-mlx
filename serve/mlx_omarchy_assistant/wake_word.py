"""Opt-in wake word listener: openWakeWord ONNX models on the CPU.

Off by default. `mlx-omarchy-assistant --wake-word hey_jarvis` starts a
thread that scores 16 kHz mono microphone frames through the pinned
openWakeWord release models and records detections for /api/status; the
browser UI polls status and starts a hands-free dictation on a fresh
detection. Model files download once into the assistant home and are
verified against the SHA-256 pins below; a mismatch is a hard error,
never a silently used file.
"""

from __future__ import annotations

import hashlib
import queue
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODEL_RELEASE = "v0.5.1"
MODEL_BASE = f"https://github.com/dscripka/openWakeWord/releases/download/{MODEL_RELEASE}"
# SHA-256 pins captured 2026-10-07 from the release assets (Apache-2.0).
MODEL_FILES = {
    "melspectrogram.onnx": (
        f"{MODEL_BASE}/melspectrogram.onnx",
        "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f"),
    "embedding_model.onnx": (
        f"{MODEL_BASE}/embedding_model.onnx",
        "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f"),
    "hey_jarvis_v0.1.onnx": (
        f"{MODEL_BASE}/hey_jarvis_v0.1.onnx",
        "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb"),
}
WAKE_MODELS = {"hey_jarvis": "hey_jarvis_v0.1.onnx"}
SAMPLE_RATE = 16000
CHUNK = 1280  # 80 ms at 16 kHz: the streaming step the listener asks for


class WakeModelError(RuntimeError):
    pass


def model_dir(home) -> Path:
    return Path(home) / "wake" / MODEL_RELEASE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(home, model: str = "hey_jarvis", fetch=None) -> Path:
    """Verify (downloading if missing) every pinned file; return the wake model path.

    `fetch(url) -> bytes` replaces the network for tests. The `--wake-word`
    flag on the launcher is the download approval, matching the pair and
    voice-asset approval flow.
    """
    if model not in WAKE_MODELS:
        raise WakeModelError(
            f"Unknown wake word {model!r}; choose one of {sorted(WAKE_MODELS)}")
    directory = model_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in MODEL_FILES.items():
        path = directory / name
        if path.exists() and _sha256(path) == digest:
            continue
        if fetch is not None:
            data = fetch(url)
        else:
            with urllib.request.urlopen(url, timeout=120) as response:
                if response.status != 200:
                    raise WakeModelError(f"{url} returned HTTP {response.status}")
                data = response.read()
        staging = path.with_name(path.name + ".part")
        staging.write_bytes(data)
        if _sha256(staging) != digest:
            staging.unlink()
            raise WakeModelError(
                f"{name} does not match the pinned SHA-256; refusing to use it")
        staging.replace(path)
    return directory / WAKE_MODELS[model]


class WakeWordDetector:
    """Scores 16 kHz mono little-endian int16 PCM and fires once per refractory window."""

    def __init__(self, model_path, threshold: float = 0.5,
                 refractory_seconds: float = 1.5, session=None):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if session is None:
            from openwakeword.model import Model
            directory = Path(model_path).parent
            session = Model(
                wakeword_models=[str(model_path)],
                melspec_model_path=str(directory / "melspectrogram.onnx"),
                embedding_model_path=str(directory / "embedding_model.onnx"),
                inference_framework="onnx")
        self._session = session
        self.threshold = threshold
        self._refractory = refractory_seconds
        self._consumed = 0.0
        self._last_fire = float("-inf")
        self._peak = 0.0

    def process(self, pcm: bytes | bytearray | memoryview) -> bool:
        samples = len(pcm) // 2
        if samples == 0:
            return False
        import numpy as np
        scores = self._session.predict(np.frombuffer(pcm, dtype=np.int16))
        self._consumed += samples / SAMPLE_RATE
        score = max(scores.values())
        if score > self._peak:
            self._peak = score
        if score < self.threshold:
            return False
        if self._consumed - self._last_fire < self._refractory:
            return False
        self._last_fire = self._consumed
        return True

    def reset(self) -> None:
        self._session.reset()
        self._consumed = 0.0
        self._last_fire = float("-inf")
        self._peak = 0.0


class WakeState:
    """Thread-safe detection record surfaced through /api/status."""

    def __init__(self, model: str, threshold: float):
        self._lock = threading.Lock()
        self._model = model
        self._threshold = threshold
        self._detections = 0
        self._last = None
        self._running = False
        self._error = None

    def record(self) -> None:
        with self._lock:
            self._detections += 1
            self._last = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def set_running(self, running: bool) -> None:
        with self._lock:
            self._running = running

    def set_error(self, error: str) -> None:
        with self._lock:
            self._error = error
            self._running = False

    def snapshot(self) -> dict:
        with self._lock:
            return {"model": self._model, "threshold": self._threshold,
                    "detections": self._detections, "last_detection": self._last,
                    "listening": self._running, "error": self._error}


class WakeListener(threading.Thread):
    """Scores microphone frames until stop(). Audio-device errors land in WakeState."""

    def __init__(self, detector: WakeWordDetector, state: WakeState,
                 stream_factory=None):
        super().__init__(daemon=True, name="wake-word")
        self._detector = detector
        self._state = state
        self._frames: queue.SimpleQueue = queue.SimpleQueue()
        self._stopping = threading.Event()
        self._factory = stream_factory or self._open_stream

    @staticmethod
    def _select_input_device(sounddevice):
        # Prefer the ALSA "pulse" device (libpulse-backed, honors PULSE_SOURCE
        # / pactl set-default-source) so the assistant hears the same
        # microphone the user's browser + parecord see. Fall back to the
        # system default if "pulse" is absent. On the PulseAudio fallback
        # path, sync the OS default source to PULSE_SOURCE so the listener
        # hears the named wireplumber filter (e.g. j293-mic) rather than
        # whichever hardware input is configured today.
        try:
            sounddevice.query_devices("pulse", kind="input")
            return "pulse", None
        except Exception:
            pass
        source = __import__("os").environ.get("PULSE_SOURCE")
        if source:
            try:
                subprocess = __import__("subprocess")
                subprocess.run(["pactl", "set-default-source", source],
                               check=False, capture_output=True, timeout=5)
            except Exception:
                pass
        return sounddevice.default.device[0], source

    @staticmethod
    def _open_stream(callback):
        import sounddevice
        device, _ = WakeListener._select_input_device(sounddevice)
        return sounddevice.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK,
            callback=callback, device=device)

    def _callback(self, indata, frames, time_info, status) -> None:
        self._frames.put(bytes(indata))

    def run(self) -> None:
        stream = None
        try:
            stream = self._factory(self._callback)
            stream.start()
            self._state.set_running(True)
            while not self._stopping.is_set():
                try:
                    pcm = self._frames.get(timeout=0.5)
                except queue.Empty:
                    continue
                if self._detector.process(pcm):
                    self._state.record()
        except Exception as exc:  # the assistant stays up; status explains
            self._state.set_error(f"{type(exc).__name__}: {exc}")
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            self._state.set_running(False)

    def stop(self) -> None:
        self._stopping.set()
