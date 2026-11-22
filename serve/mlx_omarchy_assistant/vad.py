"""Silero VAD: the speech decision behind the voice loop's endpointing.

The shipped endpointing gate (#625) decides voice activity from chunk RMS,
which cannot tell speech from loud noise (V25 receipt: F1 0.667,
specificity 0.00 on FluidAudio's mini50 — every noise clip contains a
chunk louder than any useful floor). This module adds the real VAD:
silero-vad v5 (MIT, https://github.com/snakers4/silero-vad), ONNX on CPU,
512-sample (32 ms) frames at 16 kHz with an internal state tensor.

The energy gate stays available as a pre-filter (Endpointer's
prefilter_floor): it only rejects early — when the chunk is silent the
model is never asked — and never overrides the model.

Model pin: silero_vad.onnx, sha-256
1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3
(captured 2026-11-22 from snakers4/silero-vad master,
src/silero_vad/data/silero_vad.onnx). The `--vad silero` flag is the
download approval, matching the wake-model pair flow.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 512  # samples per silero frame (32 ms)
CONTEXT = 64  # samples of overlap silero v5 carries between frames
FRAME_BYTES = FRAME * 2
STATE_SHAPE = (2, 1, 128)

MODEL_URL = ("https://raw.githubusercontent.com/snakers4/silero-vad/"
             "066409f3d9db45a30d21dfcbf9147826ffdd7052/"
             "src/silero_vad/data/silero_vad.onnx")
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"


class VadModelError(RuntimeError):
    pass


def prepare(home, fetch=None, digest: str = MODEL_SHA256) -> Path:
    """Verify (downloading if missing) the pinned silero model; return its path.

    `fetch(url) -> bytes` replaces the network for tests. The `--vad silero`
    flag on the launcher is the download approval, matching the wake-model
    flow.
    """
    directory = Path(home) / "vad"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "silero_vad.onnx"
    if path.exists() and _sha256(path) == digest:
        return path
    if fetch is not None:
        data = fetch(MODEL_URL)
    else:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
            if response.status != 200:
                raise VadModelError(f"{MODEL_URL} returned HTTP {response.status}")
            data = response.read()
    staging = path.with_name(path.name + ".part")
    staging.write_bytes(data)
    if _sha256(staging) != digest:
        staging.unlink()
        raise VadModelError(
            "silero_vad.onnx does not match the pinned SHA-256; refusing to use it")
    staging.replace(path)
    return path


def prepared(home) -> Path | None:
    """The staged model's path when present and sha-valid, else None."""
    path = Path(home) / "vad" / "silero_vad.onnx"
    return path if path.exists() and _sha256(path) == MODEL_SHA256 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SileroVad:
    """Feeds 512-sample frames to the silero ONNX and thresholds the output.

    `session` replaces the onnxruntime session for tests (anything with
    .run(out_names, feed) -> (probs, state)).
    """

    def __init__(self, model_path, threshold: float = 0.5, session=None):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(
                str(model_path), sess_options=options,
                providers=["CPUExecutionProvider"])
        self._session = session
        self._threshold = threshold
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._buf = b""

    def _run_frame(self, samples: np.ndarray) -> bool:
        # silero v5 expects float audio in [-1, 1] and a 64-sample context
        # prepended from the previous frame (its own ONNXWrapper does the
        # same concat; without it the mel window misaligns and scores
        # collapse to near zero on real speech - measured). The samples are
        # scaled once; the carried context is already scaled and must not
        # be divided again (that crushes it to ~1e-5).
        x = np.concatenate([self._context, samples.astype(np.float32) / 32768.0])
        probs, state = self._session.run(
            None, {"input": x[None], "state": self._state, "sr": self._sr})
        self._state = np.asarray(state, dtype=np.float32).reshape(STATE_SHAPE)
        self._context = x[-CONTEXT:].astype(np.float32)
        return bool(np.asarray(probs, dtype=np.float32).reshape(-1)[0] >= self._threshold)

    def process(self, pcm: bytes) -> bool:
        """Feed mic-sized PCM; True when any completed frame is speech."""
        self._buf += pcm
        voiced = False
        while len(self._buf) >= FRAME_BYTES:
            frame = np.frombuffer(self._buf[:FRAME_BYTES], dtype=np.int16)
            self._buf = self._buf[FRAME_BYTES:]
            if self._run_frame(frame.astype(np.float32)):
                voiced = True
        return voiced

    def voiced(self, pcm: bytes) -> bool:
        return self.process(pcm)
