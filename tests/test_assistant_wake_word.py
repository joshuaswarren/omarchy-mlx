"""Wake word listener: detector firing rules, pinned downloads, status wiring.

UNIT + protocol only, no audio hardware: the detector runs against a
scripted score session, prepare() runs against an injected fetch, and the
listener runs against a fake stream factory. What is proven: the
threshold and refractory rules, exact chunk accounting, hash-pinned
model staging (mismatch refuses, never silently uses), the listener's
detection-to-state path and its error containment, and the server status
surface. Real openWakeWord inference is hardware work; the models test
below runs only where the pinned files already exist.
"""

import hashlib
import os
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import wake_word  # noqa: E402


class ScriptedSession:
    """openWakeWord Model stand-in: pops one scripted score per process call."""

    def __init__(self, scores, default=0.0):
        self._scores = list(scores)
        self._default = default
        self.calls = []

    def predict(self, pcm):
        self.calls.append(len(pcm) // 2)
        return {"hey_jarvis_v0.1": self._scores.pop(0) if self._scores else self._default}

    def reset(self):
        pass


def make_detector(scores, default=0.0, **kwargs):
    path = Path("/unused/hey_jarvis_v0.1.onnx")
    return wake_word.WakeWordDetector(
        path, session=ScriptedSession(scores, default=default), **kwargs)


class DetectorTests(unittest.TestCase):
    def test_fires_at_threshold(self):
        detector = make_detector([0.0, 0.6])
        self.assertFalse(detector.process(b"\x00" * (wake_word.CHUNK * 2)))
        self.assertTrue(detector.process(b"\x00" * (wake_word.CHUNK * 2)))

    def test_below_threshold_never_fires(self):
        detector = make_detector([0.49, 0.0])
        self.assertFalse(detector.process(b"\x00" * (wake_word.CHUNK * 2)))

    def test_refractory_suppresses_immediate_refire(self):
        detector = make_detector([0.6], default=0.9)
        # 1.5 s refractory at 16 kHz; chunks are 80 ms, so 19 more chunks
        # must pass before the second crossing fires.
        self.assertTrue(detector.process(b"\x00" * (wake_word.CHUNK * 2)))
        for _ in range(18):
            self.assertFalse(detector.process(b"\x00" * (wake_word.CHUNK * 2)))
        self.assertTrue(detector.process(b"\x00" * (wake_word.CHUNK * 2)))

    def test_partial_chunks_are_accounted(self):
        detector = make_detector([0.0, 0.0, 0.7])
        detector.process(b"\x00" * 1000)          # 500 samples
        detector.process(b"\x00" * 1000)          # 500 samples
        # 80 ms boundary not yet crossed by sample count? The session is
        # called per process() regardless of size; what matters is the
        # consumed-seconds ledger the refractory uses.
        self.assertTrue(detector.process(b"\x00" * 1000))
        self.assertEqual(detector._consumed, 1500 / wake_word.SAMPLE_RATE)

    def test_empty_pcm_is_a_no_call(self):
        detector = make_detector([0.9])
        self.assertFalse(detector.process(b""))
        detector.process(b"\x00" * (wake_word.CHUNK * 2))
        self.assertEqual(len(detector._session.calls), 1)

    def test_threshold_bounds(self):
        with self.assertRaises(ValueError):
            make_detector([0.0], threshold=0.0)
        with self.assertRaises(ValueError):
            make_detector([0.0], threshold=1.0)

    def test_unknown_model_name_is_rejected_by_prepare(self):
        with self.assertRaises(wake_word.WakeModelError):
            wake_word.prepare(Path("/unused"), model="alexa")


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def _payload(self, name, mutate=False):
        data = (name + ":payload").encode()
        if mutate:
            data += b"corrupt"
        return data

    def _fetch(self, mutate=False):
        def fetch(url):
            name = url.rsplit("/", 1)[-1]
            if name not in wake_word.MODEL_FILES:
                raise urllib.error.HTTPError(url, 404, "nope", {}, None)
            return self._payload(name, mutate=mutate)
        return fetch

    def test_download_stages_hash_verified_files(self):
        real = {name: wake_word.MODEL_FILES[name][1]
                for name in wake_word.MODEL_FILES}
        # Recompute the digests for our synthetic payloads; prepare() must
        # accept exactly these and nothing else.
        import hashlib
        for name in real:
            real[name] = hashlib.sha256(self._payload(name)).hexdigest()
        saved = wake_word.MODEL_FILES
        wake_word.MODEL_FILES = {
            name: (saved[name][0], real[name]) for name in saved}
        try:
            fetches = []
            def fetch(url):
                fetches.append(url)
                return self._fetch()(url)
            path = wake_word.prepare(self.home, fetch=fetch)
            self.assertTrue(path.exists())
            self.assertEqual(len(fetches), len(wake_word.MODEL_FILES))
            # Second call: everything verified, no network.
            fetches.clear()
            wake_word.prepare(self.home, fetch=fetch)
            self.assertEqual(fetches, [])
        finally:
            wake_word.MODEL_FILES = saved

    def test_hash_mismatch_refuses_and_removes_staging(self):
        saved = wake_word.MODEL_FILES
        import hashlib
        wake_word.MODEL_FILES = {
            name: (saved[name][0], hashlib.sha256(b"never").hexdigest())
            for name in saved}
        try:
            with self.assertRaises(wake_word.WakeModelError):
                wake_word.prepare(self.home, fetch=self._fetch(mutate=True))
            leftovers = list(self.home.rglob("*.part"))
            self.assertEqual(leftovers, [])
        finally:
            wake_word.MODEL_FILES = saved


class WakeStateTests(unittest.TestCase):
    def test_snapshot_shape_and_counts(self):
        state = wake_word.WakeState("hey_jarvis", 0.5)
        snap = state.snapshot()
        self.assertEqual(snap, {"model": "hey_jarvis", "threshold": 0.5,
                                "detections": 0, "last_detection": None,
                                "listening": False, "error": None})
        state.record()
        state.set_running(True)
        snap = state.snapshot()
        self.assertEqual(snap["detections"], 1)
        self.assertTrue(snap["listening"])
        self.assertTrue(snap["last_detection"])

    def test_error_contains_and_stops(self):
        state = wake_word.WakeState("hey_jarvis", 0.5)
        state.set_running(True)
        state.set_error("RuntimeError: mic gone")
        snap = state.snapshot()
        self.assertIn("mic gone", snap["error"])
        self.assertFalse(snap["listening"])


class FakeStream:
    def __init__(self, callback):
        self.callback = callback
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


class ListenerTests(unittest.TestCase):
    def _run_listener(self, detector, frames):
        state = wake_word.WakeState("hey_jarvis", 0.5)
        holder = {}

        def factory(callback):
            holder["callback"] = callback
            return FakeStream(callback)

        listener = wake_word.WakeListener(detector, state, stream_factory=factory)
        listener.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if holder.get("callback") and state.snapshot()["listening"]:
                break
            time.sleep(0.02)
        for frame in frames:
            holder["callback"](frame, wake_word.CHUNK, None, None)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if state.snapshot()["detections"] or state.snapshot()["error"]:
                break
            time.sleep(0.02)
        listener.stop()
        listener.join(5)
        return state

    def test_frames_become_detections(self):
        detector = make_detector([0.0, 0.6])
        frame = b"\x00" * (wake_word.CHUNK * 2)
        state = self._run_listener(detector, [frame, frame])
        self.assertIsNone(state.snapshot()["error"])
        self.assertEqual(state.snapshot()["detections"], 1)
        self.assertFalse(state.snapshot()["listening"])

    def test_stream_error_lands_in_state(self):
        detector = make_detector([0.6])
        state = wake_word.WakeState("hey_jarvis", 0.5)

        def boom(callback):
            raise OSError("no audio device")

        listener = wake_word.WakeListener(detector, state, stream_factory=boom)
        listener.run()
        self.assertIn("no audio device", state.snapshot()["error"])


class RealModelTests(unittest.TestCase):
    """Runs only where the pinned models are already staged (jwm1 receipt runs)."""

    def setUp(self):
        import mlx_omarchy_paths
        self.dir = wake_word.model_dir(mlx_omarchy_paths.default_data_home())
        alt = Path(os.environ.get("MLX_OMARCHY_WAKE_MODELS_DIR", ""))
        if not all((self.dir / name).exists() for name in wake_word.MODEL_FILES) \
                and alt.exists():
            self.dir = alt
        self.have = all((self.dir / name).exists() for name in wake_word.MODEL_FILES)

    def test_silence_stays_below_threshold(self):
        if not self.have:
            self.skipTest("pinned wake models not staged on this machine")
        detector = wake_word.WakeWordDetector(self.dir / "hey_jarvis_v0.1.onnx")
        import array
        chunk = array.array("h", [0] * wake_word.CHUNK)
        fired = False
        for _ in range(50):  # 4 s of digital silence
            fired = detector.process(chunk.tobytes()) or fired
        self.assertFalse(fired)


class ServerStatusTests(unittest.TestCase):
    def test_status_includes_wake_snapshot_when_started(self):
        from mlx_omarchy_assistant.server import AssistantServer
        home = Path(tempfile.mkdtemp())
        server = AssistantServer(("127.0.0.1", 0), home)
        try:
            self.assertNotIn("wake", server.status())
            server.wake = wake_word.WakeState("hey_jarvis", 0.5)
            server.wake.record()
            self.assertEqual(server.status()["wake"]["detections"], 1)
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
