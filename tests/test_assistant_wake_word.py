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
import http.server
import importlib.util
import os
import shutil
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

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


class ThresholdDefaultTests(unittest.TestCase):
    """The tuned wake threshold lives in one place."""

    def test_cli_and_server_use_the_tuned_default(self):
        from mlx_omarchy_assistant import __main__ as cli
        from mlx_omarchy_assistant import server, wake_word

        self.assertEqual(wake_word.WAKE_THRESHOLD_DEFAULT, 0.5)
        parser = cli.build_parser()
        args = parser.parse_args(["--wake-word"])
        self.assertEqual(args.wake_threshold, wake_word.WAKE_THRESHOLD_DEFAULT)
        import inspect
        self.assertEqual(
            inspect.signature(server.AssistantServer.start_wake_word)
            .parameters["threshold"].default,
            wake_word.WAKE_THRESHOLD_DEFAULT)


class DetectorTests(unittest.TestCase):
    def test_fires_at_threshold(self):
        detector = make_detector([0.0, 0.6])
        self.assertFalse(detector.process(b"\x00" * (wake_word.CHUNK * 2)))
        self.assertTrue(detector.process(b"\x00" * (wake_word.CHUNK * 2)))

    def test_below_threshold_never_fires(self):
        detector = make_detector([wake_word.WAKE_THRESHOLD_DEFAULT - 0.01, 0.0])
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

    def test_vad_floor_vetoes_a_wake_on_a_quiet_frame(self):
        session = ScriptedSession([], default=0.9)
        detector = wake_word.WakeWordDetector(
            Path("/unused/hey_jarvis_v0.1.onnx"), session=session,
            vad_rms_floor=200.0)
        detector.process(b"\x00" * (wake_word.CHUNK * 2))
        self.assertEqual(len(session.calls), 1)      # the frame was scored
        self.assertFalse(detector.process(b"\x00" * wake_word.CHUNK))
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(detector._peak, 0.9)        # scored, but vetoed

    def test_vad_floor_passes_loud_chunks_through(self):
        detector = make_detector([0.9], vad_rms_floor=200.0)
        loud = (b"\x01\x10") * (wake_word.CHUNK * 2)  # int16 ~4097 per sample
        self.assertTrue(detector.process(loud))
        self.assertGreater(detector._peak, 0.0)

    def test_vad_floor_zero_keeps_legacy_behavior(self):
        session = ScriptedSession([], default=0.4)
        detector = wake_word.WakeWordDetector(
            Path("/unused/hey_jarvis_v0.1.onnx"), session=session,
            vad_rms_floor=0.0)
        self.assertFalse(detector.process(b"\x00" * (wake_word.CHUNK * 2)))
        self.assertEqual(len(session.calls), 1)  # still scored

    def test_vad_floor_rejects_negative(self):
        with self.assertRaises(ValueError):
            make_detector([0.0], vad_rms_floor=-1.0)


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


def bounded(call, seconds=5.0):
    """Run call() on a daemon thread; fail instead of hanging the suite."""
    box = {}

    def run():
        try:
            box["value"] = call()
        except BaseException as exc:  # re-raised in the test thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AssertionError(f"call still blocked after {seconds} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class FakeResponse:
    """A finite urlopen() stand-in that counts the bytes handed out."""

    def __init__(self, data, status=200, fail_after=None):
        self.data = data
        self.status = status
        self.fail_after = fail_after  # exception raised once this many bytes went out
        self.served = 0
        self.unbounded_reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        if n is None or n < 0:
            self.unbounded_reads += 1
            n = len(self.data)
        if self.fail_after is not None and self.served >= self.fail_after[0]:
            raise self.fail_after[1]
        block = self.data[self.served:self.served + n]
        self.served += len(block)
        return block


class PinnedFileTests(unittest.TestCase):
    """fetch_pinned_file / pinned_file_ok: the model trust boundary the wake
    word and the VAD share. Every source here is finite."""

    DATA = b"pinned-model-bytes" * 8

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="wake-pin-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.path = self.dir / "model.onnx"
        self.digest = hashlib.sha256(self.DATA).hexdigest()
        self.fetched = []

    def fetch(self, data=None):
        def fetch(url):
            self.fetched.append(url)
            return self.DATA if data is None else data
        return fetch

    def stage(self, **kwargs):
        return bounded(lambda: wake_word.fetch_pinned_file(
            self.path, "https://example.invalid/model.onnx", self.digest,
            wake_word.WakeModelError, **kwargs))

    def names(self):
        return sorted(p.name for p in self.dir.iterdir())

    def cap(self, limit):
        return mock.patch.object(wake_word, "MAX_MODEL_BYTES", limit)

    def urlopen(self, response):
        return mock.patch.object(wake_word, "_open", lambda url: response)

    # staging

    def test_download_lands_readable_with_no_temp_left(self):
        for mask, expected in ((0o077, 0o600), (0o022, 0o644)):
            with self.subTest(umask=oct(mask)):
                self.clear()
                old = os.umask(mask)
                try:
                    self.stage(fetch=self.fetch())
                finally:
                    os.umask(old)
                self.assertEqual(self.path.read_bytes(), self.DATA)
                self.assertEqual(self.names(), ["model.onnx"])
                self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), expected)

    def test_old_part_symlink_is_never_followed(self):
        victim = Path(tempfile.mkdtemp(prefix="wake-victim-")) / "victim"
        self.addCleanup(shutil.rmtree, victim.parent, ignore_errors=True)
        victim.write_bytes(b"keep me")
        os.symlink(victim, self.dir / "model.onnx.part")
        self.stage(fetch=self.fetch())
        self.assertEqual(victim.read_bytes(), b"keep me")
        self.assertTrue((self.dir / "model.onnx.part").is_symlink())
        self.assertEqual(self.path.read_bytes(), self.DATA)
        self.assertEqual(self.names(), ["model.onnx", "model.onnx.part"])

    def test_hash_mismatch_removes_the_temp_file(self):
        with self.assertRaises(wake_word.WakeModelError):
            self.stage(fetch=self.fetch(b"not the model"))
        self.assertEqual(self.names(), [])

    def test_cleanup_unlink_error_does_not_hide_hash_mismatch(self):
        with mock.patch.object(
                wake_word.os, "unlink", side_effect=PermissionError("denied")) as unlink:
            with self.assertRaisesRegex(wake_word.WakeModelError, "SHA-256"):
                self.stage(fetch=self.fetch(b"not the model"))
        unlink.assert_called_once()

    def test_failed_or_interrupted_download_removes_the_temp_file(self):
        for exc in (ConnectionResetError("reset"), KeyboardInterrupt()):
            response = FakeResponse(self.DATA, fail_after=(16, exc))
            with self.subTest(exc=type(exc).__name__), self.urlopen(response):
                with self.assertRaises(type(exc)):
                    self.stage()
                self.assertGreater(response.served, 0)
                self.assertEqual(self.names(), [])

    def test_a_non_200_answer_is_refused_before_any_read(self):
        response = FakeResponse(self.DATA, status=203)
        with self.urlopen(response):
            with self.assertRaises(wake_word.WakeModelError):
                self.stage()
        self.assertEqual(response.served, 0)
        self.assertEqual(self.names(), [])

    def test_concurrent_downloads_both_finish_cleanly(self):
        gate = threading.Barrier(2, timeout=5)
        data = self.DATA

        class OverlappingResponse(FakeResponse):
            def read(self, n=-1):
                if not self.served:
                    gate.wait()  # both calls hold open temp files together
                return super().read(n)

        errors = []

        def run():
            try:
                wake_word.fetch_pinned_file(
                    self.path, "u", self.digest, wake_word.WakeModelError)
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(wake_word, "_open",
                               lambda url: OverlappingResponse(data)):
            workers = [threading.Thread(target=run, daemon=True) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(10)
        for worker in workers:
            self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.path.read_bytes(), self.DATA)
        self.assertEqual(self.names(), ["model.onnx"])

    # size ceiling

    def test_download_reads_one_byte_past_the_cap_then_refuses(self):
        response = FakeResponse(b"z" * 10_000)
        with self.cap(1000), self.urlopen(response):
            with self.assertRaises(wake_word.WakeModelError):
                self.stage()
        self.assertEqual(response.served, 1001)
        self.assertEqual(response.unbounded_reads, 0)
        self.assertEqual(self.names(), [])

    def test_fetched_bytes_past_the_cap_are_refused(self):
        with self.cap(len(self.DATA) - 1):
            with self.assertRaises(wake_word.WakeModelError):
                self.stage(fetch=self.fetch())
        self.assertEqual(self.names(), [])

    def test_a_file_exactly_at_the_cap_is_accepted_and_reused(self):
        response = FakeResponse(self.DATA)
        with self.cap(len(self.DATA)), self.urlopen(response):
            self.stage()
            self.assertEqual(self.path.read_bytes(), self.DATA)
            self.stage(fetch=lambda url: self.fail("verified file refetched"))

    def test_an_oversized_cached_file_is_refetched(self):
        with open(self.path, "wb") as handle:  # sparse: no bytes on disk
            handle.truncate(wake_word.MAX_MODEL_BYTES + 1)
        self.assertFalse(wake_word.pinned_file_ok(
            self.path, self.digest, wake_word.WakeModelError))
        self.stage(fetch=self.fetch())
        self.assertEqual(len(self.fetched), 1)
        self.assertEqual(self.path.read_bytes(), self.DATA)

    def test_a_corrupt_cached_file_is_refetched(self):
        self.path.write_bytes(b"half a model")
        self.stage(fetch=self.fetch())
        self.assertEqual(len(self.fetched), 1)
        self.assertEqual(self.path.read_bytes(), self.DATA)

    def test_a_verified_cached_file_is_not_fetched(self):
        self.path.write_bytes(self.DATA)
        self.stage(fetch=self.fetch())
        self.assertEqual(self.fetched, [])

    # cached path that is not a regular file

    def special_paths(self):
        """(label, make) pairs; make() turns self.path into that kind."""
        good_dir = Path(tempfile.mkdtemp(prefix="wake-good-"))
        self.addCleanup(shutil.rmtree, good_dir, ignore_errors=True)
        good = good_dir / "model.onnx"
        good.write_bytes(self.DATA)

        def fifo_with_writer():
            os.mkfifo(self.path)
            fd = os.open(self.path, os.O_RDWR | os.O_NONBLOCK)
            os.write(fd, b"x" * 64)
            self.addCleanup(os.close, fd)

        def unix_socket():
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.addCleanup(sock.close)
            try:
                sock.bind(str(self.path))
            except OSError as exc:
                self.skipTest(f"cannot bind a unix socket here: {exc}")

        return [
            ("symlink to the right bytes", lambda: os.symlink(good, self.path)),
            ("symlink to /dev/null", lambda: os.symlink("/dev/null", self.path)),
            ("dangling symlink", lambda: os.symlink(self.dir / "gone", self.path)),
            ("fifo without a writer", lambda: os.mkfifo(self.path)),
            ("fifo with a writer", fifo_with_writer),
            ("directory", lambda: self.path.mkdir()),
            ("unix socket", unix_socket),
        ]

    def clear(self):
        if self.path.is_dir() and not self.path.is_symlink():
            self.path.rmdir()
        elif os.path.lexists(self.path):
            self.path.unlink()

    def test_non_regular_cached_paths_are_refused_without_a_fetch(self):
        for label, make in self.special_paths():
            with self.subTest(label):
                self.clear()
                make()
                before = os.lstat(self.path)
                with self.assertRaises(wake_word.WakeModelError):
                    self.stage(fetch=self.fetch())
                self.assertEqual(self.fetched, [])
                after = os.lstat(self.path)
                self.assertEqual((before.st_ino, stat.S_IFMT(before.st_mode)),
                                 (after.st_ino, stat.S_IFMT(after.st_mode)))
                self.assertEqual(sorted(p.name for p in self.dir.iterdir()
                                        if p.name != "model.onnx"), [])

    def test_an_unreadable_cached_file_is_refused(self):
        if os.geteuid() == 0:
            self.skipTest("root reads mode 000 files")
        self.path.write_bytes(self.DATA)
        self.path.chmod(0)
        self.addCleanup(self.path.chmod, 0o600)
        with self.assertRaises(wake_word.WakeModelError):
            self.stage(fetch=self.fetch())
        self.assertEqual(self.fetched, [])

    def test_prepare_refuses_a_symlinked_model_file(self):
        directory = wake_word.model_dir(self.dir)
        directory.mkdir(parents=True)
        os.symlink("/dev/null", directory / "melspectrogram.onnx")
        with self.assertRaises(wake_word.WakeModelError):
            bounded(lambda: wake_word.prepare(self.dir, fetch=self.fetch()))
        self.assertEqual(self.fetched, [])


class PinnedDownloadHttpTests(unittest.TestCase):
    """The real urlopen path against a loopback server: GitHub release
    assets answer 302 to a CDN, so the bounded reader must follow it."""

    BODY = b"onnx" * 500

    def setUp(self):
        body = self.BODY
        redirect_state = {"sent": 0}
        redirect_done = threading.Event()
        self.redirect_state = redirect_state
        self.redirect_done = redirect_done

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/release/model.onnx":
                    self.send_response(302)
                    self.send_header("Location", "/cdn/model.onnx")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path == "/huge-redirect/model.onnx":
                    total = 64 << 20
                    chunk = b"\x00" * (64 << 10)
                    self.send_response(302)
                    self.send_header("Location", "/cdn/model.onnx")
                    self.send_header("Content-Length", str(total))
                    self.end_headers()
                    sent = 0
                    try:
                        while sent < total:
                            self.wfile.write(chunk)
                            sent += len(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        redirect_state["sent"] = sent
                        redirect_done.set()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        proxy = mock.patch.dict(os.environ, {"no_proxy": "*", "NO_PROXY": "*"})
        proxy.start()
        self.addCleanup(proxy.stop)
        self.dir = Path(tempfile.mkdtemp(prefix="wake-http-"))
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/release/model.onnx"

    def fetch(self, url=None):
        return bounded(lambda: wake_word.fetch_pinned_file(
            self.dir / "model.onnx", url or self.url,
            hashlib.sha256(self.BODY).hexdigest(), wake_word.WakeModelError),
            seconds=30)

    def test_redirected_download_is_followed_and_verified(self):
        self.fetch()
        self.assertEqual((self.dir / "model.onnx").read_bytes(), self.BODY)

    def test_redirected_download_past_the_cap_is_refused(self):
        with mock.patch.object(wake_word, "MAX_MODEL_BYTES", len(self.BODY) - 1):
            with self.assertRaises(wake_word.WakeModelError):
                self.fetch()
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_redirect_body_is_not_drained(self):
        url = f"http://127.0.0.1:{self.server.server_port}/huge-redirect/model.onnx"
        self.fetch(url)
        self.assertEqual((self.dir / "model.onnx").read_bytes(), self.BODY)
        self.assertTrue(self.redirect_done.wait(30), "redirect handler did not finish")
        self.assertLess(self.redirect_state["sent"], 32 << 20)


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


class DeviceSelectionTests(unittest.TestCase):
    def test_pulse_preferred_when_queryable_succeeds(self):
        class FakeSD:
            default = type("D", (), {"device": (-1, -1)})()
            def query_devices(self, name, kind):
                return {"name": name}
        device, source = wake_word.WakeListener._select_input_device(FakeSD())
        self.assertEqual(device, "pulse")
        self.assertIsNone(source)

    def test_falls_back_to_default_when_pulse_absent(self):
        class FakeSD:
            default = type("D", (), {"device": (3, 4)})()
            def query_devices(self, name, kind):
                raise ValueError("No input device matching 'pulse'")
        device, source = wake_word.WakeListener._select_input_device(FakeSD())
        self.assertEqual(device, 3)
        self.assertIsNone(source)

    def test_pactl_set_default_source_invoked_on_fallback(self):
        os.environ["PULSE_SOURCE"] = "effect_output.j293-mic"
        class FakeSD:
            default = type("D", (), {"device": (3, 4)})()
            def query_devices(self, name, kind):
                raise ValueError("No input device matching 'pulse'")
        import subprocess as sp
        original = sp.run
        calls = []
        sp.run = lambda *a, **k: calls.append((a, k)) or None
        try:
            device, source = wake_word.WakeListener._select_input_device(FakeSD())
        finally:
            sp.run = original
            del os.environ["PULSE_SOURCE"]
        self.assertEqual(device, 3)
        self.assertEqual(source, "effect_output.j293-mic")
        self.assertTrue(any("set-default-source" in str(args[0]) for args, _ in calls))


def _staged_wake_dir():
    import mlx_omarchy_paths
    staged = wake_word.model_dir(mlx_omarchy_paths.default_data_home())
    alt = Path(os.environ.get("MLX_OMARCHY_WAKE_MODELS_DIR", ""))
    if not all((staged / name).exists() for name in wake_word.MODEL_FILES) \
            and alt.exists():
        staged = alt
    return staged


def real_model_skip_reason(models_dir=None):
    """Why the real openWakeWord run cannot execute here, or None."""
    staged = models_dir if models_dir is not None else _staged_wake_dir()
    if not all((staged / name).exists() for name in wake_word.MODEL_FILES):
        return "pinned wake models not staged on this machine"
    for package in ("openwakeword", "onnxruntime", "numpy"):
        if importlib.util.find_spec(package) is None:
            return f"{package} is not installed in this interpreter"
    return None


class RealModelTests(unittest.TestCase):
    """Runs only where the pinned models are already staged (jwm1 receipt runs)."""

    def setUp(self):
        self.dir = _staged_wake_dir()
        self.reason = real_model_skip_reason(self.dir)

    def test_silence_stays_below_threshold(self):
        if self.reason:
            self.skipTest(self.reason)
        detector = wake_word.WakeWordDetector(self.dir / "hey_jarvis_v0.1.onnx")
        import array
        chunk = array.array("h", [0] * wake_word.CHUNK)
        fired = False
        for _ in range(50):  # 4 s of digital silence
            fired = detector.process(chunk.tobytes()) or fired
        self.assertFalse(fired)


class RealModelGateTests(unittest.TestCase):
    """A machine that cannot run the detector gets a stated skip, not an error."""

    def test_missing_models_reports_models(self):
        empty = Path(tempfile.mkdtemp())
        self.assertEqual(real_model_skip_reason(empty),
                         "pinned wake models not staged on this machine")

    def test_missing_openwakeword_names_the_package(self):
        staged = _staged_wake_dir()
        if not all((staged / name).exists() for name in wake_word.MODEL_FILES):
            self.skipTest("pinned wake models not staged on this machine")
        real_find = importlib.util.find_spec
        try:
            importlib.util.find_spec = (
                lambda name, *a, **k: None if name == "openwakeword"
                else real_find(name, *a, **k))
            reason = real_model_skip_reason()
        finally:
            importlib.util.find_spec = real_find
        self.assertIsNotNone(reason)
        self.assertIn("openwakeword", reason)

    def test_missing_onnxruntime_names_the_package(self):
        staged = _staged_wake_dir()
        if not all((staged / name).exists() for name in wake_word.MODEL_FILES):
            self.skipTest("pinned wake models not staged on this machine")
        if importlib.util.find_spec("openwakeword") is None:
            self.skipTest("openwakeword is absent here, so its gate fires first")
        real_find = importlib.util.find_spec
        try:
            importlib.util.find_spec = (
                lambda name, *a, **k: None if name == "onnxruntime"
                else real_find(name, *a, **k))
            reason = real_model_skip_reason()
        finally:
            importlib.util.find_spec = real_find
        self.assertIsNotNone(reason)
        self.assertIn("onnxruntime", reason)

    def test_real_model_test_skips_instead_of_erroring(self):
        staged = _staged_wake_dir()
        if not all((staged / name).exists() for name in wake_word.MODEL_FILES):
            self.skipTest("pinned wake models not staged on this machine")
        real_find = importlib.util.find_spec
        caught = None
        try:
            importlib.util.find_spec = (
                lambda name, *a, **k: None if name == "openwakeword"
                else real_find(name, *a, **k))
            case = RealModelTests("test_silence_stays_below_threshold")
            try:
                case.setUp()
                case.test_silence_stays_below_threshold()
            except unittest.SkipTest as skip:
                caught = str(skip)
        finally:
            importlib.util.find_spec = real_find
        self.assertIsNotNone(caught)
        self.assertIn("openwakeword", caught)


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
