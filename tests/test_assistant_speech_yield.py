"""Speech-yield scheduling: bounded TTS between generation chunks.

UNIT + protocol only, no inference: the chat worker is a real loopback HTTP
server that speaks both the SSE wire and the /v1/internal/yield control
route; the decode side is a real thread stepping a scripted fake
BatchGenerator through the shim's wrapped class. What is proven: the
hold/park/release ordering, KV-state preservation across a park (the cache
object identity survives), the no-overlap invariant (no decode step inside a
synthesis window), the bounded queue, and honest refusal when the pause
cannot be proven. mlx/metal behavior is hardware work, not this file.
"""

import http.server
import json
import sys
import threading
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve._mlxlm_server import (  # noqa: E402
    YIELD_CONTROL_PATH,
    YIELD_SECRET_HEADER,
    YieldGate,
    install_yield_gate,
)
from mlx_omarchy_assistant.speech_yield import (  # noqa: E402
    MAX_PENDING_SPEAKS,
    SPEAK_PARK_ACK_SECONDS,
    SpeechYieldScheduler,
    YieldClient,
)


class GateProtocolTests(unittest.TestCase):
    def setUp(self):
        self.gate = YieldGate(hold_max_seconds=0.5)

    def test_probe_binds_secret_and_reports_gate(self):
        status, payload = self.gate.control("probe", "s1", 1)
        self.assertEqual(status, 200)
        self.assertTrue(payload["gate"])
        self.assertTrue(payload["bound"])
        self.assertTrue(payload["secret_ok"])
        status, payload = self.gate.control("probe", "other", 1)
        self.assertFalse(payload["secret_ok"])
        self.assertEqual(payload["bound"], True)

    def test_hold_when_quiescent_holds_and_blocks_future_steps(self):
        self.gate.bind_secret("s1")
        status, payload = self.gate.control("hold", "s1", 1)
        self.assertEqual((status, payload["held"]), (200, True))
        # A decode step must not start while the desire is set: before_step
        # parks (bounded by hold_max) instead of stepping.
        started = threading.Event()
        stepped = threading.Event()

        def step():
            started.set()
            self.gate._before_step()
            stepped.set()

        thread = threading.Thread(target=step, daemon=True)
        thread.start()
        self.assertTrue(started.wait(1))
        self.assertFalse(stepped.wait(0.3))
        self.gate.control("release", "s1", 1)
        self.assertTrue(stepped.wait(1))

    def test_wrong_secret_is_refused_without_touching_desire(self):
        self.gate.bind_secret("s1")
        status, _ = self.gate.control("hold", "nope", 0.2)
        self.assertEqual(status, 403)
        with self.gate._cond:
            self.assertFalse(self.gate._desire)

    def test_park_ack_timeout_fails_honestly(self):
        self.gate.bind_secret("s1")
        gate = self.gate
        release_entered = threading.Event()

        class SlowWorker:
            def next(self):
                gate._before_step()  # parks forever on hold_max=0.5? no: desire unset
                release_entered.set()
                return ["tok"]

        # Simulate a step that is already running and never checks the gate:
        with gate._cond:
            gate._in_step = True  # a decode step is "in flight"
        result = {}

        def hold():
            result["r"] = gate.control("hold", "s1", 0.2)

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        thread.join(2)
        status, payload = result["r"]
        self.assertEqual(status, 409)
        self.assertFalse(payload["held"])
        self.assertEqual(payload["reason"], "park-ack-timeout")
        with gate._cond:
            self.assertFalse(gate._desire)
        gate._after_step()

    def test_close_releases_parked_thread(self):
        self.gate.bind_secret("s1")
        self.gate.control("hold", "s1", 1)
        parked = threading.Event()

        def step():
            self.gate._before_step()
            parked.set()

        thread = threading.Thread(target=step, daemon=True)
        thread.start()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            with self.gate._cond:
                if self.gate._parked:
                    break
            time.sleep(0.01)
        self.gate.close()
        self.assertTrue(parked.wait(1))


class FakeBatchGenerator:
    """Scripted decode steps; records the cache object it holds."""

    def __init__(self, tokens, log):
        self._tokens = list(tokens)
        self.log = log
        self.cache = {"kv": object()}  # identity-checked across parks

    def next(self):
        if not self._tokens:
            return []
        token = self._tokens.pop(0)
        self.log.append(("step", token))
        return [token]

    def insert_segments(self, **kwargs):
        self.log.append(("insert",))


class _GatedHandlerStub(http.server.BaseHTTPRequestHandler):
    """Control-route handler built the way install_yield_gate builds it."""

    gate = None

    def do_POST(self):  # noqa: N802 (stdlib API)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        secret = self.headers.get(YIELD_SECRET_HEADER)
        if secret:
            self.gate.bind_secret(secret)
        status, payload = self.gate.control(body.get("action"),
                                            body.get("secret"),
                                            body.get("ack_timeout", 5))
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class InstallAndWorkerTests(unittest.TestCase):
    def test_install_wraps_generator_and_refuses_moved_signature(self):
        import types

        class FakeAPIHandler(http.server.BaseHTTPRequestHandler):
            pass

        class FakeServerModule:
            ThreadingHTTPServer = http.server.ThreadingHTTPServer
            APIHandler = FakeAPIHandler

        module = FakeServerModule()

        class BatchGenerator:
            def next(self):
                return []

        module.BatchGenerator = BatchGenerator

        def _run_http_server(host, port, response_generator,
                             server_class=None, handler_class=None):
            module.handler_class = handler_class
            return "server"

        module._run_http_server = _run_http_server
        gate = YieldGate()
        install_yield_gate(module, gate)
        self.assertTrue(module.BatchGenerator._omarchy_yield_wrapped)
        self.assertIsNot(module.BatchGenerator, BatchGenerator)
        # The wrapped _run_http_server must receive the gated handler class.
        module._run_http_server("127.0.0.1", 0, None)
        self.assertIsNotNone(module.handler_class)

        # A moved BatchGenerator.next must refuse the launch.
        class Moved:
            def next(self, extra):
                return []

        module2 = FakeServerModule()
        module2.BatchGenerator = Moved
        module2._run_http_server = _run_http_server
        with self.assertRaises(RuntimeError):
            install_yield_gate(module2, YieldGate())

    def test_parked_generation_preserves_cache_and_stops_stepping(self):
        """Real thread, real control HTTP, scripted steps: hold stops the
        steps at a boundary, the cache object survives the park, release
        resumes the same stream."""
        log = []
        real = FakeBatchGenerator([1, 2, 3, 4, 5, 6], log)
        gate = YieldGate()

        class FakeAPIHandler(http.server.BaseHTTPRequestHandler):
            pass

        class FakeServerModule:
            ThreadingHTTPServer = http.server.ThreadingHTTPServer
            APIHandler = FakeAPIHandler

        module = FakeServerModule()

        class BatchGenerator:
            def next(self):
                return real.next()

        module.BatchGenerator = BatchGenerator

        def _run_http_server(host, port, response_generator,
                             server_class=http.server.ThreadingHTTPServer,
                             handler_class=None):
            server = server_class(("127.0.0.1", 0), handler_class)
            self.addCleanup(server.shutdown)
            self.addCleanup(server.server_close)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.control_port = server.server_address[1]
            return server

        module._run_http_server = _run_http_server
        install_yield_gate(module, gate)
        module._run_http_server("127.0.0.1", 0, None)

        generator = module.BatchGenerator()
        done = threading.Event()

        def generate():
            while generator.next():
                time.sleep(0.005)
            done.set()

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(log) < 2:
            time.sleep(0.01)
        self.assertGreaterEqual(len(log), 2)

        client = YieldClient("http://127.0.0.1:%d" % self.control_port)
        secret = "turn-secret"
        self.assertTrue(client.probe(secret))
        self.assertTrue(client.hold(secret, SPEAK_PARK_ACK_SECONDS))
        steps_before = len([entry for entry in log if entry[0] == "step"])
        cache_before = real.cache
        time.sleep(0.25)  # parked: no steps may run inside this window
        steps_inside = len([entry for entry in log if entry[0] == "step"])
        client.release(secret)
        self.assertTrue(done.wait(2))
        self.assertEqual(steps_inside, steps_before)
        self.assertIs(real.cache, cache_before)
        tokens = [entry[1] for entry in log if entry[0] == "step"]
        self.assertEqual(tokens, [1, 2, 3, 4, 5, 6])


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.gpu = threading.Lock()
        self.scheduler = SpeechYieldScheduler(self.gpu)

    def test_fast_path_takes_free_gpu(self):
        grant = self.scheduler.enter(threading.Event(), wait=0.1)
        self.assertIsNotNone(grant)
        self.assertTrue(self.gpu.locked())
        grant.release()
        self.assertFalse(self.gpu.locked())

    def test_busy_without_generation_times_out_and_stays_locked(self):
        self.gpu.acquire()
        start = time.monotonic()
        grant = self.scheduler.enter(threading.Event(), wait=0.2)
        elapsed = time.monotonic() - start
        self.assertIsNone(grant)
        self.assertGreaterEqual(elapsed, 0.2)
        self.assertTrue(self.gpu.locked())
        self.gpu.release()

    def test_queue_bound_rejects_the_third_pending(self):
        self.gpu.acquire()
        results = {}

        def enter(name):
            results[name] = self.scheduler.enter(threading.Event(), wait=5)

        first = threading.Thread(target=enter, args=("first",), daemon=True)
        second = threading.Thread(target=enter, args=("second",), daemon=True)
        first.start()
        second.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(self.scheduler._pending) < 2:
            time.sleep(0.01)
        third = self.scheduler.enter(threading.Event(), wait=0.1)
        self.assertIsNone(third)
        self.assertEqual(len(self.scheduler._pending), MAX_PENDING_SPEAKS)
        self.scheduler.refuse_pending()
        first.join(2)
        second.join(2)
        self.assertIsNone(results["first"])
        self.assertIsNone(results["second"])
        self.gpu.release()

    def test_cancel_wakes_pending_request(self):
        self.gpu.acquire()
        cancel = threading.Event()
        result = {}

        def enter():
            result["grant"] = self.scheduler.enter(cancel, wait=5)

        thread = threading.Thread(target=enter, daemon=True)
        thread.start()
        time.sleep(0.1)
        cancel.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(result["grant"])
        self.gpu.release()

    def test_yield_point_serves_pending_with_proven_park_ordering(self):
        """The no-overlap invariant end to end: a fake client acks the hold
        only when the worker is parked; the log must show hold-ack before
        the lock leaves the generator, and no granted synthesis before the
        generation re-took the lock."""
        log = []
        lock = threading.Lock()

        class FakeClient:
            def hold(self, secret, ack_timeout):
                with lock:
                    log.append(("hold-ack",))
                return True

            def release(self, secret):
                with lock:
                    log.append(("release-sent",))

        self.gpu.acquire()
        waiter_result = {}

        def speak():
            grant = self.scheduler.enter(threading.Event(), wait=5)
            waiter_result["grant"] = grant
            if grant is not None:
                with lock:
                    log.append(("synth-start",))
                time.sleep(0.05)
                with lock:
                    log.append(("synth-end",))
                grant.release()

        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not self.scheduler._pending:
            time.sleep(0.01)

        self.scheduler.yield_point(FakeClient(), "s", threading.Event())
        thread.join(2)
        names = [entry[0] for entry in log]
        self.assertIn("hold-ack", names)
        self.assertLess(names.index("hold-ack"), names.index("synth-start"))
        self.assertLess(names.index("synth-end"), names.index("release-sent"))
        # The lock is back with the generator and the queue is drained.
        self.assertTrue(self.gpu.locked())
        self.assertFalse(self.scheduler._pending)

    def test_unprovable_park_refuses_pending_and_keeps_lock(self):
        class RefusingClient:
            def hold(self, secret, ack_timeout):
                return False

            def release(self, secret):
                raise AssertionError("release must not follow a failed hold")

        self.gpu.acquire()
        result = {}

        def speak():
            result["grant"] = self.scheduler.enter(threading.Event(), wait=5)

        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not self.scheduler._pending:
            time.sleep(0.01)
        self.scheduler.yield_point(RefusingClient(), "s", threading.Event())
        thread.join(2)
        self.assertIsNone(result["grant"])
        self.assertTrue(self.gpu.locked())


if __name__ == "__main__":
    unittest.main()
