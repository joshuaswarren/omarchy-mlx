"""Stopping the assistant server on a signal.

The server runs as a real subprocess (CPU only, no models, no network). A
stub voice worker stands in for the GPU synthesis worker: it is busy in one
long job when the signals arrive."""

import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SERVE = Path(__file__).resolve().parent.parent / "serve"
sys.path.insert(0, str(SERVE))
from mlx_omarchy_assistant import synthesis  # noqa: E402

STOP_BOUND = 5.0

HARNESS = r"""
import sys, time
serve, mode, home = sys.argv[1:4]
sys.path.insert(0, serve)
from mlx_omarchy_assistant import __main__ as app, server as srv, synthesis

def busy_worker(conn, assets_dir):
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg.get("type") == "shutdown":
            break
        if msg.get("type") == "speak":
            time.sleep(60)

synthesis.MP_START_METHOD = "fork"
synthesis._worker_main = busy_worker

class Server(srv.AssistantServer):
    def __init__(self, *args):
        super().__init__(*args)
        worker = self.synthesis._ensure_worker()
        worker.conn.send({"type": "speak", "id": 1, "text": "x"})
        print("WORKER", worker.process.pid, flush=True)

    def server_close(self):
        if mode == "hung-stage":
            time.sleep(60)
        super().server_close()

app.AssistantServer = Server
sys.exit(app.main(["--home", home, "--no-browser"]))
"""


def process_gone(pid):
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                           capture_output=True, text=True).stdout.strip()
    return not state or state.startswith("Z")


class ServerSignalTests(unittest.TestCase):
    def run_stop(self, mode, signals):
        with tempfile.TemporaryDirectory() as home:
            proc = subprocess.Popen(
                [sys.executable, "-c", HARNESS, str(SERVE), mode, home],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            guard = threading.Timer(30, proc.kill)
            guard.start()
            self.addCleanup(guard.cancel)
            worker_pid = None
            for line in proc.stdout:
                if line.startswith("WORKER"):
                    worker_pid = int(line.split()[1])
                if line.startswith("MLX Chat is running"):
                    break
            self.assertIsNotNone(worker_pid, "server did not start a worker")
            runtime_file = Path(home) / "assistant" / "application.json"
            self.assertTrue(runtime_file.exists())
            started = time.monotonic()
            for index in range(signals):
                if index:
                    time.sleep(0.1)
                proc.send_signal(signal.SIGTERM)
            code = proc.wait(timeout=STOP_BOUND + 5)
            elapsed = time.monotonic() - started
            stderr = proc.stderr.read()
            proc.stdout.close()
            proc.stderr.close()
            return code, elapsed, stderr, runtime_file.exists(), worker_pid

    def test_three_signals_stop_cleanly_and_remove_the_worker(self):
        code, elapsed, stderr, runtime_left, worker = self.run_stop("busy", 3)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(code, 0, stderr)
        self.assertLess(elapsed, STOP_BOUND)
        self.assertFalse(runtime_left)
        self.assertTrue(process_gone(worker))

    def test_a_hung_stage_cannot_hold_the_stop_past_the_bound(self):
        code, elapsed, stderr, runtime_left, worker = self.run_stop(
            "hung-stage", 3)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(code, 0, stderr)
        self.assertLess(elapsed, STOP_BOUND)
        self.assertFalse(runtime_left)
        self.assertTrue(process_gone(worker))


class SlowKokoro:
    pipe = SimpleNamespace(model=SimpleNamespace(sample_rate=24000))

    def __call__(self, text, voice):
        import numpy as np
        for _ in range(100):
            time.sleep(0.1)
            yield np.zeros(240, dtype=np.float32)


class WorkerSignalTests(unittest.TestCase):
    def test_a_signal_mid_job_ends_the_segment_and_exits_zero(self):
        """The group TERM from a stop script must not kill the worker in
        the middle of a job: it finishes the current chunk and exits."""
        with mock.patch.object(synthesis, "_worker_guard"), \
                mock.patch.object(synthesis, "_kokoro_streamer",
                                  return_value=SlowKokoro()):
            context = multiprocessing.get_context("fork")
            parent, child = context.Pipe()
            worker = context.Process(target=synthesis._worker_main,
                                     args=(child, "unused"), daemon=True)
            worker.start()
        child.close()
        self.addCleanup(worker.kill)
        parent.send({"type": "init", "pack": synthesis.VOICE_ENGINES[0]["id"]})
        parent.send({"type": "speak", "id": 1, "text": "hello",
                     "voice": "af_heart"})
        seen = []
        while not any(m["type"] == "chunk" for m in seen):
            self.assertTrue(parent.poll(10), seen)
            seen.append(parent.recv())
        os.kill(worker.pid, signal.SIGTERM)
        worker.join(3)
        while parent.poll(0.2):
            try:
                seen.append(parent.recv())
            except EOFError:
                break
        self.assertEqual(worker.exitcode, 0)
        self.assertEqual(seen[-1]["type"], "cancelled")
        self.assertLess(len(seen), 50)

    def test_an_idle_worker_exits_zero_on_a_signal(self):
        with mock.patch.object(synthesis, "_worker_guard"):
            context = multiprocessing.get_context("fork")
            parent, child = context.Pipe()
            worker = context.Process(target=synthesis._worker_main,
                                     args=(child, "unused"), daemon=True)
            worker.start()
        child.close()
        self.addCleanup(worker.kill)
        time.sleep(0.3)
        os.kill(worker.pid, signal.SIGTERM)
        worker.join(3)
        self.assertEqual(worker.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
