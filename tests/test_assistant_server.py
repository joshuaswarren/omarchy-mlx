"""AssistantServer setup/resume lifecycle against server_close()."""

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from mlx_omarchy_assistant import server as server_module
from mlx_omarchy_assistant.server import AssistantServer

WAIT = 5


class Manager:
    """PairManager's lifecycle shape: setup() and start() clear the cancel
    flag on entry; a start aborts once cancel is set after that."""

    def __init__(self):
        self.cancelled = threading.Event()
        self.cancel_seen = threading.Event()
        self.in_prep = threading.Event()
        self.leave_prep = threading.Event()
        self.leave_prep.set()
        self.in_start = threading.Event()
        self.leave_start = threading.Event()
        self.leave_start.set()
        self.starts = []

    def cancel(self):
        self.cancelled.set()
        self.cancel_seen.set()
        return True

    def stop(self):
        return {"stopped": True}

    def setup(self, **_):
        self.cancelled.clear()
        self.adopt_saved()
        return {"state": "prepared"}

    def adopt_saved(self):
        self.in_prep.set()
        self.leave_prep.wait(WAIT)

    def start(self):
        self.in_start.set()
        self.leave_start.wait(WAIT)
        self.cancelled.clear()  # PairManager.start() clears cancel on entry
        if self.cancelled.wait(WAIT):
            self.starts.append("cancelled")
            raise RuntimeError("cancelled during health wait")
        self.starts.append("up")
        return {"ready_offline": True}


class Coordinator:
    def __init__(self, on_close=None):
        self.gpu = threading.Lock()
        self.on_close = on_close
        self.closed = False

    def close(self):
        if self.on_close:
            self.on_close()
        self.closed = True


class SetupLifecycleTests(unittest.TestCase):
    def make(self, on_close=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        server = AssistantServer(("127.0.0.1", 0), Path(temp.name))
        server._manager = Manager()
        server._coordinator = Coordinator(on_close)
        return server

    def begin(self, server, kind):
        if kind == "setup":
            server.setup({"pair_id": "everyday"})
        else:
            server.resume()

    def test_setup_and_resume_are_refused_once_close_starts(self):
        for kind in ("setup", "resume"):
            with self.subTest(kind):
                refused = []

                def begin_mid_close():
                    # server_close() is past its setup-thread check here.
                    try:
                        self.begin(server, kind)
                    except RuntimeError as error:
                        refused.append(str(error))

                server = self.make(begin_mid_close)
                server.server_close()
                if server.setup_thread:
                    server.setup_thread.join(WAIT * 2)
                self.assertEqual(refused, ["The assistant is shutting down"])
                self.assertEqual(server._manager.starts, [])
                self.assertIsNone(server.setup_thread)
                with self.assertRaisesRegex(RuntimeError, "shutting down"):
                    self.begin(server, kind)

    def test_close_waits_for_a_setup_launch_in_progress(self):
        server = self.make()
        manager = server._manager
        lock = server._coordinator.gpu
        closer = threading.Thread(target=server.server_close)
        overtaken = []

        class Gpu:
            def acquire(self, blocking=True):
                closer.start()
                # server_close() must not run past a launch that holds the mutex.
                overtaken.append(manager.cancel_seen.wait(0.5))
                return lock.acquire(blocking)

            def release(self):
                lock.release()

        server._coordinator.gpu = Gpu()
        manager.leave_prep.clear()
        server.resume()
        self.assertTrue(manager.cancel_seen.wait(WAIT))
        manager.leave_prep.set()
        closer.join(WAIT * 3)
        self.assertFalse(closer.is_alive())
        self.assertEqual(overtaken, [False])
        self.assertFalse(server.setup_thread.is_alive())
        self.assertEqual(manager.starts, [])
        self.assertTrue(server._coordinator.closed)

    def test_close_during_setup_preparation_skips_the_pair_start(self):
        for kind in ("setup", "resume"):
            with self.subTest(kind):
                server = self.make()
                manager = server._manager
                manager.leave_prep.clear()
                self.begin(server, kind)
                self.assertTrue(manager.in_prep.wait(WAIT))
                closer = threading.Thread(target=server.server_close)
                closer.start()
                self.assertTrue(manager.cancel_seen.wait(WAIT))
                manager.leave_prep.set()
                closer.join(WAIT * 3)
                self.assertFalse(closer.is_alive())
                self.assertEqual(manager.starts, [])
                self.assertEqual(server.setup_state["state"], "error")
                self.assertTrue(server._coordinator.closed)

    def test_close_keeps_cancelling_a_pair_start_that_clears_cancel(self):
        for kind in ("setup", "resume"):
            with self.subTest(kind):
                server = self.make()
                manager = server._manager
                manager.leave_start.clear()
                self.begin(server, kind)
                self.assertTrue(manager.in_start.wait(WAIT))
                closer = threading.Thread(target=server.server_close)
                closer.start()
                self.assertTrue(manager.cancel_seen.wait(WAIT))
                manager.leave_start.set()  # start() now clears that cancel
                closer.join(WAIT * 3)
                self.assertFalse(closer.is_alive())
                self.assertEqual(manager.starts, ["cancelled"])
                self.assertTrue(server._coordinator.closed)

    def test_a_setup_thread_that_fails_to_start_frees_the_gpu(self):
        class Unstartable(threading.Thread):
            def start(self):
                raise RuntimeError("can't start new thread")

        for kind in ("setup", "resume"):
            with self.subTest(kind):
                server = self.make()
                self.addCleanup(server.server_close)
                with mock.patch.object(server_module.threading, "Thread", Unstartable):
                    with self.assertRaisesRegex(RuntimeError, "can't start new thread"):
                        self.begin(server, kind)
                self.assertIsNone(server.setup_thread)
                self.assertIsNone(server.setup_state)
                self.assertTrue(server._coordinator.gpu.acquire(blocking=False))
                server._coordinator.gpu.release()


if __name__ == "__main__":
    unittest.main()
