"""The assistant's single-instance lock and its secret-bearing runtime file.

main() runs in-process against a stub server (no sockets, no models). A
path that used to block (a FIFO opened for reading or writing) runs under
an alarm, so a regression fails the test instead of hanging the suite."""

import builtins
import contextlib
import fcntl
import io
import json
import os
import signal
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

SERVE = Path(__file__).resolve().parent.parent / "serve"
sys.path.insert(0, str(SERVE))
from mlx_omarchy_assistant import __main__ as app  # noqa: E402

RUNTIME = {"port": 4321, "cookie": "mlx_chat=session", "csrf": "token"}


def open_fd_count():
    for path in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(path):
            return len(os.listdir(path))
    raise unittest.SkipTest("cannot inspect open file descriptors on this platform")


class Hung(Exception):
    pass


class SpyStream:
    """Wraps a file object to record reads and kernel offset before close."""

    def __init__(self, stream):
        self._stream = stream
        self.reads = 0
        self.offset_at_close = None

    def fileno(self):
        return self._stream.fileno()

    def read(self, size=-1):
        self.reads += 1
        return self._stream.read(size)

    def close(self):
        if self.offset_at_close is None and not self._stream.closed:
            self.offset_at_close = os.lseek(self._stream.fileno(), 0, os.SEEK_CUR)
        return self._stream.close()

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.offset_at_close is None and not self._stream.closed:
            self.offset_at_close = os.lseek(self._stream.fileno(), 0, os.SEEK_CUR)
        return self._stream.__exit__(exc_type, exc, tb)


class StubServer:
    """Records the runtime file as serve_forever saw it, then returns."""

    instances = []

    def __init__(self, address, home):
        self.server_port = 4321
        self.cookie_name = "mlx_chat"
        self.session = "session"
        self.csrf = "token"
        self.seen = None
        self.stopped = False
        self.closed = False
        self.runtime_file = Path(home) / "assistant" / "application.json"
        StubServer.instances.append(self)

    def serve_forever(self):
        info = os.lstat(self.runtime_file)
        self.seen = (stat.S_IMODE(info.st_mode), stat.S_ISREG(info.st_mode),
                     json.loads(self.runtime_file.read_text()))

    def shutdown(self):
        self.stopped = True

    def server_close(self):
        self.closed = True


class MainTests(unittest.TestCase):
    def setUp(self):
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            self.addCleanup(signal.signal, signum, signal.getsignal(signum))
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.directory = self.home / "assistant"
        self.directory.mkdir(mode=0o700)
        self.lock_path = self.directory / "application.lock"
        self.runtime_file = self.directory / "application.json"
        StubServer.instances = []
        self.enterContext(mock.patch.object(app, "AssistantServer", StubServer))
        self.request = self.enterContext(mock.patch.object(app, "request", return_value={}))

    def run_main(self, stderr=None):
        def expired(_signum, _frame):
            raise Hung("main() blocked")

        signal.signal(signal.SIGALRM, expired)
        stream = io.StringIO() if stderr is None else stderr
        signal.alarm(5)
        try:
            with contextlib.redirect_stderr(stream):
                code = app.main(["--home", str(self.home), "--no-browser"])
        except KeyboardInterrupt:
            self.fail("KeyboardInterrupt escaped main()")
        finally:
            signal.alarm(0)
        return code, stream.getvalue()

    def hold_lock(self):
        holder = open(self.lock_path, "ab")
        self.addCleanup(holder.close)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def write_private(self, text, mode=0o600):
        self.runtime_file.write_text(text)
        os.chmod(self.runtime_file, mode)

    def leftovers(self):
        return sorted(p.name for p in self.directory.iterdir()
                      if p.name not in ("application.lock", "application.json"))

    # The lock

    def test_lock_is_created_private(self):
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stat.S_IMODE(os.lstat(self.lock_path).st_mode), 0o600)

    def test_a_symlinked_lock_is_refused_and_its_target_left_alone(self):
        target = self.home / "elsewhere"
        self.lock_path.symlink_to(target)
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("Application lock", stderr)
        self.assertFalse(target.exists())
        self.assertEqual(StubServer.instances, [])

    def test_a_symlinked_lock_to_an_existing_file_is_refused_untouched(self):
        target = self.home / "elsewhere"
        target.write_text("keep")
        self.lock_path.symlink_to(target)
        code, _stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertEqual(target.read_text(), "keep")

    def test_a_fifo_lock_fails_at_once(self):
        os.mkfifo(self.lock_path, 0o600)
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("Application lock", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_a_fifo_lock_with_a_reader_is_refused(self):
        # A reader lets the non-blocking open succeed; only the regular-file
        # check stands between the FIFO and flock.
        os.mkfifo(self.lock_path, 0o600)
        reader = os.open(self.lock_path, os.O_RDONLY | os.O_NONBLOCK)
        self.addCleanup(os.close, reader)
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("is not a regular file", stderr)
        self.assertEqual(StubServer.instances, [])

    def test_open_lock_closes_its_descriptor_when_refusing_a_non_regular_file(self):
        os.mkfifo(self.lock_path, 0o600)
        reader = os.open(self.lock_path, os.O_RDONLY | os.O_NONBLOCK)
        self.addCleanup(os.close, reader)
        before = open_fd_count()
        try:
            lock = app.open_lock(self.lock_path)
        except RuntimeError as caught:
            # Check while the traceback still holds open_lock()'s locals;
            # assertRaises strips __traceback__, which would reap an unclosed file.
            after = open_fd_count()
            self.assertIn("is not a regular file", str(caught))
            self.assertEqual(after, before)
        else:
            lock.close()
            self.fail("open_lock() accepted a FIFO")

    def test_a_directory_lock_fails_cleanly(self):
        self.lock_path.mkdir()
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("Application lock", stderr)

    def test_a_regular_file_at_the_assistant_directory_fails_cleanly(self):
        self.directory.rmdir()
        self.directory.write_text("not a directory")
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat:", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(StubServer.instances, [])

    # Attaching to the running instance

    def test_attaches_with_the_private_runtime_file(self):
        self.hold_lock()
        self.write_private(json.dumps(RUNTIME))
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.request.assert_any_call(RUNTIME, "GET", "/api/session")
        self.assertEqual(StubServer.instances, [])

    def test_a_symlinked_runtime_file_is_refused(self):
        self.hold_lock()
        real = self.home / "real.json"
        real.write_text(json.dumps(RUNTIME))
        os.chmod(real, 0o600)
        self.runtime_file.symlink_to(real)
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn(app.PRIVATE_RUNTIME, stderr)
        self.request.assert_not_called()

    def test_a_readable_runtime_file_is_refused(self):
        self.hold_lock()
        self.write_private(json.dumps(RUNTIME), mode=0o644)
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn(app.PRIVATE_RUNTIME, stderr)
        self.request.assert_not_called()

    def test_a_runtime_file_owned_by_someone_else_is_refused(self):
        self.hold_lock()
        self.write_private(json.dumps(RUNTIME))
        with mock.patch.object(app.os, "getuid", return_value=os.getuid() + 1):
            code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn(app.PRIVATE_RUNTIME, stderr)
        self.request.assert_not_called()

    def test_a_fifo_runtime_file_is_refused_without_blocking(self):
        self.hold_lock()
        os.mkfifo(self.runtime_file, 0o600)
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn(app.PRIVATE_RUNTIME, stderr)
        self.request.assert_not_called()

    def test_a_directory_runtime_file_is_refused_without_leaking_a_descriptor(self):
        self.hold_lock()
        self.runtime_file.mkdir(mode=0o700)
        before = open_fd_count()
        code, stderr = self.run_main()
        after = open_fd_count()
        self.assertEqual(code, 1)
        self.assertIn(app.PRIVATE_RUNTIME, stderr)
        self.assertEqual(after, before)
        self.request.assert_not_called()

    def test_an_oversized_runtime_file_is_refused_unread(self):
        self.hold_lock()
        self.write_private(json.dumps(RUNTIME) + " " * app.RUNTIME_MAX_BYTES)
        real_open = builtins.open
        spies = []

        def spy_open(file, mode="r", *args, **kwargs):
            stream = real_open(file, mode, *args, **kwargs)
            if Path(file) == self.runtime_file and mode == "rb":
                spy = SpyStream(stream)
                spies.append(spy)
                return spy
            return stream

        with mock.patch.object(builtins, "open", side_effect=spy_open):
            code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn(f"exceeds {app.RUNTIME_MAX_BYTES} bytes", stderr)
        self.assertEqual(len(spies), 1)
        self.assertEqual(spies[0].reads, 0)
        self.assertEqual(spies[0].offset_at_close, 0)
        self.request.assert_not_called()

    def test_a_growing_runtime_file_is_refused_after_a_bounded_read(self):
        # The file fits under the cap at fstat() time, then grows past it
        # before read() runs. The unbuffered capped read stops at cap + 1.
        self.hold_lock()
        self.write_private(json.dumps(RUNTIME))
        target_stat = os.lstat(self.runtime_file)
        real_open = builtins.open
        real_fstat = os.fstat
        spies = []
        grew = False

        def spy_open(file, mode="r", *args, **kwargs):
            stream = real_open(file, mode, *args, **kwargs)
            if Path(file) == self.runtime_file and mode == "rb":
                spy = SpyStream(stream)
                spies.append(spy)
                return spy
            return stream

        def grow_after_fstat(fd):
            nonlocal grew
            info = real_fstat(fd)
            if not grew and (info.st_dev, info.st_ino) == (target_stat.st_dev, target_stat.st_ino):
                grew = True
                with real_open(self.runtime_file, "ab") as writer:
                    writer.write(b" " * (app.RUNTIME_MAX_BYTES * 4))
            return info

        with mock.patch.object(builtins, "open", side_effect=spy_open), \
                mock.patch.object(app.os, "fstat", side_effect=grow_after_fstat):
            code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn(f"exceeds {app.RUNTIME_MAX_BYTES} bytes", stderr)
        self.assertTrue(grew)
        self.assertEqual(len(spies), 1)
        self.assertEqual(spies[0].offset_at_close, app.RUNTIME_MAX_BYTES + 1)
        self.request.assert_not_called()

    def test_a_runtime_file_at_the_size_limit_is_read(self):
        self.hold_lock()
        text = json.dumps(RUNTIME)
        self.write_private(text + " " * (app.RUNTIME_MAX_BYTES - len(text)))
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.request.assert_any_call(RUNTIME, "GET", "/api/session")

    # Publishing the runtime file

    def test_publishes_a_private_runtime_file_and_removes_it_on_stop(self):
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        (server,) = StubServer.instances
        self.assertEqual(server.seen, (0o600, True, RUNTIME))
        self.assertTrue(server.stopped)
        self.assertTrue(server.closed)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])

    def test_publishing_replaces_an_existing_runtime_file_atomically(self):
        old = {"port": 1111, "cookie": "mlx_chat=old", "csrf": "old-token"}
        self.write_private(json.dumps(old))
        real_replace = os.replace
        observed = []

        def observe(src, dst):
            observed.append(app.read_runtime(self.runtime_file))
            real_replace(src, dst)
            observed.append(app.read_runtime(self.runtime_file))

        with mock.patch.object(app.os, "replace", side_effect=observe):
            code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(observed, [old, RUNTIME])

    def test_a_stale_readable_runtime_file_is_replaced_private(self):
        self.write_private("{}", mode=0o644)
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(StubServer.instances[0].seen, (0o600, True, RUNTIME))

    def test_a_stale_fifo_runtime_file_is_replaced_without_blocking(self):
        os.mkfifo(self.runtime_file, 0o600)
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(StubServer.instances[0].seen, (0o600, True, RUNTIME))

    def test_a_stale_symlink_is_replaced_and_its_target_left_alone(self):
        target = self.home / "elsewhere"
        target.write_text("keep")
        self.runtime_file.symlink_to(target)
        code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(StubServer.instances[0].seen, (0o600, True, RUNTIME))
        self.assertEqual(target.read_text(), "keep")

    def test_a_directory_in_the_way_stops_the_server_cleanly(self):
        self.runtime_file.mkdir()
        code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat:", stderr)
        self.assertNotIn("Traceback", stderr)
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)
        self.assertTrue(self.runtime_file.is_dir())
        self.assertEqual(self.leftovers(), [])

    def test_a_publish_failure_does_not_hang_on_shutdown(self):
        # Like socketserver.BaseServer.shutdown(), shutdown() deadlocks when
        # serve_forever() never started.
        release = threading.Event()
        self.addCleanup(release.set)

        class BlockingShutdownServer(StubServer):
            def __init__(self, address, home):
                super().__init__(address, home)
                self._served = False

            def serve_forever(self):
                self._served = True
                super().serve_forever()

            def shutdown(self):
                if not self._served:
                    release.wait()
                super().shutdown()

        self.runtime_file.mkdir()
        started = time.monotonic()
        with mock.patch.object(app, "AssistantServer", BlockingShutdownServer), \
                mock.patch.object(app.os, "_exit", side_effect=AssertionError("stop exceeded STOP_BUDGET")) as forced_exit:
            code, stderr = self.run_main()
        elapsed = time.monotonic() - started
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat:", stderr)
        self.assertLess(elapsed, app.STOP_BUDGET / 2)
        forced_exit.assert_not_called()
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)
        self.assertTrue(self.runtime_file.is_dir())
        self.assertEqual(self.leftovers(), [])

    def test_a_signal_during_stop_after_publish_failure_still_cleans_up(self):
        def sentinel(_signum, _frame):
            raise AssertionError("SIGTERM handler was not installed")

        signal.signal(signal.SIGTERM, sentinel)
        signal_sent = threading.Event()

        class SignalDuringCloseServer(StubServer):
            def server_close(self):
                os.kill(os.getpid(), signal.SIGTERM)
                signal_sent.set()
                time.sleep(0.05)
                super().server_close()

        self.runtime_file.mkdir()
        before = open_fd_count()
        with mock.patch.object(app, "AssistantServer", SignalDuringCloseServer):
            code, stderr = self.run_main()
        after = open_fd_count()
        self.assertTrue(signal_sent.is_set())
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat:", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(after, before)
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)
        self.assertTrue(self.runtime_file.is_dir())
        self.assertEqual(self.leftovers(), [])

    def test_a_signal_during_stop_after_normal_return_still_cleans_up(self):
        def sentinel(_signum, _frame):
            raise AssertionError("SIGTERM handler was not installed")

        signal.signal(signal.SIGTERM, sentinel)
        signal_sent = threading.Event()

        class SignalDuringCloseServer(StubServer):
            def server_close(self):
                os.kill(os.getpid(), signal.SIGTERM)
                signal_sent.set()
                time.sleep(0.05)
                super().server_close()

        before = open_fd_count()
        with mock.patch.object(app, "AssistantServer", SignalDuringCloseServer):
            code, stderr = self.run_main()
        after = open_fd_count()
        self.assertTrue(signal_sent.is_set())
        self.assertEqual(code, 0, stderr)
        self.assertEqual(after, before)
        (server,) = StubServer.instances
        self.assertTrue(server.stopped)
        self.assertTrue(server.closed)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])

    def test_a_signal_just_before_rename_leaves_no_files_and_exits_zero(self):
        def sentinel(_signum, _frame):
            raise AssertionError("SIGTERM handler was not installed before publishing")

        signal.signal(signal.SIGTERM, sentinel)
        real_replace = os.replace

        def interrupt_before_rename(src, dst):
            self.assertTrue(Path(src).exists())
            os.kill(os.getpid(), signal.SIGTERM)
            real_replace(src, dst)

        with mock.patch.object(app.os, "replace", side_effect=interrupt_before_rename):
            code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)

    def test_a_signal_just_after_rename_leaves_no_files_and_exits_zero(self):
        def sentinel(_signum, _frame):
            raise AssertionError("SIGINT handler was not installed before publishing")

        signal.signal(signal.SIGINT, sentinel)
        real_replace = os.replace

        def interrupt_after_rename(src, dst):
            real_replace(src, dst)
            self.assertTrue(self.runtime_file.exists())
            os.kill(os.getpid(), signal.SIGINT)

        with mock.patch.object(app.os, "replace", side_effect=interrupt_after_rename):
            code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)

    def test_a_failed_write_leaves_no_temporary_file(self):
        with mock.patch.object(app.json, "dump", side_effect=ValueError("no")):
            code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat: no", stderr)
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])

    def test_a_failed_write_leaves_a_non_file_untouched(self):
        target = self.home / "elsewhere"
        target.write_text("keep")
        self.runtime_file.symlink_to(target)
        with mock.patch.object(app.json, "dump", side_effect=ValueError("no")):
            code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat: no", stderr)
        self.assertTrue(self.runtime_file.is_symlink())
        self.assertEqual(target.read_text(), "keep")
        self.assertEqual(self.leftovers(), [])

    def test_a_cleanup_unlink_error_does_not_mask_the_write_error(self):
        with mock.patch.object(app.json, "dump", side_effect=ValueError("original")), \
                mock.patch.object(app.os, "unlink", side_effect=PermissionError("unlink failed")):
            code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat: original", stderr)
        self.assertNotIn("unlink failed", stderr)

    def test_a_temp_name_collision_leaves_the_existing_file_untouched(self):
        existing = self.directory / "application.json.0000000000000000.tmp"
        existing.write_text("keep")
        with mock.patch.object(app.secrets, "token_hex", return_value="0" * 16):
            code, stderr = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat:", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(existing.read_text(), "keep")
        self.assertFalse(self.runtime_file.exists())
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)

    def test_a_signal_while_marking_stop_does_not_deadlock_or_escape(self):
        main_ident = threading.get_ident()

        class SignalInSetEvent(threading.Event):
            def __init__(self):
                super().__init__()
                self._in_set = False
                self._fired = False
                event = self

                class Condition(threading.Condition):
                    def __enter__(self):
                        result = super().__enter__()
                        if (
                            event._in_set
                            and not event._fired
                            and threading.get_ident() == main_ident
                        ):
                            event._fired = True
                            os.kill(os.getpid(), signal.SIGTERM)
                        return result

                self._cond = Condition(threading.Lock())

            def set(self):
                was_in_set = self._in_set
                self._in_set = True
                try:
                    return super().set()
                finally:
                    self._in_set = was_in_set

        with mock.patch.object(app.threading, "Event", SignalInSetEvent):
            code, stderr = self.run_main()
        self.assertEqual(code, 0, stderr)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])
        self.assertTrue(StubServer.instances[-1].closed)

        def sentinel(_signum, _frame):
            raise AssertionError("SIGTERM handler was not installed")

        signal.signal(signal.SIGTERM, sentinel)

        class SignalOnWrite(io.StringIO):
            def __init__(self):
                super().__init__()
                self.signaled = False

            def write(self, text):
                if not self.signaled:
                    self.signaled = True
                    os.kill(os.getpid(), signal.SIGTERM)
                return super().write(text)

        self.runtime_file.mkdir()
        stream = SignalOnWrite()
        code, stderr = self.run_main(stderr=stream)
        self.assertTrue(stream.signaled)
        self.assertEqual(code, 1)
        self.assertIn("MLX Chat:", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertTrue(StubServer.instances[-1].closed)
        self.assertTrue(self.runtime_file.is_dir())
        self.assertEqual(self.leftovers(), [])

    def test_a_signal_while_serving_calls_shutdown_before_server_close(self):
        serving = threading.Event()
        stop = threading.Event()
        events = []
        cleanup_threads = []

        def cleanup():
            stop.set()
            for worker in cleanup_threads:
                worker.join(timeout=2)

        self.addCleanup(cleanup)

        class LiveServer(StubServer):
            def serve_forever(self):
                super().serve_forever()
                serving.set()
                stop.wait()

            def shutdown(self):
                events.append("shutdown")
                super().shutdown()
                stop.set()

            def server_close(self):
                events.append("server_close")
                super().server_close()

        real_thread = threading.Thread
        signaled = threading.Event()
        serve_thread = None

        def send_term():
            if serving.wait(timeout=2):
                time.sleep(0.05)
                os.kill(os.getpid(), signal.SIGTERM)

        class SignalingThread(real_thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if getattr(kwargs.get("target"), "__name__", None) == "serve_forever":
                    nonlocal serve_thread
                    serve_thread = self
                    cleanup_threads.append(self)

            def join(self, timeout=None):
                if self is serve_thread and not signaled.is_set():
                    signaled.set()
                    sender = real_thread(target=send_term, daemon=True)
                    cleanup_threads.append(sender)
                    sender.start()
                return super().join(timeout)

        with mock.patch.object(app, "AssistantServer", LiveServer), \
                mock.patch.object(app.threading, "Thread", SignalingThread):
            code, stderr = self.run_main()
        self.assertTrue(signaled.is_set())
        self.assertEqual(code, 0, stderr)
        self.assertEqual(events, ["shutdown", "server_close"])
        (server,) = StubServer.instances
        self.assertTrue(server.stopped)
        self.assertTrue(server.closed)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])

    def test_a_signal_right_after_opening_temp_file_leaves_no_files(self):
        def sentinel(_signum, _frame):
            raise AssertionError("SIGTERM handler was not installed before publishing")

        signal.signal(signal.SIGTERM, sentinel)
        signaled = False

        def trace(_frame, _event, _arg):
            nonlocal signaled
            if not signaled and any(name.endswith(".tmp") for name in os.listdir(self.directory)):
                signaled = True
                sys.settrace(None)
                os.kill(os.getpid(), signal.SIGTERM)
                return None
            return trace

        before = open_fd_count()
        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            code, stderr = self.run_main()
        finally:
            sys.settrace(previous)
        after = open_fd_count()
        self.assertTrue(signaled)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(after, before)
        self.assertFalse(self.runtime_file.exists())
        self.assertEqual(self.leftovers(), [])
        (server,) = StubServer.instances
        self.assertFalse(server.stopped)
        self.assertTrue(server.closed)

    def test_stop_server_without_a_runtime_file_closes_cleanly(self):
        stub = StubServer(("127.0.0.1", 0), self.home)
        app.stop_server(stub, None, serving=False)
        self.assertFalse(stub.stopped)
        self.assertTrue(stub.closed)


if __name__ == "__main__":
    unittest.main()
