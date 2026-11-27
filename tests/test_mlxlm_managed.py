"""The mlx-lm shim as a managed pair worker.

Under the assistant manager the shim must claim its reservation before any
weight loads and must exit when the manager's lifetime pipe closes. Without
that, a manager killed hard leaves the chat worker running (it holds the
weights and the GPU) while the reaper frees its unclaimed reservation.

The REAL shim runs in a subprocess against a minimal fake mlx_lm.server (the
version pin, the _tokenize hook and a /v1/models endpoint), spawned through
the REAL manager-side plumbing.
"""

import errno
import os
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVE_ROOT = str(REPO_ROOT / "serve")
sys.path.insert(0, SERVE_ROOT)

from mlx_omarchy_assistant import managed, pairs  # noqa: E402
from mlx_omarchy_serve import budget  # noqa: E402
from tests.test_assistant_pairs import (  # noqa: E402
    CHAT_SMALL_ID, DECISION_ID, BasePairTest, stub_argv_builder)

GiB = 1 << 30
SHIM = [sys.executable, "-m", "mlx_omarchy_serve._mlxlm_server"]

# Importing mlx_lm.server stands in for loading weights: it writes a marker.
FAKE_SERVER = """
import os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
open(os.environ["FAKE_IMPORTED"], "w").close()
__version__ = "0.31.3"
class ResponseGenerator:
    def _tokenize(self, tokenizer, request, args):
        return ([], [], [], None)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *args):
        pass
def main():
    port = int(sys.argv[sys.argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def make_fake_mlx_lm(root: Path) -> Path:
    """A directory to put on PYTHONPATH that provides mlx_lm.server."""
    package = root / "mlx_lm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "server.py").write_text(FAKE_SERVER)
    return root


def wait_until(test, predicate, what, seconds=60):
    deadline = time.monotonic() + seconds
    while not predicate():
        test.assertLess(time.monotonic(), deadline, what)
        time.sleep(0.05)


class ManagedShimTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.home = root / "home"
        self.home.mkdir()
        self.imported = root / "imported"
        self.env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join([str(make_fake_mlx_lm(root / "fake")), SERVE_ROOT]),
            MLX_OMARCHY_HOME=str(self.home),
            MLX_OMARCHY_SERVE_CONTEXT_LIMIT="4096",
            FAKE_IMPORTED=str(self.imported),
        )
        self.argv = SHIM + ["--port", str(managed.free_port())]
        self.manager_end = None

    def _reap(self, proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=15)

    def _close_manager_end(self):
        if self.manager_end is not None:
            os.close(self.manager_end)
            self.manager_end = None

    def _spawn_managed(self, claim_token=None):
        record = budget.admit_and_reserve_batch(
            [("chat", 1024, "")], pair_id="everyday", home=self.home,
            available_bytes=10 * GiB)[0]
        payload = {"name": "chat", "bytes": record["bytes"],
                   "claim_token": claim_token or record["claim_token"],
                   "pair_id": "everyday"}
        lifetime_r, self.manager_end = os.pipe()
        os.set_inheritable(lifetime_r, True)
        self.addCleanup(self._close_manager_end)
        proc = managed.spawn_claiming_worker(
            self.argv, payload, lifetime_read_fd=lifetime_r, env=self.env,
            log_path=self.home / "chat.log")
        os.close(lifetime_r)
        self.addCleanup(self._reap, proc)
        return proc

    def test_claims_before_loading_and_exits_when_the_manager_dies(self):
        proc = self._spawn_managed()
        wait_until(self, self.imported.exists, "worker never reached the server import")
        held = budget.load_reservations(self.home)["chat"]["claimed"]
        self.assertEqual(held["pid"], proc.pid)
        self._close_manager_end()
        wait_until(self, lambda: proc.poll() is not None, "worker outlived its manager")
        self.assertEqual(proc.returncode, -signal.SIGTERM)

    def test_refused_claim_exits_before_anything_loads(self):
        proc = self._spawn_managed(claim_token="not-the-token")
        self.assertEqual(proc.wait(timeout=60), 3)
        self.assertFalse(self.imported.exists())
        self.assertIsNone(budget.load_reservations(self.home)["chat"]["claimed"])

    def test_standalone_launch_serves_without_a_manager(self):
        proc = subprocess.Popen(self.argv, env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self._reap, proc)
        port = self.argv[-1]
        self.assertTrue(managed.wait_health(f"http://127.0.0.1:{port}/v1/models", 60))


class RealShimUnderPairManagerTests(BasePairTest):
    """The real chat worker beside the stub decision worker, started by the
    real PairManager: the claim must pass the manager's identity checks."""

    def test_start_claims_the_chat_worker_and_it_follows_the_manager(self):
        fake = make_fake_mlx_lm(self.home / "fake")
        imported = self.home / "imported"
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        manager = pairs.PairManager(self.home)
        with mock.patch.object(pairs, "_decision_worker_argv", stub_argv_builder("decision")), \
                mock.patch.dict(os.environ, {"PYTHONPATH": str(fake),
                                             "FAKE_IMPORTED": str(imported)}):
            manager.setup("everyday", approve_download=True)
            self.assertEqual(manager.start()["state"], "ready")
        self.addCleanup(manager.stop)
        chat = manager._children["chat"].popen
        claimed = budget.load_reservations(self.home)[CHAT_SMALL_ID]["claimed"]
        self.assertEqual(claimed["pid"], chat.pid)
        # the manager dies: its end of the lifetime pipe closes
        manager._close_lifetime()
        wait_until(self, lambda: chat.poll() is not None, "chat worker outlived its manager")
        self.assertEqual(chat.returncode, -signal.SIGTERM)
        decision = manager._children["decision"].popen
        wait_until(self, lambda: decision.poll() is not None, "decision worker outlived its manager")
        managed.reap_pair_records("everyday", self.home, children=[])
        self.assertNotIn(CHAT_SMALL_ID, budget.load_reservations(self.home))
        self.assertNotIn(DECISION_ID, budget.load_reservations(self.home))


def _run_bounded(case, fn, *, timeout=5.0):
    """Run fn() on a daemon thread and fail if it blocks past `timeout`."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    case.assertFalse(worker.is_alive(), "operation blocked past timeout")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


class ManagedLogHardeningTests(unittest.TestCase):
    """spawn_supervised() and read_log_tail() stay on regular log files:
    never follow symlinks, never block on FIFOs, never leak fds, and never
    read unbounded logs into memory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.lifetime_r, self.lifetime_w = os.pipe()
        self.addCleanup(self._close_lifetime)

    def _close_lifetime(self):
        for attr in ("lifetime_r", "lifetime_w"):
            fd = getattr(self, attr, None)
            if fd is not None:
                os.close(fd)
                setattr(self, attr, None)

    def _claim(self):
        return {"name": "chat", "claim_token": "tok", "bytes": 1024,
                "pair_id": "everyday"}

    def _open_fds(self):
        return sorted(os.listdir("/dev/fd"))

    def test_spawn_supervised_creates_private_regular_log_appends_and_clears_nonblock(self):
        log_path = self.root / "assistant" / "logs" / "worker.log"
        script = (
            "import fcntl, os, sys; "
            "nb = bool(fcntl.fcntl(1, fcntl.F_GETFL) & os.O_NONBLOCK); "
            "print(f'out1 nonblock={nb}'); "
            "sys.stderr.write('err1\\n')"
        )
        self.assertFalse(os.get_inheritable(self.lifetime_r))
        before_fds = self._open_fds()
        proc1 = managed.spawn_supervised(
            [sys.executable, "-c", script],
            lifetime_read_fd=self.lifetime_r,
            env=dict(os.environ),
            log_path=log_path,
            log_root=self.root,
        )
        self.assertEqual(proc1.wait(timeout=15), 0)
        self.assertEqual(self._open_fds(), before_fds)
        self.assertFalse(os.get_inheritable(self.lifetime_r))
        st = os.lstat(log_path)
        self.assertTrue(stat.S_ISREG(st.st_mode))
        self.assertEqual(st.st_mode & 0o077, 0)

        proc2 = managed.spawn_supervised(
            [sys.executable, "-c", "print('out2')"],
            lifetime_read_fd=self.lifetime_r,
            env=dict(os.environ),
            log_path=log_path,
            log_root=self.root,
        )
        self.assertEqual(proc2.wait(timeout=15), 0)
        self.assertEqual(self._open_fds(), before_fds)
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("out1 nonblock=False", text)
        self.assertIn("err1", text)
        self.assertIn("out2", text)

    def test_spawn_and_read_refuse_symlinked_parent_dirs_and_root_escape(self):
        outside = self.root / "outside"
        outside.mkdir()
        secret = outside / "worker.log"
        secret.write_text("OUTSIDE-SECRET\n", encoding="utf-8")
        marker = self.root / "child-ran"
        argv = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]

        home = self.root / "home"
        home.mkdir()
        # 1. Symlinked intermediate `assistant/logs` directory
        (home / "assistant").mkdir()
        os.symlink(outside, home / "assistant" / "logs")
        log_path = home / "assistant" / "logs" / "worker.log"
        before_fds = self._open_fds()
        with self.assertRaises(OSError):
            managed.spawn_supervised(
                argv, lifetime_read_fd=self.lifetime_r,
                env=dict(os.environ), log_path=log_path, log_root=home)
        with self.assertRaises(OSError):
            managed.read_log_tail(log_path, root=home)
        self.assertFalse(marker.exists())
        self.assertEqual(secret.read_text(encoding="utf-8"), "OUTSIDE-SECRET\n")
        self.assertEqual(self._open_fds(), before_fds)

        # 2. Symlinked `assistant` directory itself
        (home / "assistant" / "logs").unlink()
        (home / "assistant").rmdir()
        os.symlink(outside, home / "assistant")
        with self.assertRaises(OSError):
            managed.spawn_supervised(
                argv, lifetime_read_fd=self.lifetime_r,
                env=dict(os.environ), log_path=log_path, log_root=home)
        with self.assertRaises(OSError):
            managed.read_log_tail(log_path, root=home)
        self.assertEqual(self._open_fds(), before_fds)

        # 3. Path escaping root or containing `..` (using a real directory so `..` itself is tested)
        (home / "assistant").unlink()
        (home / "realdir").mkdir()
        for bad in (secret, home / "realdir" / ".." / "escape.log"):
            with self.subTest(bad=str(bad)):
                with self.assertRaises(OSError):
                    managed.spawn_supervised(
                        argv, lifetime_read_fd=self.lifetime_r,
                        env=dict(os.environ), log_path=bad, log_root=home)
                with self.assertRaises(OSError):
                    managed.read_log_tail(bad, root=home)
                self.assertEqual(self._open_fds(), before_fds)

    def _reap(self, proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=15)

    def test_spawn_supervised_terminates_child_if_claim_delivery_fails(self):
        log_path = self.root / "claim-fail.log"
        spawned = {}
        real_popen = subprocess.Popen

        def capture_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            self.addCleanup(self._reap, proc)
            spawned["proc"] = proc
            return proc

        before_fds = self._open_fds()
        with mock.patch.object(subprocess, "Popen", capture_popen), \
                mock.patch.object(managed, "_write_claim",
                                  side_effect=BrokenPipeError(errno.EPIPE, "broken pipe")):
            with self.assertRaises(BrokenPipeError):
                managed.spawn_claiming_worker(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    self._claim(),
                    lifetime_read_fd=self.lifetime_r,
                    env=dict(os.environ),
                    log_path=log_path,
                )
        self.assertIn("proc", spawned)
        self.assertIsNotNone(spawned["proc"].poll())
        self.assertEqual(self._open_fds(), before_fds)
        self.assertFalse(os.get_inheritable(self.lifetime_r))

    def test_spawn_supervised_guards_post_spawn_close_and_terminate_errors(self):
        log_path = self.root / "post-spawn-err.log"
        real_popen = subprocess.Popen
        real_close = os.close

        # 1. Error closing log_fd after Popen still closes claim_r/claim_w once and terminates child
        spawned = []
        closed_fds = []
        fail_next_close = [True]

        def capture_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            self.addCleanup(self._reap, proc)
            spawned.append(proc)
            return proc

        def fail_first_post_spawn_close(fd):
            real_close(fd)
            if spawned:
                closed_fds.append(fd)
                if fail_next_close[0]:
                    fail_next_close[0] = False
                    raise OSError(errno.EIO, "simulated close failure")

        before_fds = self._open_fds()
        with mock.patch.object(subprocess, "Popen", capture_popen), \
                mock.patch.object(os, "close", fail_first_post_spawn_close):
            with self.assertRaises(OSError):
                managed.spawn_claiming_worker(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    self._claim(),
                    lifetime_read_fd=self.lifetime_r,
                    env=dict(os.environ),
                    log_path=log_path,
                )
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())
        self.assertEqual(len(closed_fds), 3)  # log_fd, claim_r, claim_w — each once
        self.assertEqual(len(set(closed_fds)), 3)
        self.assertEqual(self._open_fds(), before_fds)

        # 2. Error closing claim_r closes it once (never double-closes) and still closes claim_w
        spawned2 = []
        closed_post = []

        def capture_popen2(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            self.addCleanup(self._reap, proc)
            spawned2.append(proc)
            return proc

        def fail_second_post_spawn_close(fd):
            real_close(fd)
            if spawned2:
                closed_post.append(fd)
                if len(closed_post) == 2:
                    raise OSError(errno.EIO, "simulated claim_r close failure")

        before_fds = self._open_fds()
        with mock.patch.object(subprocess, "Popen", capture_popen2), \
                mock.patch.object(os, "close", fail_second_post_spawn_close):
            with self.assertRaises(OSError):
                managed.spawn_claiming_worker(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    self._claim(),
                    lifetime_read_fd=self.lifetime_r,
                    env=dict(os.environ),
                    log_path=log_path,
                )
        self.assertEqual(len(spawned2), 1)
        self.assertIsNotNone(spawned2[0].poll())
        self.assertEqual(len(closed_post), 3)  # log_fd, claim_r, claim_w — each once
        self.assertEqual(len(set(closed_post)), 3)
        self.assertEqual(self._open_fds(), before_fds)

        # 3. If terminate(proc) itself raises, WorkerError still returns proc handle
        spawned3 = []

        def capture_popen3(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            self.addCleanup(self._reap, proc)
            spawned3.append(proc)
            return proc

        before_fds = self._open_fds()
        with mock.patch.object(subprocess, "Popen", capture_popen3), \
                mock.patch.object(managed, "_write_claim",
                                  side_effect=BrokenPipeError(errno.EPIPE, "broken pipe")), \
                mock.patch.object(managed, "terminate",
                                  side_effect=OSError(errno.EPERM, "cannot signal")):
            with self.assertRaises(managed.WorkerError) as ctx:
                managed.spawn_claiming_worker(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    self._claim(),
                    lifetime_read_fd=self.lifetime_r,
                    env=dict(os.environ),
                    log_path=log_path,
                )
        self.assertIs(ctx.exception.proc, spawned3[0])
        self.assertEqual(self._open_fds(), before_fds)

    def test_reap_pair_records_ignores_exited_child_with_mismatched_owner_and_terminate_handles_oserror(self):
        home = self.root / "reap-home"
        home.mkdir()
        record = budget.admit_and_reserve_batch(
            [("chat", 1024, "")], pair_id="everyday", home=home,
            available_bytes=10 * GiB)[0]
        exited = subprocess.Popen([sys.executable, "-c", "pass"])
        self.assertEqual(exited.wait(timeout=15), 0)
        # Stale child with old owner token must not clear the newer live owner's record
        stale_child = managed.ManagedChild("chat", exited, "chat", owner="stale-owner")
        self.assertEqual(managed.reap_pair_records("everyday", home, children=[stale_child]), [])
        self.assertIn("chat", budget.load_reservations(home))
        # Matching owner clears the record on verified exit
        owned_child = managed.ManagedChild("chat", exited, "chat", owner=record["owner"])
        self.assertEqual(managed.reap_pair_records("everyday", home, children=[owned_child]), ["chat"])
        self.assertNotIn("chat", budget.load_reservations(home))

        # terminate() treats signaling OSError as unconfirmed exit (False) rather than crashing
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(self._reap, sleeper)
        with mock.patch.object(sleeper, "terminate",
                               side_effect=PermissionError(errno.EPERM, "not permitted")):
            self.assertFalse(managed.terminate(sleeper, timeout=1))

    def test_spawn_supervised_refuses_symlink_without_touching_target_or_leaking_fds(self):
        existing_target = self.root / "existing.txt"
        existing_target.write_bytes(b"original-sentinel\n")
        dangling_target = self.root / "never-created.txt"
        marker = self.root / "child-ran"
        argv = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]

        for name, target in (("existing", existing_target), ("dangling", dangling_target)):
            link = self.root / f"{name}.log"
            os.symlink(target, link)
            for claim in (None, self._claim()):
                with self.subTest(kind=name, claim=claim is not None):
                    before_fds = self._open_fds()
                    with self.assertRaises(OSError) as ctx:
                        managed.spawn_supervised(
                            argv, lifetime_read_fd=self.lifetime_r,
                            env=dict(os.environ), log_path=link,
                            claim_payload=claim,
                        )
                    self.assertEqual(ctx.exception.errno, errno.ELOOP)
                    self.assertFalse(marker.exists())
                    self.assertEqual(self._open_fds(), before_fds)
        self.assertEqual(existing_target.read_bytes(), b"original-sentinel\n")
        self.assertFalse(dangling_target.exists())

    def test_spawn_supervised_refuses_fifo_without_blocking(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO unavailable")
        fifo = self.root / "worker.fifo"
        os.mkfifo(fifo)
        marker = self.root / "child-ran"
        argv = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]

        for with_reader in (False, True):
            for claim in (None, self._claim()):
                with self.subTest(with_reader=with_reader, claim=claim is not None):
                    rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK) if with_reader else None
                    try:
                        before_fds = self._open_fds()
                        with self.assertRaises(OSError) as ctx:
                            _run_bounded(
                                self,
                                lambda: managed.spawn_supervised(
                                    argv, lifetime_read_fd=self.lifetime_r,
                                    env=dict(os.environ), log_path=fifo,
                                    claim_payload=claim,
                                ),
                            )
                        if with_reader:
                            self.assertEqual(ctx.exception.errno, errno.EINVAL)
                            self.assertIn("not a regular file", str(ctx.exception))
                            self.assertEqual(os.read(rfd, 64), b"")
                        else:
                            self.assertEqual(ctx.exception.errno, errno.ENXIO)
                        self.assertFalse(marker.exists())
                        self.assertEqual(self._open_fds(), before_fds)
                    finally:
                        if rfd is not None:
                            os.close(rfd)

    def test_spawn_supervised_refuses_directory_device_and_socket(self):
        marker = self.root / "child-ran"
        argv = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
        log_dir = self.root / "logdir"
        log_dir.mkdir()
        sock_path = self.root / "log.sock"
        sock = socket.socket(socket.AF_UNIX)
        self.addCleanup(sock.close)
        sock.bind(str(sock_path))

        for name, path in (("directory", log_dir),
                           ("device", Path(os.devnull)),
                           ("socket", sock_path)):
            for claim in (None, self._claim()):
                with self.subTest(kind=name, claim=claim is not None):
                    before_fds = self._open_fds()
                    with self.assertRaises(OSError):
                        managed.spawn_supervised(
                            argv, lifetime_read_fd=self.lifetime_r,
                            env=dict(os.environ), log_path=path,
                            claim_payload=claim,
                        )
                    self.assertFalse(marker.exists())
                    self.assertEqual(self._open_fds(), before_fds)

    def test_spawn_supervised_closes_fds_when_popen_fails(self):
        log_path = self.root / "fail-popen.log"
        before_fds = self._open_fds()
        with self.assertRaises(FileNotFoundError):
            managed.spawn_claiming_worker(
                [str(self.root / "does-not-exist-binary")],
                self._claim(),
                lifetime_read_fd=self.lifetime_r,
                env=dict(os.environ),
                log_path=log_path,
            )
        self.assertEqual(self._open_fds(), before_fds)

    def test_read_log_tail_refuses_symlink_fifo_device_directory_and_socket(self):
        secret = self.root / "secret.txt"
        secret.write_text("TOP-SECRET\n", encoding="utf-8")
        link = self.root / "link.log"
        os.symlink(secret, link)
        dangling = self.root / "dangling.log"
        os.symlink(self.root / "missing.txt", dangling)
        log_dir = self.root / "dir.log"
        log_dir.mkdir()
        sock_path = self.root / "read.sock"
        sock = socket.socket(socket.AF_UNIX)
        self.addCleanup(sock.close)
        sock.bind(str(sock_path))

        for name, path in (("symlink", link),
                           ("dangling", dangling),
                           ("device", Path(os.devnull)),
                           ("directory", log_dir),
                           ("socket", sock_path)):
            with self.subTest(kind=name):
                before_fds = self._open_fds()
                with self.assertRaises(OSError):
                    managed.read_log_tail(path)
                self.assertEqual(self._open_fds(), before_fds)

        if hasattr(os, "mkfifo"):
            fifo = self.root / "read.fifo"
            os.mkfifo(fifo)
            for with_writer in (False, True):
                with self.subTest(fifo_with_writer=with_writer):
                    wfd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK) if with_writer else None
                    try:
                        if wfd is not None:
                            os.write(wfd, b"hello\n")
                        before_fds = self._open_fds()
                        with self.assertRaises(OSError) as ctx:
                            _run_bounded(self, lambda: managed.read_log_tail(fifo))
                        self.assertIn("not a regular file", str(ctx.exception))
                        self.assertEqual(self._open_fds(), before_fds)
                    finally:
                        if wfd is not None:
                            os.close(wfd)

    def test_read_log_tail_boundaries_utf8_and_partial_lines(self):
        p = self.root / "tail.log"
        # Empty and small files
        p.write_bytes(b"")
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "")
        p.write_bytes(b"line1\nline2\n")
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "line1\nline2\n")
        # Exact size == max_bytes
        exact = b"a" * 32
        p.write_bytes(exact)
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "a" * 32)
        # size == max_bytes + 1 with leading newline -> aligned, line preserved
        p.write_bytes(b"\n" + b"b" * 32)
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "b" * 32)
        # size == max_bytes + 1 with bare \r at lookback byte -> aligned, line preserved
        p.write_bytes(b"\r" + b"r" * 32)
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "r" * 32)
        # size == max_bytes + 2 with CRLF split right between \r and \n -> no stray leading \n
        p.write_bytes(b"x\r\n" + b"c" * 31)
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "c" * 31)
        # Mid-line truncation followed by CRLF inside _drop_leading_partial_line -> no leading \n
        p.write_bytes(b"prefix\r\n" + b"k" * 20)
        self.assertEqual(managed.read_log_tail(p, max_bytes=25), "k" * 20)
        # size == max_bytes + 1 with non-newline prefix -> single truncated line dropped
        p.write_bytes(b"x" + b"d" * 32)
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "")
        # Single unbroken > max_bytes line ending in \n -> dropped
        p.write_bytes(b"e" * 80 + b"\n")
        self.assertEqual(managed.read_log_tail(p, max_bytes=32), "")
        # Multi-byte UTF-8 ("█" = b"\xe2\x96\x88") split across seek boundary:
        # partial line dropped in bytes before decoding -> zero replacement chars.
        block = "█".encode("utf-8")
        p.write_bytes(block * 30 + b"\n/cache/snap/ok\n")
        tail = managed.read_log_tail(p, max_bytes=32)
        self.assertEqual(tail, "/cache/snap/ok\n")
        self.assertNotIn("\ufffd", tail)
        # Carriage-return progress bar updates (\r) followed by final path
        p.write_bytes(b"step 1\r" * 40 + b"/cache/snap/tqdm\n")
        lines = [ln for ln in managed.read_log_tail(p, max_bytes=48).strip().splitlines()
                 if ln.strip()]
        self.assertEqual(lines[-1], "/cache/snap/tqdm")
        # Invalid UTF-8 earlier in the file decoded safely with errors="replace"
        p.write_bytes(b"warn \xff\xfe\n/cache/snap/utf8\n")
        lines = [ln for ln in managed.read_log_tail(p, max_bytes=64).strip().splitlines()
                 if ln.strip()]
        self.assertEqual(lines[-1], "/cache/snap/utf8")
        # Invalid max_bytes rejected
        for bad in (0, -1, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                managed.read_log_tail(p, max_bytes=bad)

    def test_read_log_tail_bounds_memory_and_reads_on_large_sparse_and_growing_files(self):
        p = self.root / "sparse.log"
        # 64 MiB sparse regular file with trailing lines at the end
        with open(p, "wb") as fh:
            fh.truncate(64 * 1024 * 1024)
            fh.seek(0, os.SEEK_END)
            fh.write(b"\nline-a\nline-b\n/snap/final\n")

        read_bytes = 0
        real_read = os.read

        def counting_read(fd, n):
            nonlocal read_bytes
            chunk = real_read(fd, n)
            read_bytes += len(chunk)
            return chunk

        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            with mock.patch.object(os, "read", counting_read):
                tail = managed.read_log_tail(p)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            if not tracing:
                tracemalloc.stop()
        self.assertEqual(tail.splitlines(), ["line-a", "line-b", "/snap/final"])
        self.assertLessEqual(read_bytes, managed.LOG_TAIL_MAX_BYTES + 2)
        self.assertLess(peak - before, 8 * managed.LOG_TAIL_MAX_BYTES)

        # File that grew after fstat (both stale <= max_bytes and stale > max_bytes):
        # re-anchors once, returns real tail, and stays bounded.
        real_fstat = os.fstat
        for reported_size in (16, managed.LOG_TAIL_MAX_BYTES + 1024):
            with self.subTest(growth_reported_size=reported_size):
                read_bytes = 0

                def stale_fstat(fd, _sz=reported_size):
                    st = real_fstat(fd)
                    return os.stat_result(st[:6] + (_sz,) + st[7:10])

                with mock.patch.object(os, "fstat", stale_fstat), \
                        mock.patch.object(os, "read", counting_read):
                    tail = managed.read_log_tail(p)
                self.assertEqual(tail.splitlines(), ["line-a", "line-b", "/snap/final"])
                self.assertLessEqual(read_bytes, 2 * managed.LOG_TAIL_MAX_BYTES + 4)

        # File that shrank after fstat (both past-EOF seek and mid-window shrink):
        # re-anchors once and returns the full expected tail of the shrunk file.
        shrunk = self.root / "shrunk.log"
        shrunk.write_bytes(b"old\n" * 40 + b"keep-a\nkeep-b\n/snap/shrunk\n")
        shrunk_size = shrunk.stat().st_size
        expected_tail = managed.read_log_tail(shrunk, max_bytes=64)
        self.assertIn("keep-a\nkeep-b\n/snap/shrunk\n", expected_tail)
        for inflated_size in (shrunk_size + 30, shrunk_size + 10_000):
            with self.subTest(shrink_inflated_size=inflated_size):
                read_bytes = 0

                def inflated_fstat(fd, _sz=inflated_size):
                    st = real_fstat(fd)
                    return os.stat_result(st[:6] + (_sz,) + st[7:10])

                with mock.patch.object(os, "fstat", inflated_fstat), \
                        mock.patch.object(os, "read", counting_read):
                    tail = managed.read_log_tail(shrunk, max_bytes=64)
                self.assertEqual(tail, expected_tail)
                self.assertLessEqual(read_bytes, 2 * 64 + 4)

        # Short reads are joined cleanly
        small = self.root / "short.log"
        small.write_bytes(b"old\n" * 60 + b"final-short-line\n")

        def short_read(fd, n):
            return real_read(fd, min(n, 17))

        with mock.patch.object(os, "read", short_read):
            tail = managed.read_log_tail(small, max_bytes=64)
        self.assertEqual(tail.splitlines()[-1], "final-short-line")


if __name__ == "__main__":
    unittest.main()
