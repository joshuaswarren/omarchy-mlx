# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the resident ANE worker client (phase 8).

The worker is a scripted stand-in that speaks the ``--serve`` protocol:
no ANE hardware, no libane, no device. What is under test is the
client's half of the contract -- one process and one load for many
submits, payloads that really round-trip, and a failure, a death or a
stall that is named and torn down rather than retried or waited on.
"""

import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

_TOOLS_DIR = Path(_TOOLS).resolve()
for _entry in (str(_TOOLS_DIR), str(_TOOLS_DIR / "coreml")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from coreml.ane_resident import (  # noqa: E402
    ResidentAneWorker,
    ResidentWorkerError,
)

# A stand-in for mlx-omarchy-ane-worker --serve. It counts its own
# starts and loads in a file so a test can prove the client did not
# relaunch it, echoes each input payload into the saved outputs so a
# test can prove the bytes crossed, and takes its failure mode from
# MLX_FAKE_MODE.
_FAKE_WORKER = textwrap.dedent(
    '''\
    #!/usr/bin/env python3
    import os
    import sys
    import time
    from pathlib import Path

    mode = os.environ.get("MLX_FAKE_MODE", "ok")
    ledger = Path(os.environ["MLX_FAKE_LEDGER"])
    bundles = {}
    deadline_ms = 2000
    arguments = sys.argv[1:]
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag == "--bundle":
            name, _, directory = arguments[index + 1].partition("=")
            bundles[name] = directory
            index += 2
        elif flag in ("--libane", "--iterations"):
            index += 2
        elif flag == "--deadline-ms":
            deadline_ms = int(arguments[index + 1])
            index += 2
        elif flag == "--serve":
            index += 1
        else:
            index += 2
    with ledger.open("a") as log:
        log.write("start\\n")
    for position, (name, directory) in enumerate(bundles.items()):
        with ledger.open("a") as log:
            log.write("load %s\\n" % name)
        print(
            "resident bundle=%s index=%d name=fake-%s programs=1 "
            "driver_abi=1 graph=deadbeef" % (name, position, name),
            flush=True,
        )
    if mode == "refuse-open":
        sys.stderr.write("fake device refused to load\\n")
        sys.exit(1)
    print(
        "resident loaded pid=%d deadline_ms=%d iterations=1 detail="
        "resident worker loaded %d program(s) from %d bundle(s)"
        % (os.getpid(), deadline_ms, len(bundles), len(bundles)),
        flush=True,
    )

    submits = 0
    batch_open = False
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    def read_line():
        chunks = bytearray()
        while True:
            byte = stdin.read(1)
            if not byte:
                return None
            if byte == b"\\n":
                return chunks.decode()
            chunks += byte

    def read_exact(count):
        payload = bytearray()
        while len(payload) < count:
            chunk = stdin.read(count - len(payload))
            if not chunk:
                raise SystemExit("stdin ended inside a payload")
            payload += chunk
        return bytes(payload)

    while True:
        line = read_line()
        if line is None:
            break
        line = line.strip()
        if not line:
            continue
        if line == "quit":
            print("resident released programs=%d" % len(bundles), flush=True)
            sys.exit(0)
        if line.startswith("batch "):
            if batch_open:
                sys.stderr.write("fake refused double batch open\\n")
                sys.exit(64)
            batch_open = True
            with ledger.open("a") as log:
                log.write("batch open\\n")
            print("batch opened deadline_ms=%s" % line.split()[1], flush=True)
            continue
        if line == "batch-end":
            if not batch_open:
                sys.stderr.write("fake refused batch close without open\\n")
                sys.exit(64)
            batch_open = False
            with ledger.open("a") as log:
                log.write("batch close\\n")
            print("batch closed rounds=%d" % submits, flush=True)
            continue
        tokens = line.split()
        assert tokens[0] == "submit", line
        bundle = tokens[1]
        inline = []
        emits = []
        position = 2
        while position < len(tokens):
            flag = tokens[position]
            argument = tokens[position + 1]
            if flag == "--inline":
                name, _, length = argument.partition("=")
                inline.append((name, int(length)))
            elif flag == "--emit":
                emits.append(argument)
            position += 2
        # Payloads follow the job line in the order it named them.
        payload = b"".join(read_exact(length) for _, length in inline)
        submits += 1
        with ledger.open("a") as log:
            log.write("submit %s\\n" % bundle)
        if mode == "fail-second" and submits == 2:
            print(
                "job status=1 bundle=%s elapsed_ms=3 detail=fake device "
                "refused exec" % bundle,
                flush=True,
            )
            sys.exit(1)
        if mode == "die-second" and submits == 2:
            os._exit(9)
        if mode == "hang-second" and submits == 2:
            time.sleep(600)
        # Echo the submit's own payload back on every emitted output, so
        # the parent can prove these bytes crossed on this submit.
        sys.stdout.flush()
        for name in emits:
            stdout.write(("out %s %d\\n" % (name, len(payload))).encode())
            stdout.write(payload)
            stdout.flush()
        print(
            "job status=0 bundle=%s elapsed_ms=4 iterations=1 "
            "input_bytes=%d output_bytes=%d stage_ms=0 save_ms=0"
            % (bundle, len(payload), len(payload) * len(emits)),
            flush=True,
        )
    print("resident released programs=%d" % len(bundles), flush=True)
    '''
)


class ResidentAneWorkerTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.worker = self.root / "fake-worker"
        self.worker.write_text("".join(_FAKE_WORKER))
        self.worker.chmod(0o755)
        self.ledger = self.root / "ledger.txt"
        self.bundles = {
            "A": self.root / "bundles" / "island-a",
            "C": self.root / "bundles" / "island-c",
        }
        for path in self.bundles.values():
            path.mkdir(parents=True)

    def tearDown(self):
        self._temporary.cleanup()

    def _session(self, mode: str = "ok", deadline_ms: int = 2000):
        import os

        os.environ["MLX_FAKE_MODE"] = mode
        os.environ["MLX_FAKE_LEDGER"] = str(self.ledger)
        return ResidentAneWorker(
            worker=self.worker,
            libane=self.root / "libane.so",
            bundles=self.bundles,
            scratch=self.root / "scratch",
            deadline_ms=deadline_ms,
        )

    def _ledger(self) -> list[str]:
        if not self.ledger.exists():
            return []
        return self.ledger.read_text().split()

    def test_one_process_and_one_load_serve_every_submit(self):
        session = self._session()
        with session:
            for round_index in range(6):
                marker = bytes([0x40 + round_index]) * 8
                results = session.submit(
                    bundle="A" if round_index % 2 == 0 else "C",
                    tag=f"L{round_index:02d}",
                    inputs={"q": marker, "k": marker},
                    outputs=("y",),
                )
                # Each submit's own payload comes back, so the bytes
                # crossed on every submit rather than once at load.
                self.assertEqual(results["y"], marker + marker)

        counters = session.counters()
        self.assertEqual(counters["submissions"], 6)
        self.assertEqual(counters["worker_starts"], 1)
        self.assertEqual(counters["bundle_loads"], 2)
        self.assertEqual(counters["device_program_loads"], 2)
        self.assertEqual(counters["timeouts"], 0)
        self.assertEqual(counters["input_bytes"], 6 * 16)
        self.assertEqual(counters["output_bytes"], 6 * 16)
        # The worker itself saw one start and one load per bundle for
        # all six submits.
        ledger = self._ledger()
        self.assertEqual(ledger.count("start"), 1)
        self.assertEqual(ledger.count("load"), 2)
        self.assertEqual(ledger.count("submit"), 6)

    def test_payloads_never_touch_the_filesystem(self):
        session = self._session()
        with session:
            session.submit(
                bundle="A", tag="L00", inputs={"q": b"\x01\x02"},
                outputs=("y",),
            )
        # Only the worker's stderr log belongs in the scratch directory:
        # payloads go inline, so no submit writes or reads a file.
        written = sorted(
            path.name for path in (self.root / "scratch").rglob("*")
        )
        self.assertEqual(written, ["resident-worker.stderr"])

    def test_a_failed_submit_is_named_and_ends_the_session(self):
        session = self._session(mode="fail-second")
        session.start()
        session.submit(bundle="A", tag="L00", inputs={"q": b"ab"}, outputs=("y",))
        with self.assertRaises(ResidentWorkerError) as raised:
            session.submit(
                bundle="A", tag="L01", inputs={"q": b"cd"}, outputs=("y",)
            )
        self.assertIn("refused exec", str(raised.exception))
        self.assertEqual(session.submissions, 2)
        self.assertEqual(session.timeouts, 0)
        # The session is gone, so a later call is refused rather than
        # silently relaunching a second worker.
        with self.assertRaises(ResidentWorkerError):
            session.submit(
                bundle="A", tag="L02", inputs={"q": b"ef"}, outputs=("y",)
            )
        self.assertEqual(self._ledger().count("start"), 1)

    def test_a_dead_worker_is_reported_with_its_stderr(self):
        session = self._session(mode="die-second")
        session.start()
        session.submit(bundle="A", tag="L00", inputs={"q": b"ab"}, outputs=("y",))
        with self.assertRaises(ResidentWorkerError) as raised:
            session.submit(
                bundle="A", tag="L01", inputs={"q": b"cd"}, outputs=("y",)
            )
        self.assertIn("closed its output", str(raised.exception))

    def test_a_stalled_worker_is_killed_at_the_client_guard(self):
        session = self._session(mode="hang-second", deadline_ms=200)
        session.start()
        session.submit(bundle="A", tag="L00", inputs={"q": b"ab"}, outputs=("y",))
        started = time.monotonic()
        with self.assertRaises(ResidentWorkerError) as raised:
            session.submit(
                bundle="A", tag="L01", inputs={"q": b"cd"}, outputs=("y",)
            )
        waited = time.monotonic() - started
        self.assertIn("did not produce", str(raised.exception))
        self.assertEqual(session.timeouts, 1)
        # Bounded: the client waited the worker deadline plus its grace,
        # not the worker's 600-second sleep.
        self.assertLess(waited, 30)

    def test_batch_scope_bounds_submits_under_one_deadline(self):
        session = self._session()
        with session:
            session.begin_batch(8000)
            for round_index in range(3):
                marker = bytes([0x70 + round_index]) * 8
                results = session.submit(
                    bundle="A" if round_index % 2 == 0 else "C",
                    tag=f"L{round_index:02d}",
                    inputs={"q": marker, "k": marker},
                    outputs=("y",),
                )
                self.assertEqual(results["y"], marker + marker)
            rounds = session.end_batch()

        self.assertEqual(rounds, 3)
        counters = session.counters()
        self.assertEqual(counters["batch_opens"], 1)
        self.assertEqual(counters["batch_rounds"], 3)
        self.assertEqual(counters["submissions"], 3)
        ledger = self._ledger()
        self.assertIn("open", ledger[ledger.index("batch"):])
        self.assertIn("close", ledger[ledger.index("batch", 2):])

    def test_batch_scope_refusals_are_named(self):
        session = self._session()
        with self.assertRaises(ResidentWorkerError):
            session.begin_batch(4000)  # no session open
        with session:
            with self.assertRaises(ResidentWorkerError):
                session.begin_batch(0)  # unbounded batches are a refusal
            with self.assertRaises(ResidentWorkerError):
                session.end_batch()  # no scope open
            session.begin_batch(4000)
            with self.assertRaises(ResidentWorkerError):
                session.begin_batch(4000)  # one scope at a time; ends the session
            with self.assertRaises(ResidentWorkerError):
                session.end_batch()  # the refused open killed the worker
        with self.assertRaises(ResidentWorkerError):
            session.end_batch()  # session closed with the with-block

    def test_a_refused_open_is_named(self):
        session = self._session(mode="refuse-open")
        with self.assertRaises(ResidentWorkerError) as raised:
            session.start()
        self.assertIn("refused to load", str(raised.exception))

    def test_unknown_bundles_and_double_start_are_rejected(self):
        session = self._session()
        session.start()
        with self.assertRaises(ResidentWorkerError):
            session.start()
        with self.assertRaises(ResidentWorkerError):
            session.submit(
                bundle="Z", tag="L00", inputs={"q": b"ab"}, outputs=("y",)
            )
        session.close()
        with self.assertRaises(ResidentWorkerError):
            session.close()

    def test_a_session_needs_a_bundle_and_a_positive_deadline(self):
        with self.assertRaises(ResidentWorkerError):
            ResidentAneWorker(
                worker=self.worker,
                libane=self.root / "libane.so",
                bundles={},
                scratch=self.root / "scratch",
            )
        with self.assertRaises(ResidentWorkerError):
            ResidentAneWorker(
                worker=self.worker,
                libane=self.root / "libane.so",
                bundles=self.bundles,
                scratch=self.root / "scratch",
                deadline_ms=0,
            )


class _FakeDaemon:
    """A unix-socket stand-in for `mlx-omarchy-ane-worker --daemon`.

    Per client: one `daemon session` banner, then the raw resident
    frames. A submit echoes its concatenated inputs as output `y`; a
    `close` is answered with `released` the way the real daemon answers
    a detach, and the next client is served on the same session.
    """

    def __init__(self, path: Path):
        import socket
        import threading

        self.path = path
        self.clients = 0
        self.submits = 0
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(path))
        self._server.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            self.clients += 1
            with conn, conn.makefile("rwb") as stream:
                stream.write(b"daemon session pid=1 deadline_ms=2000 bundles=2\n")
                stream.flush()
                payload = b""
                for raw in stream:
                    line = raw.rstrip(b"\n")
                    if line.startswith(b"in "):
                        payload += stream.read(int(line.rsplit(b" ", 1)[1]))
                    elif line == b"run":
                        self.submits += 1
                        stream.write(b"iter\nout y %d\n" % len(payload))
                        stream.write(payload + b"done\n")
                        stream.flush()
                        payload = b""
                    elif line == b"close":
                        stream.write(b"released\n")
                        stream.flush()

    def stop(self):
        self._server.close()


class DaemonAttachTest(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.worker = self.root / "fake-worker"
        self.worker.write_text("".join(_FAKE_WORKER))
        self.worker.chmod(0o755)
        self.ledger = self.root / "ledger.txt"
        self.bundles = {"A": self.root / "bundles" / "island-a"}
        self.bundles["A"].mkdir(parents=True)
        self.sock = self.root / "ane.sock"
        import os

        os.environ["MLX_FAKE_MODE"] = "ok"
        os.environ["MLX_FAKE_LEDGER"] = str(self.ledger)

    def tearDown(self):
        self._temporary.cleanup()

    def _session(self, daemon_socket=None):
        return ResidentAneWorker(
            worker=self.worker,
            libane=self.root / "libane.so",
            bundles=self.bundles,
            scratch=self.root / "scratch",
            deadline_ms=2000,
            daemon_socket=daemon_socket,
        )

    def _worker_starts(self) -> int:
        if not self.ledger.exists():
            return 0
        return self.ledger.read_text().split().count("start")

    def test_attach_serves_consecutive_clients_without_spawning(self):
        daemon = _FakeDaemon(self.sock)
        try:
            for marker in (b"\x11" * 4, b"\x22" * 4):
                session = self._session(daemon_socket=self.sock)
                session.start()
                self.assertEqual(session.transport, "daemon")
                results = session.submit(
                    bundle="A", tag="L00", inputs={"q": marker, "k": marker},
                    outputs=("y",),
                )
                self.assertEqual(results["y"], marker + marker)
                self.assertEqual(session.close()["released"], "released")
        finally:
            daemon.stop()
        self.assertEqual(daemon.clients, 2)
        self.assertEqual(daemon.submits, 2)
        self.assertEqual(self._worker_starts(), 0)

    def test_missing_socket_falls_back_to_a_private_worker(self):
        session = self._session(daemon_socket=self.sock)
        with session:
            self.assertEqual(session.transport, "private")
            results = session.submit(
                bundle="A", tag="L00", inputs={"q": b"\x01"}, outputs=("y",),
            )
            self.assertEqual(results["y"], b"\x01")
        self.assertEqual(self._worker_starts(), 1)

    def test_stale_socket_falls_back_within_the_connect_bound(self):
        import socket

        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(self.sock))
        stale.close()  # socket file remains, nothing listens
        started = time.monotonic()
        session = self._session(daemon_socket=self.sock)
        with session:
            self.assertEqual(session.transport, "private")
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(self._worker_starts(), 1)

    def test_daemon_is_opt_in_even_when_the_socket_env_is_set(self):
        import os

        daemon = _FakeDaemon(self.sock)
        os.environ.pop("MLX_OMARCHY_PK_KEEP_WORKER", None)
        os.environ["MLX_OMARCHY_ANE_SOCK"] = str(self.sock)
        try:
            session = self._session()
            with session:
                self.assertEqual(session.transport, "private")
        finally:
            os.environ.pop("MLX_OMARCHY_ANE_SOCK", None)
            daemon.stop()
        self.assertEqual(daemon.clients, 0)
        self.assertEqual(self._worker_starts(), 1)


if __name__ == "__main__":
    unittest.main()
