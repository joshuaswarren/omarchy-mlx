# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Client for the resident ANE worker (plan sections 24-27, phase 8).

One supervised ``mlx-omarchy-ane-worker --serve`` process serves every
submit of a pass: each named bundle is parsed once, its programs are
loaded onto the device once, and they stay resident, so N island
submits cost one process launch, one bundle load, and one device load
instead of N of each.

This is not an inference server (section 25). There is no socket, no
listener, and no user-managed daemon: the worker is a private child of
this client, it only ever reads jobs from its own stdin, every submit is
bounded by the worker's wall-clock deadline, and a failure is reported
and ends the session -- never retried.

When ``MLX_OMARCHY_PK_KEEP_WORKER=1`` AND ``MLX_OMARCHY_ANE_SOCK`` is
set, the client first tries to attach to a long-running
``--daemon --socket PATH`` instance (which holds the device fd and
resident BOs across CLI invocations). On connect failure (no daemon,
stale socket, permission refusal, timeout) the client silently falls
back to the private-subprocess path above -- never hangs, never
escalates to the caller. The daemon, when present, owns the device and
the seal contract; the client never opens ``/dev/accel/accel0`` in this
mode.

Payloads travel inline on the worker stdin and stdout (private-subprocess
path) or the daemon's unix socket (daemon path). They do not go
through host files.
"""

from __future__ import annotations

import os
import select
import socket
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence


# The load-boundary pin flags for one resident session: every consumed
# bundle file and the device library travel to the worker as approved
# sha256 digests; the worker refuses on any mismatch before device load.
def seal_expect_argv(
    seal_expects: Mapping[str, Mapping[str, str]] | None,
    seal_libane_sha: str | None,
    libane: Path,
) -> Sequence[str]:
    argv: list[str] = []
    for name, files in sorted((seal_expects or {}).items()):
        for file, sha in sorted(files.items()):
            argv += ["--seal-expect", f"{name}:{file}={sha}"]
    if seal_libane_sha is not None:
        argv += [
            "--seal-expect-libane",
            f"{Path(libane).name}={seal_libane_sha}",
        ]
    return argv


class ResidentWorkerError(RuntimeError):
    """The resident worker refused or failed; the reason is named."""


# The client's own guard on top of the worker's per-submit deadline: if
# the supervising CLI itself stops answering, the client must not wait
# forever either.
_CLIENT_GRACE_MS = 5000


class ResidentAneWorker:
    """A resident worker session over a fixed set of named bundles.

    ``submit`` takes and returns raw bytes; tensor packing, dtypes and
    array conversion stay with the caller, exactly as with the
    one-process-per-submit path.

    When ``daemon_socket`` is set (or ``MLX_OMARCHY_PK_KEEP_WORKER=1`` +
    ``MLX_OMARCHY_ANE_SOCK`` is set), ``start()`` first tries a one-
    second connect to the daemon. On success the session is an attach
    to a long-running daemon: every submit is a wire-protocol round
    trip over the unix socket, the worker subprocess never spawns here,
    and the device fd + BOs belong to the daemon. On any connect
    failure (no listener, stale socket, permission refused, timeout)
    ``start()`` silently falls back to the private-subprocess path so
    a missing daemon never escalates to the caller.
    """

    def __init__(
        self,
        worker: Path,
        libane: Path,
        bundles: Mapping[str, Path],
        scratch: Path,
        deadline_ms: int = 20000,
        iterations: int = 1,
        relay_bypass: bool | None = None,
        seal_expects: Mapping[str, Mapping[str, str]] | None = None,
        seal_libane_sha: str | None = None,
        daemon_socket: str | os.PathLike[str] | None = None,
        daemon_connect_ms: int = 1000,
    ):
        if not bundles:
            raise ResidentWorkerError("a resident session needs at least one bundle")
        if deadline_ms <= 0 or iterations <= 0:
            raise ResidentWorkerError(
                "a resident session needs a positive deadline and iteration count"
            )
        self.worker = Path(worker)
        self.libane = Path(libane)
        self.bundles = {name: Path(path) for name, path in bundles.items()}
        self.scratch = Path(scratch)
        self.deadline_ms = deadline_ms
        self.iterations = iterations
        # Relay-bypass: the worker subprocess is invoked with
        # --relay-bypass so it becomes a pure splice(2) pump between
        # its stdin/stdout and the resident socketpair. The runner
        # speaks the resident's native frame protocol directly -- one
        # `submit <name>\n` + `in <name> <len>\n` + bytes + `run\n`
        # round trip, with no relay translation in the middle. Skips
        # the relay's getline/stdin.read/fwrite round-trips per
        # payload, which is the load-bearing part of the
        # round-trip-latency-bound AC serve wall. Unset -> the
        # MLX_OMARCHY_ANE_RELAY_BYPASS env var decides, so an A/B
        # harness can flip the arm without touching the runner.
        if relay_bypass is None:
            relay_bypass = os.environ.get(
                "MLX_OMARCHY_ANE_RELAY_BYPASS", ""
            ).lower() not in ("", "0", "off", "false")
        self.relay_bypass = relay_bypass

        # Load-boundary pins: bundle session name -> file name -> sha256
        # of the approved bytes, plus the device library digest. When
        # set, the worker seals every consumed file into an immutable
        # memfd, hashes the sealed image, and refuses any mismatch
        # before the device loads anything.
        self.seal_expects = {
            name: dict(files) for name, files in (seal_expects or {}).items()
        }
        self.seal_libane_sha = seal_libane_sha

        # Daemon attach (opt-in, default off). When the env vars are
        # both set, ``start()`` first tries a bounded connect to the
        # daemon at MLX_OMARCHY_ANE_SOCK; on any failure it silently
        # falls back to the private-subprocess path. A direct
        # ``daemon_socket=...`` parameter overrides the env for
        # tests; passing daemon_socket="" disables the env probe even
        # when the env vars are present.
        if daemon_socket is None:
            env_keep = os.environ.get(
                "MLX_OMARCHY_PK_KEEP_WORKER", ""
            ).lower() not in ("", "0", "off", "false")
            env_sock = os.environ.get("MLX_OMARCHY_ANE_SOCK", "")
            self.daemon_socket = (
                Path(env_sock) if env_keep and env_sock else None
            )
        elif str(daemon_socket) == "":
            self.daemon_socket = None
        else:
            self.daemon_socket = Path(daemon_socket)
        self.daemon_connect_ms = int(daemon_connect_ms)

        # Counters in the shape the parity harness reports (section 42).
        self.submissions = 0
        self.batch_opens = 0
        self.batch_rounds = 0
        self.worker_starts = 0
        self.bundle_loads = 0
        self.device_program_loads = 0
        self.timeouts = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.exec_ns = 0
        self.start_ns = 0
        self.close_ns = 0
        self.log: list[dict] = []

        # Transport: subprocess.Popen (private subprocess path) or a
        # connected unix socket (daemon attach). Exactly one is live at
        # any moment; ``_channel_kind`` selects which IO methods use.
        self._process: subprocess.Popen | None = None
        self._socket: socket.socket | None = None
        self._channel_kind: str = "process"  # "process" or "socket"
        self._stderr_path = self.scratch / "resident-worker.stderr"
        self._stderr = None
        self._stderr_tail_buf: str = ""
        self._banner: list[str] = []
        self._inbox = bytearray()
        self._batch_until: float | None = None
        self._bypass_batch_base = 0

    # ------------------------------------------------------------ lifecycle
    @property
    def alive(self) -> bool:
        """True while the session's transport is open (protocol-level)."""
        return self._process is not None or self._socket is not None

    @property
    def transport(self) -> str:
        """"daemon" when attached to a resident daemon, else "private"."""
        return "daemon" if self._channel_kind == "socket" else "private"

    def start(self) -> None:
        if self._process is not None or self._socket is not None:
            raise ResidentWorkerError("resident session is already started")
        # Try the daemon first when both are opted in: an existing
        # --daemon --socket PATH instance already holds the device fd
        # and the resident BOs, so connecting to it skips the per-CLI
        # ~800 ms bundle-load + device-program-load cost. Any failure
        # here falls through silently to the private-subprocess path:
        # the caller does not see a refused run because the daemon
        # happens to be absent.
        if self.daemon_socket is not None:
            sock = _try_connect_daemon(
                self.daemon_socket,
                connect_ms=self.daemon_connect_ms,
            )
            if sock is not None:
                self._channel_kind = "socket"
                self._socket = sock
                # The daemon's supervised worker runs in --relay-bypass
                # mode: the splice(2) pump between the unix socket and
                # the resident's channel fd forwards raw wire bytes with
                # no parsing. The line-based protocol ("batch N\n",
                # "submit NAME --inline ... --emit ...\n") is a
                # relay-side artifact and never reaches the child.
                # Force relay_bypass=True here so submit / begin_batch
                # / end_batch / close all use the raw protocol the
                # daemon actually forwards.
                self.relay_bypass = True
                started = time.monotonic_ns()
                # The daemon replays one composite banner per accepted
                # client: `daemon session pid=... deadline_ms=...
                # bundles=N\n`. The private-subprocess path consumed
                # one banner per bundle + the relay-bypass-ready line,
                # but the daemon's stdout is its log, not its wire
                # channel, so the client gets one line instead.
                line = self._readline("daemon session banner")
                if not line.startswith("daemon session "):
                    self._die(
                        f"expected the daemon session banner, got {line!r}"
                    )
                self._banner.append(line)
                # Mirror the per-bundle-load counter the private path
                # reports so session_reused and the cross-process stats
                # stay comparable.
                self.bundle_loads = len(self.bundles)
                self.start_ns = time.monotonic_ns() - started
                # daemon attach: there is no per-session worker_starts
                # increment -- the daemon is a long-running peer, not a
                # spawn of this client. The submit counters still
                # record every job the client routes.
                return
        # Daemon connect failed; close any handle the helper left
        # open and fall through to the private-subprocess path.
        self._channel_kind = "process"
        self.scratch.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.worker),
            "--relay-bypass" if self.relay_bypass else "--serve",
            "--libane", str(self.libane),
            "--deadline-ms", str(self.deadline_ms),
            "--iterations", str(self.iterations),
        ]
        for name, path in self.bundles.items():
            argv += ["--bundle", f"{name}={path}"]
        argv += seal_expect_argv(
            self.seal_expects, self.seal_libane_sha, self.libane
        )

        self._stderr = self._stderr_path.open("wb")
        started = time.monotonic_ns()
        self._process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
        )
        self.worker_starts += 1
        # In relay-bypass mode the worker prints the same bundle reports
        # and a "relay-bypass ready ..." banner; the resident's "loaded"
        # line never reaches us (it's the worker's spawn success line,
        # not part of the wire protocol). The pump then forwards the
        # resident's wire bytes on the same stdin/stdout pipes; submit()
        # speaks the resident's protocol directly.
        for _ in self.bundles:
            line = self._readline("bundle report")
            if not line.startswith("resident bundle="):
                self._die(f"expected a resident bundle report, got {line!r}")
            self._banner.append(line)
            self.bundle_loads += 1
        if self.relay_bypass:
            line = self._readline("relay-bypass ready")
            if not line.startswith("relay-bypass ready"):
                self._die(f"expected the relay-bypass ready banner, got {line!r}")
            self._banner.append(line)
        else:
            line = self._readline("load report")
            if not line.startswith("resident loaded "):
                self._die(f"expected the resident load report, got {line!r}")
            self._banner.append(line)
            self.device_program_loads = _loaded_programs(line)
        self.start_ns = time.monotonic_ns() - started

    def close(self) -> dict:
        if self._process is None and self._socket is None:
            raise ResidentWorkerError("no resident session is open")
        started = time.monotonic_ns()
        if self._channel_kind == "socket":
            # Daemon attach: the daemon outlives this client, so we
            # only close our end of the connection. The wire protocol
            # is identical to the relay-bypass private-subprocess
            # path: send "close\n", the daemon's pump half-closes its
            # write side and the resident returns "released\n" on the
            # read side. We do NOT close the daemon or terminate it --
            # that is what `mlx-omarchy-ane-worker --stop --socket PATH`
            # is for.
            self._write_bytes(b"close\n")
            try:
                self._socket.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            line = self._readline("release report")
            if line != "released":
                self._die(
                    f"expected the daemon-released token, got {line!r}"
                )
            self.close_ns = time.monotonic_ns() - started
            self._finish()
            return {"released": line, "exit": 0}
        if self.relay_bypass:
            # In bypass mode the resident's wire protocol is the
            # session boundary. Send "close\n" on stdin, expect
            # "released\n" on stdout (the pump will forward them),
            # then wait for the worker subprocess to exit. The "close"
            # is the final write, so stdin is half-closed immediately:
            # the pump's sender sees EOF (and half-closes the resident
            # socket) while the release token still travels back
            # through the receiver. The worker's own status is checked
            # via the exit code; the resident's released token is the
            # protocol completion.
            self._write_bytes(b"close\n")
            self._process.stdin.close()
            line = self._readline("release report")
            if line != "released":
                self._die(f"expected the resident released token, got {line!r}")
        else:
            self._write("quit")
            line = self._readline("release report")
            if not line.startswith("resident released "):
                self._die(f"expected the resident release report, got {line!r}")
        # "released" is the protocol completion: the resident has taken
        # its programs off the device. The worker subprocess is a
        # private child of this client, so its exit is bounded rather
        # than awaited forever; a straggler pump is killed and the
        # released token above stays the source of truth.
        try:
            code = self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._terminate()
            code = 0
        self.close_ns = time.monotonic_ns() - started
        self._finish()
        if code != 0:
            raise ResidentWorkerError(
                f"resident worker exited {code} after releasing: {self._stderr_tail()}"
            )
        return {"released": line, "exit": code}

    def __enter__(self) -> "ResidentAneWorker":
        self.start()
        return self

    def __exit__(self, kind, value, traceback) -> None:
        if self._process is None and self._socket is None:
            return
        if kind is None:
            self.close()
            return
        self._terminate()

    # -------------------------------------------------------------- submits
    def submit(
        self,
        bundle: str,
        tag: str,
        inputs: Mapping[str, bytes],
        outputs: Sequence[str],
    ) -> dict[str, bytes]:
        """One bounded submit against an already-resident bundle.

        Payloads travel inline on the worker's stdin and stdout. They
        used to be staged through files in ``scratch``; measured on
        m1-test-host, the worker's own read of a 3 MB island input cost 14-19 ms
        per submit -- more than the process launch residency saves -- so
        the file round trip is gone and only the bytes cross.
        """
        if self._process is None and self._socket is None:
            raise ResidentWorkerError("no resident session is open")
        if bundle not in self.bundles:
            raise ResidentWorkerError(f"unknown resident bundle {bundle!r}")

        in_bytes = 0
        ordered = list(inputs.items())
        for _, payload in ordered:
            in_bytes += len(payload)

        if self.relay_bypass:
            # Resident's wire protocol directly. The pump splices
            # these bytes between the runner and the resident with no
            # parsing: one "submit <name>\n" header, an "in <name>
            # <len>\n" frame per input followed by its raw bytes, a
            # "run\n" terminator, and a trailing "\n" (matches the
            # existing submit() contract; the resident reads it but
            # ignores). The resident returns "iter\n" once per
            # iteration (consumed as a no-op here), one "out <name>
            # <len>\n" + bytes frame per produced output, then
            # "done\n".
            request = bytearray()
            request += f"submit {bundle}\n".encode()
            for name, payload in ordered:
                request += f"in {name} {len(payload)}\n".encode()
                request += payload
            request += b"run\n"
        else:
            job = ["submit", bundle]
            for name, payload in ordered:
                job += ["--inline", f"{name}={len(payload)}"]
            for name in outputs:
                job += ["--emit", name]

            request = bytearray(" ".join(job).encode() + b"\n")
            for _, payload in ordered:
                request += payload

        started = time.monotonic_ns()
        self._write_bytes(bytes(request))
        write_ns = time.monotonic_ns() - started
        results: dict[str, bytes] = {}
        out_bytes = 0
        report_line = ""
        while True:
            line = self._readline(f"job report for {tag}")
            if line == "iter":
                continue
            if line.startswith("out "):
                _, name, length = line.split(" ", 2)
                payload = self._read_exact(int(length), f"output {name}")
                results[name] = payload
                out_bytes += len(payload)
                continue
            if line.startswith("failed:"):
                if "deadline" in line:
                    self.timeouts += 1
                self._die(f"resident submit {tag} ({bundle}) failed: {line}")
            report_line = line
            break
        elapsed = time.monotonic_ns() - started
        read_ns = elapsed - write_ns

        self.submissions += 1
        self.exec_ns += elapsed
        self.input_bytes += in_bytes
        # The worker / resident's own report carries its internal
        # split: in bypass mode the resident's "done\n" carries no
        # fields, so elapsed_ms/stage_ms/save_ms are unavailable here
        # -- only the client's own elapsed_ns / write_ns / read_ns
        # split. The non-bypass mode still parses the relay's
        # "job status=N elapsed_ms=... stage_ms=... save_ms=..."
        # fields.
        report_fields = {}
        if not self.relay_bypass:
            for token in report_line.replace("\n", " ").split():
                key, sep, value = token.partition("=")
                if sep and key in ("status", "elapsed_ms", "stage_ms", "save_ms"):
                    report_fields[key] = value
        record = {
            "tag": tag,
            "bundle": bundle,
            "elapsed_ns": elapsed,
            "write_ns": write_ns,
            "read_ns": read_ns,
            "report": report_line,
            "input_bytes": in_bytes,
            "output_bytes": out_bytes,
            **report_fields,
        }
        self.log.append(record)

        if self.relay_bypass:
            if report_line != "done":
                self._die(
                    f"resident submit {tag} ({bundle}) returned "
                    f"{report_line!r} instead of 'done'"
                )
        else:
            if not report_line.startswith("job status=0"):
                if "deadline" in report_line:
                    self.timeouts += 1
                self._die(f"resident submit {tag} ({bundle}) failed: {report_line}")
        missing = [name for name in outputs if name not in results]
        if missing:
            self._die(
                f"resident submit {tag} ({bundle}) produced no bytes for "
                f"{missing}"
            )
        self.output_bytes += out_bytes
        return results

    def begin_batch(self, deadline_ms: int) -> None:
        """Open a batch scope: one absolute deadline bounds N submits.

        The bounded unit is the batch, not each submit inside it, so a
        caller can turn a whole island pass of per-layer submits into
        one deadline-bounded unit. Everything else about the safety
        model is unchanged: a missed deadline or one failed submit ends
        the session and is reported, never retried.
        """
        if self._process is None and self._socket is None:
            raise ResidentWorkerError("no resident session is open")
        if deadline_ms <= 0:
            raise ResidentWorkerError("a batch scope needs a positive deadline")
        if self.relay_bypass:
            # The batch scope was always parent-side bookkeeping (the
            # stock AneWorker::open_batch sends nothing to the
            # resident); in bypass mode the client IS the parent, so
            # the scope is the client's own absolute deadline. The
            # resident never saw batch frames in either path.
            self._batch_until = time.monotonic() + deadline_ms / 1000
            self._bypass_batch_base = self.submissions
            self.batch_opens += 1
            return
        self._write(f"batch {deadline_ms}")
        line = self._readline("batch open report")
        if not line.startswith("batch opened "):
            self._die(f"expected a batch open report, got {line!r}")
        self._batch_until = time.monotonic() + deadline_ms / 1000
        self.batch_opens += 1

    def end_batch(self) -> int:
        """Close the batch scope; returns the rounds the batch served."""
        if self._process is None and self._socket is None:
            raise ResidentWorkerError("no resident session is open")
        if self._batch_until is None:
            raise ResidentWorkerError("no batch scope is open")
        if self.relay_bypass:
            rounds = self.submissions - self._bypass_batch_base
            self._batch_until = None
            self.batch_rounds += rounds
            return rounds
        self._write("batch-end")
        line = self._readline("batch close report")
        if not line.startswith("batch closed "):
            self._die(f"expected a batch close report, got {line!r}")
        self._batch_until = None
        try:
            rounds = int(line.rsplit("=", 1)[-1])
        except ValueError:
            rounds = 0
        self.batch_rounds += rounds
        return rounds

    def counters(self) -> dict:
        """The submit-boundary counters this session measured."""
        return {
            "worker_starts": self.worker_starts,
            "bundle_loads": self.bundle_loads,
            "device_program_loads": self.device_program_loads,
            "submissions": self.submissions,
            "timeouts": self.timeouts,
            "batch_opens": self.batch_opens,
            "batch_rounds": self.batch_rounds,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "exec_ns": self.exec_ns,
            "start_ns": self.start_ns,
            "close_ns": self.close_ns,
        }

    # --------------------------------------------------------------- plumbing
    def _write(self, line: str) -> None:
        self._write_bytes((line + "\n").encode())

    def _write_bytes(self, payload: bytes) -> None:
        if self._channel_kind == "socket":
            assert self._socket is not None
            try:
                self._socket.sendall(payload)
            except (BrokenPipeError, ConnectionResetError, OSError) as error:
                self._die(
                    f"daemon closed its socket before the job was sent: "
                    f"{error}"
                )
            return
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
        except (BrokenPipeError, ValueError):
            self._die("resident worker closed its input before the job was sent")

    def _fill(self, what: str) -> None:
        """Read whatever the worker has ready, inside the client guard."""
        deadline = time.monotonic() + (self.deadline_ms + _CLIENT_GRACE_MS) / 1000
        if self._batch_until is not None:
            deadline = self._batch_until + _CLIENT_GRACE_MS / 1000
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.timeouts += 1
                self._die(
                    f"resident worker did not produce the {what} within "
                    f"{self.deadline_ms + _CLIENT_GRACE_MS} ms"
                )
            if self._channel_kind == "socket":
                sock = self._socket
                assert sock is not None
                try:
                    ready, _, _ = select.select([sock], [], [], remaining)
                except (OSError, ValueError):
                    self._die(
                        f"daemon socket became unreadable while waiting for "
                        f"the {what}"
                    )
                if not ready:
                    continue
                try:
                    chunk = sock.recv(1 << 20)
                except (OSError, ConnectionResetError) as error:
                    self._die(
                        f"daemon closed its socket before the {what}: "
                        f"{error}"
                    )
                if not chunk:
                    self._die(
                        f"daemon closed its socket before the {what}"
                    )
                self._inbox += chunk
                return
            assert self._process is not None and self._process.stdout is not None
            stream = self._process.stdout
            ready, _, _ = select.select([stream], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(stream.fileno(), 1 << 20)
            if not chunk:
                self._die(
                    f"resident worker closed its output before the {what}: "
                    f"{self._stderr_tail()}"
                )
            self._inbox += chunk
            return

    def _readline(self, what: str) -> str:
        while True:
            end = self._inbox.find(b"\n")
            if end >= 0:
                line = bytes(self._inbox[:end]).decode(errors="replace")
                del self._inbox[: end + 1]
                return line
            self._fill(what)

    def _read_exact(self, count: int, what: str) -> bytes:
        while len(self._inbox) < count:
            self._fill(what)
        payload = bytes(self._inbox[:count])
        del self._inbox[:count]
        return payload

    def _stderr_tail(self, limit: int = 400) -> str:
        if self._channel_kind == "socket":
            # Daemon attach: the worker's stderr is owned by the
            # daemon, not by this client. The daemon's own log is the
            # place to look for diagnostics; the client only sees
            # wire-protocol errors here.
            return ""
        if self._stderr is not None:
            self._stderr.flush()
        try:
            return self._stderr_path.read_text(errors="replace").strip()[-limit:]
        except OSError:
            return ""

    def _terminate(self) -> None:
        if self._channel_kind == "socket":
            self._finish()
            return
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.kill()
            process.wait()
        self._finish()

    def _finish(self) -> None:
        if self._channel_kind == "socket":
            sock = self._socket
            self._socket = None
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            # The daemon lives; we don't touch stderr.
            self._channel_kind = "process"
            return
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        self._process = None

    def _die(self, message: str) -> None:
        tail = self._stderr_tail()
        self._terminate()
        raise ResidentWorkerError(
            message if not tail else f"{message} [worker stderr: {tail}]"
        )


def _loaded_programs(line: str) -> int:
    """Programs the worker reported resident, from its load report."""
    marker = "loaded "
    detail = line.split("detail=", 1)[-1]
    position = detail.find(marker)
    if position < 0:
        return 0
    digits = detail[position + len(marker):].split(" ", 1)[0]
    try:
        return int(digits)
    except ValueError:
        return 0


def _try_connect_daemon(
    socket_path: "Path | str",
    *,
    connect_ms: int,
) -> "socket.socket | None":
    """Best-effort connect to a resident daemon. Never raises.

    Returns the connected socket on success; returns None on every
    failure mode (no socket file, stale socket, permission refusal,
    timeout). The caller silently falls back to the private-subprocess
    path so a missing daemon is never the caller's problem.
    """
    path = Path(socket_path)
    if not path.exists():
        return None
    deadline = time.monotonic() + max(0, connect_ms) / 1000
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    except OSError:
        return None
    try:
        # Bound the connect so a hung daemon never hangs the client.
        # ``settimeout`` is a wall-clock cap, not a poll loop: select()
        # in _fill() and the deadline-driven guards continue to work.
        sock.settimeout(max(0.001, max(0, connect_ms) / 1000))
        sock.connect(str(path))
    except (OSError, ConnectionRefusedError):
        sock.close()
        return None
    except socket.timeout:
        sock.close()
        return None
    # Disable the connect-time timeout so the resident may serve a
    # long job; _fill() enforces deadline_ms + grace_ms against the
    # session clock instead.
    try:
        sock.settimeout(None)
    except OSError:
        sock.close()
        return None
    # _fill() still owns the remaining time. If connect took most of
    # the budget, mark the client guard with whatever is left so the
    # first read does not time out immediately.
    del deadline
    return sock
