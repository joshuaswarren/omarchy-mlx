"""Managed worker lifecycle for the assistant pair runtime.

Parent-side plumbing only: spawning a worker with a claim payload on a
private inherited pipe, a shared parent-lifetime pipe, bounded termination,
health polling, and verified-exit reaping of pair reservation records.

The claim/watchdog/identity protocol itself lives in mlx_omarchy_serve.budget
(admit_and_reserve_batch / claim_reservation / watch_parent_fd / proc_identity)
because both worker types (mlx_omarchy_laya.server and the _mlxlm_server shim)
import it as children. Nothing here launches a model: the caller supplies the
worker argv.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import stat
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mlx_omarchy_serve import budget

TERMINATE_TIMEOUT_SECONDS = 10.0
LOG_TAIL_MAX_BYTES = 64 * 1024

# Health probes go to loopback workers: never through an environment proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class WorkerError(RuntimeError):
    """A managed worker could not be spawned, reached, or stopped."""

    def __init__(self, message: str, *, proc: subprocess.Popen | None = None):
        super().__init__(message)
        self.proc = proc


@dataclass
class WorkerSpec:
    """Everything a worker-argv builder needs. Builders live in pairs.py
    (real workers) and are monkeypatched by tests (stub workers)."""
    role: str
    model_dir: Path
    port: int
    context_tokens: int
    max_questions: int
    prompt_cache_size: int = 0


@dataclass
class ManagedChild:
    role: str
    popen: subprocess.Popen
    record_name: str  # the reservation record this child claims
    owner: str | None = None  # owner token of the reservation this child belongs to


def free_ports(count: int) -> list[int]:
    """Distinct free loopback ports. Every probe socket stays open until all
    are chosen: probed one at a time, the kernel can hand back the port it
    just gave out, and two workers then race for one port."""
    socks = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socks.append(sock)
            sock.bind(("127.0.0.1", 0))
        return [int(sock.getsockname()[1]) for sock in socks]
    finally:
        for sock in socks:
            sock.close()


def free_port() -> int:
    return free_ports(1)[0]


def _close_fd_once(fd: int | None) -> BaseException | None:
    if fd is None:
        return None
    try:
        os.close(fd)
    except BaseException as exc:
        return exc
    return None


def _open_log_parent(log_path: Path, root: Path, *,
                     create: bool) -> tuple[int, str]:
    """Walk from `root` to `log_path.parent` using dir_fd + O_NOFOLLOW on every
    intermediate component so a symlinked directory below `root` cannot redirect
    log reads or writes outside `root`. Missing intermediate directories below
    `root` are created with mode 0o700. Returns `(parent_dir_fd, leaf_name)`."""
    try:
        rel = log_path.relative_to(root)
    except ValueError as exc:
        raise OSError(errno.EINVAL, "log path escapes managed root",
                      str(log_path)) from exc
    parts = rel.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise OSError(errno.EINVAL, "invalid relative log path", str(log_path))
    if create:
        root.mkdir(parents=True, exist_ok=True)
    dir_flags = (os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK
                 | getattr(os, "O_CLOEXEC", 0))
    step_flags = dir_flags | getattr(os, "O_NOFOLLOW", 0)
    cur_fd: int | None = os.open(root, dir_flags)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=cur_fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, step_flags, dir_fd=cur_fd)
            prev_fd, cur_fd = cur_fd, next_fd
            close_err = _close_fd_once(prev_fd)
            if close_err is not None:
                raise close_err
    except BaseException:
        _close_fd_once(cur_fd)
        raise
    return cur_fd, parts[-1]


def _open_log_append(log_path: Path, *, root: Path | None = None) -> int:
    """Open a regular log file for appending without following symlinks or
    blocking on FIFOs. When `root` is supplied, every directory component below
    `root` is also opened via dir_fd with O_NOFOLLOW (same-uid hard links to
    regular files are still regular files). Newly created log files use mode
    0o600. Returns a cloexec file descriptor with O_NONBLOCK cleared once
    S_ISREG is verified."""
    log_path = Path(log_path)
    flags = (os.O_WRONLY | os.O_CREAT | os.O_APPEND
             | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0))
    if root is not None:
        parent_fd, leaf = _open_log_parent(log_path, Path(root), create=True)
        fd: int | None = None
        try:
            fd = os.open(leaf, flags, 0o600, dir_fd=parent_fd)
        finally:
            close_err = _close_fd_once(parent_fd)
        if close_err is not None:
            _close_fd_once(fd)
            raise close_err
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_path, flags, 0o600)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "log path is not a regular file",
                          str(log_path))
        cur_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if cur_flags & os.O_NONBLOCK:
            fcntl.fcntl(fd, fcntl.F_SETFL, cur_flags & ~os.O_NONBLOCK)
    except BaseException:
        _close_fd_once(fd)
        raise
    return fd


def _read_fd_bounded(fd: int, limit: int) -> bytearray:
    raw = bytearray()
    while len(raw) < limit:
        chunk = os.read(fd, limit - len(raw))
        if not chunk:
            break
        raw += chunk
    return raw


def _drop_leading_partial_line(raw: bytearray) -> bytes:
    """Drop bytes up to and including the first line break (\\n or \\r)."""
    nl = raw.find(b"\n")
    cr = raw.find(b"\r")
    if nl == -1 and cr == -1:
        return b""
    if nl == -1:
        cut = cr
    elif cr == -1:
        cut = nl
    else:
        cut = min(nl, cr)
    if raw[cut:cut + 2] == b"\r\n":
        cut += 1
    return bytes(raw[cut + 1:])


def read_log_tail(log_path: Path, *,
                  max_bytes: int = LOG_TAIL_MAX_BYTES,
                  root: Path | None = None) -> str:
    """Read at most the trailing `max_bytes` of a regular log file without
    following symlinks or blocking on FIFOs. When `root` is supplied, every
    directory component below `root` is also opened via dir_fd with O_NOFOLLOW.
    When truncated from the front, drops a leading partial line in bytes before
    UTF-8 decoding so callers never see a split UTF-8 sequence or a half-line
    path."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError(f"max_bytes must be a positive int, got {max_bytes!r}")
    log_path = Path(log_path)
    flags = (os.O_RDONLY | os.O_NONBLOCK
             | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    if root is not None:
        parent_fd, leaf = _open_log_parent(log_path, Path(root), create=False)
        fd: int | None = None
        try:
            fd = os.open(leaf, flags, dir_fd=parent_fd)
        finally:
            close_err = _close_fd_once(parent_fd)
        if close_err is not None:
            _close_fd_once(fd)
            raise close_err
    else:
        fd = os.open(log_path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise OSError(errno.EINVAL, "log path is not a regular file",
                          str(log_path))
        size = st.st_size
        if size > max_bytes:
            os.lseek(fd, size - max_bytes - 1, os.SEEK_SET)
            raw = _read_fd_bounded(fd, max_bytes + 1)
            need_drop = True
        else:
            raw = _read_fd_bounded(fd, max_bytes)
            need_drop = False
        # If the file grew or shrank after fstat (or st_size was stale),
        # re-anchor once to SEEK_END so we still return a bounded tail.
        limit = max_bytes + 1 if need_drop else max_bytes
        if (need_drop and len(raw) < limit) or (len(raw) == limit and os.read(fd, 1)):
            end = os.lseek(fd, 0, os.SEEK_END)
            if end > max_bytes:
                os.lseek(fd, end - max_bytes - 1, os.SEEK_SET)
                raw = _read_fd_bounded(fd, max_bytes + 1)
                need_drop = True
            else:
                os.lseek(fd, 0, os.SEEK_SET)
                raw = _read_fd_bounded(fd, max_bytes)
                need_drop = False
    finally:
        _close_fd_once(fd)
    if need_drop:
        if not raw:
            data = b""
        elif raw[0] == 10:  # previous byte was \n -> aligned on line boundary
            data = bytes(raw[1:])
        elif raw[0] == 13:  # previous byte was \r (or \r of \r\n)
            data = bytes(raw[2:] if raw[1:2] == b"\n" else raw[1:])
        else:
            data = _drop_leading_partial_line(raw[1:])
    else:
        data = bytes(raw)
    return data.decode("utf-8", errors="replace")


def _write_claim(claim_w: int, claim_bytes: bytes) -> None:
    view = memoryview(claim_bytes)
    while view:
        written = os.write(claim_w, view)
        if written == 0:
            raise OSError(errno.EPIPE, "claim pipe closed")
        view = view[written:]


def spawn_supervised(argv: list[str], *, lifetime_read_fd: int, env: dict,
                     log_path: Path,
                     claim_payload: dict | None = None,
                     log_root: Path | None = None) -> subprocess.Popen:
    """Spawn a supervised child sharing the manager's lifetime pipe. With a
    claim_payload it additionally receives the private claim pipe (pair
    workers); without one it is a plain supervised child (downloads)."""
    claim_bytes = (json.dumps(claim_payload).encode("utf-8") + b"\n"
                   if claim_payload is not None else None)
    log_fd: int | None = _open_log_append(log_path, root=log_root)
    claim_r: int | None = None
    claim_w: int | None = None
    try:
        child_env = dict(env)
        child_env[budget.LIFETIME_FD_ENV] = str(lifetime_read_fd)
        if claim_bytes is not None:
            claim_r, claim_w = os.pipe()
            child_env[budget.CLAIM_FD_ENV] = str(claim_r)
        pass_fds = (lifetime_read_fd,) if claim_r is None \
            else (claim_r, lifetime_read_fd)
        proc = subprocess.Popen(
            argv, env=child_env, pass_fds=pass_fds,
            stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd)
    except BaseException:
        _close_fd_once(log_fd)
        _close_fd_once(claim_r)
        _close_fd_once(claim_w)
        raise

    # Child is now spawned: close each parent fd at most once independently and
    # keep child ownership across any post-spawn failure (conservatively
    # terminating the child even on a parent-side fd close error).
    post_err: BaseException | None = _close_fd_once(log_fd)
    log_fd = None
    if claim_bytes is not None:
        rfd, claim_r = claim_r, None
        err = _close_fd_once(rfd)
        if err is not None and post_err is None:
            post_err = err
        wfd, claim_w = claim_w, None
        try:
            if post_err is None:
                _write_claim(wfd, claim_bytes)
        except BaseException as exc:
            post_err = exc
        finally:
            err = _close_fd_once(wfd)
            if err is not None and post_err is None:
                post_err = err
    if post_err is not None:
        try:
            confirmed = terminate(proc)
        except BaseException as term_exc:
            raise WorkerError(
                f"post-spawn failure ({post_err}) and termination errored "
                f"for child pid {proc.pid} ({term_exc})",
                proc=proc) from post_err
        if not confirmed:
            raise WorkerError(
                f"post-spawn failure ({post_err}) and child pid {proc.pid} "
                "did not confirm termination",
                proc=proc) from post_err
        raise post_err
    return proc


def spawn_claiming_worker(argv: list[str], claim_payload: dict, *,
                          lifetime_read_fd: int, env: dict,
                          log_path: Path,
                          log_root: Path | None = None) -> subprocess.Popen:
    return spawn_supervised(argv, lifetime_read_fd=lifetime_read_fd, env=env,
                            log_path=log_path, claim_payload=claim_payload,
                            log_root=log_root)


def terminate(proc: subprocess.Popen, timeout: float = TERMINATE_TIMEOUT_SECONDS) -> bool:
    """Bounded SIGTERM -> SIGKILL. Returns True only when the exit is
    verified (poll() returned); callers may release the worker's reservation
    only on that."""
    if proc.poll() is not None:
        return True
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    except OSError:
        return proc.poll() is not None
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except OSError:
        return proc.poll() is not None
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def wait_health(url: str, deadline: float, *, abort=None) -> bool:
    """Poll a loopback health endpoint until it answers 2xx, the (seconds)
    deadline passes, or abort() turns true (cancel flag or a child exit) —
    whichever comes first, so a dead worker never costs the full deadline.
    Network errors are expected while the worker boots."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if abort is not None and abort():
            return False
        try:
            with _OPENER.open(url, timeout=1) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def reap_pair_records(pair_id: str, home: Path,
                      children=()) -> list[str]:
    """Remove a pair's reservation records ONLY for workers whose exit is
    verified: a child we track has a poll() result, or a recorded claim
    identity no longer matches any live process (start-time identity, never
    PID alone). Live or uncertain workers RETAIN their reservations — a
    timeout label alone never releases memory."""
    children = {child.record_name: child for child in children if child.record_name}
    cleared: list[str] = []
    for name, record in budget.pair_records(pair_id, home).items():
        verified_dead = False
        tracked = children.get(name)
        if tracked is not None and tracked.owner is not None \
                and tracked.owner != record.get("owner"):
            tracked = None
        if tracked is not None and tracked.popen.poll() is not None:
            verified_dead = True
        elif tracked is None and record.get("claimed") is not None \
                and not budget.identity_alive(record["claimed"]):
            verified_dead = True
        elif tracked is None and record.get("claimed") is None \
                and record.get("parent") is not None \
                and not budget.identity_alive(record["parent"]):
            verified_dead = True
        if not verified_dead:
            continue
        try:
            budget.clear_reservation(name, home=home, owner=record["owner"])
        except budget.BudgetError:
            continue
        cleared.append(name)
    return cleared
