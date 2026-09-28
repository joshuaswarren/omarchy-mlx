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

import json
import os
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mlx_omarchy_serve import budget

TERMINATE_TIMEOUT_SECONDS = 10.0


class WorkerError(RuntimeError):
    """A managed worker could not be spawned, reached, or stopped."""


@dataclass
class WorkerSpec:
    """Everything a worker-argv builder needs. Builders live in pairs.py
    (real workers) and are monkeypatched by tests (stub workers)."""
    role: str
    model_dir: Path
    port: int
    context_tokens: int
    max_questions: int


@dataclass
class ManagedChild:
    role: str
    popen: subprocess.Popen
    record_name: str  # the reservation record this child claims


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def spawn_supervised(argv: list[str], *, lifetime_read_fd: int, env: dict,
                     log_path: Path,
                     claim_payload: dict | None = None) -> subprocess.Popen:
    """Spawn a supervised child sharing the manager's lifetime pipe. With a
    claim_payload it additionally receives the private claim pipe (pair
    workers); without one it is a plain supervised child (downloads)."""
    claim_r = None
    child_env = dict(env)
    os.set_inheritable(lifetime_read_fd, True)
    child_env[budget.LIFETIME_FD_ENV] = str(lifetime_read_fd)
    if claim_payload is not None:
        claim_r, claim_w = os.pipe()
        os.set_inheritable(claim_r, True)
        child_env[budget.CLAIM_FD_ENV] = str(claim_r)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "ab") as log:
            pass_fds = (lifetime_read_fd,) if claim_r is None \
                else (claim_r, lifetime_read_fd)
            proc = subprocess.Popen(
                argv, env=child_env, pass_fds=pass_fds,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    except BaseException:
        if claim_r is not None:
            os.close(claim_r)
            os.close(claim_w)
        raise
    if claim_payload is not None:
        os.close(claim_r)
        try:
            os.write(claim_w, json.dumps(claim_payload).encode("utf-8") + b"\n")
        finally:
            os.close(claim_w)
    return proc


def spawn_claiming_worker(argv: list[str], claim_payload: dict, *,
                          lifetime_read_fd: int, env: dict,
                          log_path: Path) -> subprocess.Popen:
    return spawn_supervised(argv, lifetime_read_fd=lifetime_read_fd, env=env,
                            log_path=log_path, claim_payload=claim_payload)


def terminate(proc: subprocess.Popen, timeout: float = TERMINATE_TIMEOUT_SECONDS) -> bool:
    """Bounded SIGTERM -> SIGKILL. Returns True only when the exit is
    verified (poll() returned); callers may release the worker's reservation
    only on that."""
    if proc.poll() is not None:
        return True
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
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
            with urllib.request.urlopen(url, timeout=1) as resp:
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
    children = {child.record_name: child.popen for child in children}
    cleared: list[str] = []
    for name, record in budget.pair_records(pair_id, home).items():
        verified_dead = False
        tracked = children.get(name)
        if tracked is not None and tracked.poll() is not None:
            verified_dead = True
        elif tracked is None and record.get("claimed") is not None \
                and not budget.identity_alive(record["claimed"]):
            verified_dead = True
        if not verified_dead:
            continue
        try:
            budget.clear_reservation(name, home=home, owner=record["owner"])
        except budget.BudgetError:
            continue
        cleared.append(name)
    return cleared
