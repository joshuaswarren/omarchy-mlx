#!/usr/bin/env python3
"""A live worker is detected, a dead one is not, a mention is neither."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ane_worker_liveness as liveness  # noqa: E402


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def fake_worker(tmp_path):
    """A real binary named like the worker, holding a file open.

    `tail -f` copied to the worker's name reproduces what breaks pgrep: a
    22-character argv[0] whose comm is truncated to 15 bytes, plus one open
    descriptor standing in for the accel device.
    """
    tail = shutil.which("tail")
    if tail is None:
        pytest.skip("no tail binary to stand in for the worker")
    exe = tmp_path / liveness.WORKER_NAME
    shutil.copy(tail, exe)
    held = tmp_path / "accel0"
    held.write_bytes(b"")
    proc = subprocess.Popen(
        [str(exe), "-f", str(held)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc, held
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _pids(name=liveness.WORKER_NAME):
    return {row["pid"] for row in liveness.worker_processes(name)}


def test_live_worker_is_detected(fake_worker):
    proc, held = fake_worker
    assert _wait(lambda: proc.pid in _pids()), "live worker missed"

    snap = liveness.snapshot(device=held)
    assert snap["worker_count"] >= 1
    assert proc.pid in snap["device_holders"]

    comm = Path(f"/proc/{proc.pid}/comm").read_text().strip()
    # The reason pgrep -x reports 0: comm is the truncated name.
    assert comm == liveness.WORKER_NAME[:15]
    assert comm != liveness.WORKER_NAME


def test_dead_worker_is_not_detected(fake_worker):
    proc, held = fake_worker
    assert _wait(lambda: proc.pid in _pids()), "live worker missed"

    proc.kill()
    proc.wait()

    assert _wait(lambda: proc.pid not in _pids()), "dead worker still counted"
    assert proc.pid not in liveness.device_holders(held)


def test_mention_in_a_command_line_is_not_a_worker(tmp_path):
    # What pgrep -f counts and this helper must not: a process that merely
    # names the worker in its arguments. A `sh -c` decoy raced: sh execs
    # its last command, and /proc/<pid>/cmdline reads back empty mid-exec,
    # which failed this test whenever it ran after the fake-worker tests
    # (4/4 in a full run, 0/4 alone). A Python child never execs away, so
    # its argv keeps the mention and argv[0] is the interpreter.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)",
         liveness.WORKER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait(lambda: liveness.WORKER_NAME
                     in Path(f"/proc/{proc.pid}/cmdline").read_bytes().decode()), \
            "decoy lost its mention"
        assert proc.pid not in _pids()
    finally:
        proc.kill()
        proc.wait()
