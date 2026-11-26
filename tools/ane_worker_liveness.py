#!/usr/bin/env python3
"""Worker liveness for the ANE device, measured instead of guessed.

pgrep cannot answer this question for `mlx-omarchy-ane-worker`:

* The name is 22 characters, so `/proc/<pid>/comm` holds only the first 15
  ("mlx-omarchy-ane"). `pgrep -x mlx-omarchy-ane-worker` compares the whole
  truncated name and therefore reports 0 whether or not a worker is alive.
* `pgrep -f mlx-omarchy-ane-worker` matches any command line that merely
  mentions the name, including the shell running the snapshot itself, so it
  reports workers that do not exist.

Both readings are unusable, in opposite directions; see the
`instrumentation_correction` note in
receipts/2026-09-14-encoder-parity-ane.json.

This helper reads the process table with ps and matches `basename(argv[0])`,
then inspects `/proc/<pid>/fd` for descriptors on the accel device. Snapshot
code in receipts predates this module and carries its own copy; new receipts
import `snapshot()` from here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WORKER_NAME = "mlx-omarchy-ane-worker"
DEVICE = "/dev/accel/accel0"


def process_table() -> list[tuple[int, str]]:
    """(pid, command line) for every process, from ps.

    `-ww` because ps truncates the command column at 80 characters when it
    writes to a pipe, which cuts long argv[0] paths in half.
    """
    out = subprocess.run(
        ["ps", "-ww", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        rows.append((int(parts[0]), parts[1]))
    return rows


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def worker_processes(
    name: str = WORKER_NAME, table: list[tuple[int, str]] | None = None
) -> list[dict]:
    """Processes whose argv[0] basename is exactly `name`.

    argv[0], not comm (truncated at 15 bytes) and not the whole command line
    (a mention is not a worker).
    """
    found = []
    for pid, args in process_table() if table is None else table:
        argv0 = args.split(None, 1)[0] if args else ""
        if argv0 and os.path.basename(argv0) == name:
            found.append({"pid": pid, "comm": _comm(pid), "args": args})
    return found


def _identity(path: str | os.PathLike) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    # A char device is the same device wherever it is opened from; a regular
    # file is identified by (st_dev, st_ino).
    return ("rdev", st.st_rdev) if st.st_rdev else ("file", st.st_dev, st.st_ino)


def device_holders(device: str | os.PathLike = DEVICE) -> list[int]:
    """Pids holding an open descriptor on `device`."""
    want = _identity(device)
    if want is None:
        return []
    holders = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fds = list((entry / "fd").iterdir())
        except OSError:
            continue  # gone, or not ours to read
        for fd in fds:
            if _identity(fd) == want:
                holders.append(int(entry.name))
                break
    return holders


def snapshot(name: str = WORKER_NAME, device: str | os.PathLike = DEVICE) -> dict:
    workers = worker_processes(name)
    holders = device_holders(device)
    return {
        "worker_name": name,
        "workers": workers,
        "worker_count": len(workers),
        "device": str(device),
        "device_holders": holders,
        "method": (
            "ps -ww -eo pid=,args= matched on basename(argv[0]); "
            "/proc/<pid>/fd compared by device identity. pgrep -x cannot match "
            "a 22-character name and pgrep -f matches the caller's own shell."
        ),
    }


def main(argv: list[str]) -> int:
    name = argv[1] if len(argv) > 1 else WORKER_NAME
    device = argv[2] if len(argv) > 2 else DEVICE
    json.dump(snapshot(name, device), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
