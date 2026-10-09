#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Stamp the built ANE worker's SHA-256 into the Parakeet runtime pin of a wheel.

Usage:
    python3 stamp_worker_pin.py WHEEL

The worker (mlx/bin/mlx-omarchy-ane-worker) compiles inside the wheel build and
is not byte-reproducible across releases, so its digest is knowable only after
the wheel exists. This writes it into ``assets.worker`` of the pin the wheel
ships, fixes the pin's RECORD line, and replaces the wheel atomically (a
temporary file beside it, then ``os.replace``), so a crash cannot leave a
truncated wheel at the final path. A wheel without a worker (non-aarch64) is
left untouched. A pin that already names a different worker is an error.
"""

import base64
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

WORKER_KEY = "mlx-omarchy-ane-worker"


def stamp(wheel: Path) -> str | None:
    """Return the stamped worker sha256, or None when the wheel has no worker."""
    worker_name = worker_sha = pin_name = record_name = None
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if name.endswith("bin/mlx-omarchy-ane-worker"):
                worker_name, worker_sha = name, hashlib.sha256(zf.read(name)).hexdigest()
            elif name.endswith("share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json"):
                pin_name = name
            elif name.endswith(".dist-info/RECORD"):
                record_name = name
        if worker_name is None:
            return None
        if pin_name is None or record_name is None:
            raise SystemExit("wheel lacks the parakeet pin or RECORD; cannot stamp the worker pin")
        pin = json.loads(zf.read(pin_name))
        record = zf.read(record_name).decode()
        entries = [(info, zf.read(info.filename)) for info in zf.infolist()]
    prior = pin.get("assets", {}).get("worker", {}).get(WORKER_KEY)
    if prior is not None and prior != worker_sha:
        raise SystemExit(f"shipped pin already names a different worker: {prior} != {worker_sha}")
    pin.setdefault("assets", {})["worker"] = {WORKER_KEY: worker_sha}
    pin.setdefault("provenance", {})["worker"] = (
        "sha256 stamped by scripts/build-wheel.sh from this wheel's own built "
        "mlx/bin/mlx-omarchy-ane-worker; the worker compiles per release and is "
        "not byte-reproducible across releases"
    )
    pin_data = json.dumps(pin, indent=2).encode() + b"\n"
    digest = base64.urlsafe_b64encode(hashlib.sha256(pin_data).digest()).rstrip(b"=").decode()
    record_data = "\n".join(
        f"{pin_name},sha256={digest},{len(pin_data)}" if line.split(",", 1)[0] == pin_name else line
        for line in record.splitlines()
    ).encode() + b"\n"
    tmp = wheel.with_name(wheel.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
            for info, data in entries:
                if info.filename == pin_name:
                    data = pin_data
                elif info.filename == record_name:
                    data = record_data
                out.writestr(info, data)
        os.replace(tmp, wheel)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return worker_sha


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    wheel = Path(argv[1])
    sha = stamp(wheel)
    if sha is None:
        print("skip: wheel ships no ANE worker (non-aarch64 build)")
    else:
        print(f"[receipt] worker pinned: sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
