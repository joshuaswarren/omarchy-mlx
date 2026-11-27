"""Bounded regular-file JSON object reader for Laya manifest and config files."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

# A converted manifest or encoder config is a few KiB; cap reads to bound
# corrupt or hostile files (the same cap convert applies).
_JSON_MAX_BYTES = 1 << 20


def read_json_object(path: Path) -> dict:
    """Read a small JSON object from a regular file.

    O_NONBLOCK keeps a FIFO from blocking the open; symlinks are followed, so
    a Hugging Face snapshot's blob links load. Reads at most cap + 1 bytes."""
    with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK)) as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise ValueError("%s is not a regular file" % path)
        raw = f.read(_JSON_MAX_BYTES + 1)
    if len(raw) > _JSON_MAX_BYTES:
        raise ValueError("%s exceeds %d bytes" % (path, _JSON_MAX_BYTES))
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("%s is not a JSON object" % path)
    return data
