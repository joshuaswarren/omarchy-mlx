"""Bounded regular-file readers for Laya manifest and config files."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

# A converted manifest or encoder config is a few KiB; cap reads to bound
# corrupt or hostile files.
_JSON_MAX_BYTES = 1 << 20


class NotRegularFileError(ValueError):
    """The path opened to something other than a regular file."""


class FileTooLargeError(ValueError):
    """The file holds more bytes than the read allows."""


def open_regular_file(path: Path):
    """Open path for binary reading and refuse anything but a regular file.

    O_NONBLOCK keeps a FIFO from blocking the open; symlinks are followed, so
    a Hugging Face snapshot's blob links load. The type check runs on the
    opened descriptor, so the path cannot be swapped after the check."""
    f = open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK))
    try:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise NotRegularFileError("%s is not a regular file" % path)
    except BaseException:
        f.close()
        raise
    return f


def read_regular_bytes(path: Path, cap: int = _JSON_MAX_BYTES) -> bytes:
    """Return the bytes of a regular file of at most cap bytes, reading at most cap + 1."""
    with open_regular_file(path) as f:
        raw = f.read(cap + 1)
    if len(raw) > cap:
        raise FileTooLargeError("%s exceeds %d bytes" % (path, cap))
    return raw


def read_json_object(path: Path) -> dict:
    """Read a small JSON object from a regular file."""
    try:
        raw = read_regular_bytes(path)
    except (NotRegularFileError, FileTooLargeError) as exc:
        # The server reports the exception type by name; keep it ValueError.
        raise ValueError(str(exc)) from None
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("%s is not a JSON object" % path)
    return data
