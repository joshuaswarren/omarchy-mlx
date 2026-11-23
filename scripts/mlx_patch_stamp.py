#!/usr/bin/env python3
"""The apply-mlx-lm-patches stamp, shared by the diag profiling harnesses.

d-0102 p1: SC-11, SC-12 and GAP-E profiled a venv whose mlx-lm carried no
omarchy patches and reported 800 dispatch launches per token where the
patched prod path runs 205. Every harness that profiles mlx-lm must refuse
to run unless the stamp exists next to the mlx_lm package it will import.

apply-mlx-lm-patches.sh writes the stamp through write_stamp() after the
series applies. The stamp records the series id and the write time, so a
stale stamp from an older series is visible in the file itself.
"""
import datetime
import os
import sys

STAMP_NAME = ".omarchy-patches-stamp"


def stamp_path(mlx_lm_file):
    return os.path.join(os.path.dirname(os.path.abspath(mlx_lm_file)), STAMP_NAME)


def require_stamp(mlx_lm_module=None):
    """Exit 3 unless the mlx_lm this process will use carries the stamp."""
    if mlx_lm_module is None:
        import mlx_lm as mlx_lm_module
    p = stamp_path(mlx_lm_module.__file__)
    if os.path.exists(p):
        return p
    sys.stderr.write(
        "refusing: this venv's mlx_lm carries no %s stamp, so it is not the "
        "apply-mlx-lm-patches build; profiling it produces wrong dispatch "
        "counts (d-0102 p1: SC-11/SC-12/GAP-E measured 800 launches/token on "
        "an unpatched mlx-lm where prod runs 205). Run scripts/"
        "apply-mlx-lm-patches.sh on this venv first.\n" % STAMP_NAME)
    raise SystemExit(3)


def write_stamp(mlx_lm_file, series_id):
    p = stamp_path(mlx_lm_file)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("series=%s\nwritten=%s\n"
                 % (series_id, datetime.datetime.now(datetime.timezone.utc)
                    .isoformat(timespec="seconds")))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "write":
        write_stamp(sys.argv[2], sys.argv[3])
        print("stamp written:", sys.argv[2])
    else:
        require_stamp()
