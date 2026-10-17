"""Run a script under one of four uclamp_min placements, to find which threads must carry the clamp.

Usage: run_clamped.py MODE SCRIPT [script args...]
  none    no clamp
  thread  import mlx and mlx_lm, run one op, THEN clamp the calling thread only. The serve shim does this: mlx_lm.server is imported first.
  before  clamp the calling thread first, then everything else starts (threads created afterwards inherit). Same as `uclampset -m 1024 cmd`.
  all     import mlx and mlx_lm, run one op, THEN clamp every existing thread of the process by tid.
The clamp value is 1024 (MLX_OMARCHY_UCLAMP_MIN overrides). A table of thread, name and util_min goes to stderr before SCRIPT runs.
No root. Linux x86_64 or aarch64.
"""
import ctypes
import os
import atexit
import runpy
import struct
import sys

KEEP_POLICY = 0x08
CLAMP_MIN = 0x20
SETATTR, GETATTR = {"x86_64": (314, 315), "aarch64": (274, 275)}[os.uname().machine]
ATTR = struct.Struct("<IIQiiQQQII")
LIBC = ctypes.CDLL(None, use_errno=True)
VALUE = int(os.environ.get("MLX_OMARCHY_UCLAMP_MIN", "1024"))


def set_min(tid):
    buf = ATTR.pack(ATTR.size, 0, KEEP_POLICY | CLAMP_MIN, 0, 0, 0, 0, 0, VALUE, 1024)
    ctypes.set_errno(0)
    return LIBC.syscall(ctypes.c_long(SETATTR), ctypes.c_uint(tid), buf, ctypes.c_uint(0))


def get_min(tid):
    buf = ctypes.create_string_buffer(ATTR.size)
    rc = LIBC.syscall(ctypes.c_long(GETATTR), ctypes.c_uint(tid), buf, ctypes.c_uint(ATTR.size), ctypes.c_uint(0))
    return ATTR.unpack(buf.raw)[8] if rc == 0 else None


def tids():
    return sorted(int(t) for t in os.listdir("/proc/self/task"))


def table(label):
    rows = []
    for t in tids():
        try:
            name = open(f"/proc/self/task/{t}/comm").read().strip()
        except OSError:
            name = "?"
        rows.append(f"{t}:{name}:{get_min(t)}")
    print(f"CLAMP {label} threads {len(rows)} " + " ".join(rows), file=sys.stderr, flush=True)


def start_mlx():
    import mlx.core as mx
    import mlx_lm  # noqa: F401
    import mlx_lm.generate  # noqa: F401  (creates the generation stream and its thread, as mlx_lm.server does)
    import mlx_lm.utils  # noqa: F401

    mx.eval(mx.ones((8, 8)) + 1)


def main():
    mode, script = sys.argv[1], sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    if mode == "before":
        rc = set_min(0)
        print(f"CLAMP before-import set rc={rc}", file=sys.stderr)
    elif mode in ("thread", "all", "none"):
        start_mlx()
        if mode == "thread":
            print(f"CLAMP thread-only set rc={set_min(0)}", file=sys.stderr)
        elif mode == "all":
            rcs = [set_min(t) for t in tids()]
            print(f"CLAMP all-tids set rcs={rcs}", file=sys.stderr)
    else:
        raise SystemExit(f"unknown mode {mode}")
    table(f"mode {mode}")
    atexit.register(table, f"mode {mode} at exit")
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
