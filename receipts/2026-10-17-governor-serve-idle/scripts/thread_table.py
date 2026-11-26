#!/usr/bin/env python3
"""Print the util_min of every thread of a process: thread_table.py PID  (sched_getattr on each tid, same user, read-only)."""
import ctypes
import os
import struct
import sys

GETATTR = {"x86_64": 315, "aarch64": 275}[os.uname().machine]
ATTR = struct.Struct("<IIQiiQQQII")
libc = ctypes.CDLL(None, use_errno=True)
pid = int(sys.argv[1])
rows = []
for tid in sorted(int(t) for t in os.listdir(f"/proc/{pid}/task")):
    buf = ctypes.create_string_buffer(ATTR.size)
    rc = libc.syscall(ctypes.c_long(GETATTR), ctypes.c_uint(tid), buf, ctypes.c_uint(ATTR.size), ctypes.c_uint(0))
    rows.append((tid, ATTR.unpack(buf.raw)[8] if rc == 0 else -1))
clamped = sum(1 for _, v in rows if v >= 1024)
print(f"THREADS pid {pid} total {len(rows)} util_min>=1024 {clamped} util_min=0 {sum(1 for _, v in rows if v == 0)} unreadable {sum(1 for _, v in rows if v < 0)} "
      + " ".join(f"{t}:{v}" for t, v in rows[:40]))
