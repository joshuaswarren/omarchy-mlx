"""Unprivileged utilization clamp (uclamp_min) for serving processes.

w71 H204-H209 and PerfHold 2026-10-02 (receipts/2026-10-02-perf-hold):
the first-token cost under the stock schedutil governor is scheduler
placement — the MLX submit thread lands on the E cluster — and
uclamp_min=1024 on the serving process recovers the TTFT gain of the
performance governor without touching any governor (no root, no sysfs).

sched_setattr() with SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_UTIL_CLAMP_MIN
sets the boost hint on ONE thread; every thread that thread starts
afterwards inherits it. MLX evaluates the graph and submits to the GPU on
its own stream threads, and importing mlx and mlx_lm starts them. The
serve therefore clamps its startup thread BEFORE those imports, so the
stream threads inherit the clamp, and then clamps every existing thread
(apply_uclamp_min_all_threads) as a second pass. Raising uclamp_min needs
no capability on these kernels (probed: util-linux uclampset -m 1024
succeeds unprivileged on both Apple Silicon hosts, and sched_setattr on
the other tids of the same process returns 0); any failure (EPERM, old
kernel, seccomp) falls back silently, so serving continues at the stock
clamp.

MLX_OMARCHY_UCLAMP_MIN: unset or 1024 -> clamp to 1024 (on);
0 -> off; any other integer -> that clamp value.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys

UCLAMP_ENV = "MLX_OMARCHY_UCLAMP_MIN"
DEFAULT_UCLAMP_MIN = 1024

SCHED_FLAG_KEEP_POLICY = 0x08
SCHED_FLAG_UTIL_CLAMP_MIN = 0x20

# sched_setattr syscall numbers (kernel uapi) for the machines this
# serves on; unknown architectures refuse (never guess).
_SCHED_SETATTR = {
    "aarch64": 274,
    "x86_64": 314,
}

# struct sched_attr: u32 size, u32 policy, u64 flags, s32 nice,
# u32 priority (padded), u64 runtime, u64 deadline, u64 period,
# u32 util_min, u32 util_max.
_SCHED_ATTR = struct.Struct("<IIQiiQQQII")


def _clamp_from_env(env=None):
    """Resolved clamp value, or None when the switch is off/garbage."""
    env = os.environ if env is None else env
    raw = env.get(UCLAMP_ENV)
    if raw is None or raw == "":
        return DEFAULT_UCLAMP_MIN
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def apply_uclamp_min(value, *, syscall=None, log=None):
    """Set uclamp_min on the calling thread. Returns (ok, detail).

    Never raises: any kernel/libc refusal is a silent fallback (the
    caller logs at debug most). ``syscall`` is injectable for tests:
    callable(nr, pid, bytes) -> int (0 ok, -1 error with errno set by
    the fake via ctypes-style).
    """
    nr = _SCHED_SETATTR.get(sys.platform if sys.platform != "linux" else os.uname().machine)
    if nr is None:
        return False, f"unsupported architecture {os.uname().machine}"
    if syscall is None:
        libc = ctypes.CDLL(None, use_errno=True)

        def syscall(nr_, pid, buf, flags):
            return libc.syscall(ctypes.c_long(nr_), ctypes.c_uint(pid), buf,
                                ctypes.c_uint(flags))

    attr = _SCHED_ATTR.pack(
        _SCHED_ATTR.size, 0, SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_UTIL_CLAMP_MIN,
        0, 0, 0, 0, 0, int(value), 1024)
    ctypes.set_errno(0)
    rc = syscall(nr, 0, attr, 0)
    if rc != 0:
        err = ctypes.get_errno()
        return False, f"sched_setattr failed errno={err}"
    return True, f"uclamp_min={value}"


def apply_uclamp_min_all_threads(value, *, syscall=None, tids=None):
    """Set uclamp_min on EVERY thread of this process. Returns (ok_count, fail_count).

    MLX evaluates the graph and submits to the GPU on its own stream threads,
    which exist as soon as mlx and mlx_lm are imported. A clamp on the calling
    thread alone does not reach them, so decode runs at the stock clamp. This
    walks /proc/self/task and clamps each tid (same user, no capability needed,
    probed on x86_64 and aarch64). ``tids`` and ``syscall`` are injectable for
    tests. Never raises.
    """
    nr = _SCHED_SETATTR.get(os.uname().machine)
    if nr is None:
        return 0, 0
    if syscall is None:
        libc = ctypes.CDLL(None, use_errno=True)

        def syscall(nr_, pid, buf, flags):
            return libc.syscall(ctypes.c_long(nr_), ctypes.c_uint(pid), buf,
                                ctypes.c_uint(flags))
    if tids is None:
        try:
            tids = sorted(int(t) for t in os.listdir("/proc/self/task"))
        except (OSError, ValueError):
            return 0, 0
    attr = _SCHED_ATTR.pack(
        _SCHED_ATTR.size, 0, SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_UTIL_CLAMP_MIN,
        0, 0, 0, 0, 0, int(value), 1024)
    ok = fail = 0
    for tid in tids:
        ctypes.set_errno(0)
        if syscall(nr, int(tid), attr, 0) == 0:
            ok += 1
        else:
            fail += 1
    return ok, fail


def apply_from_env(env=None, log=None, syscall=None, all_threads=False):
    """Startup hook: honor MLX_OMARCHY_UCLAMP_MIN.

    all_threads=False clamps the calling thread (call it before mlx is imported
    and every thread started afterwards inherits it). all_threads=True clamps
    every existing thread of the process (call it after the imports, which
    start the MLX stream threads). Returns (applied, detail) and is safe to
    call unconditionally.
    """
    value = _clamp_from_env(env)
    if value is None:
        return False, f"{UCLAMP_ENV} off or invalid"
    if all_threads:
        ok, fail = apply_uclamp_min_all_threads(value, syscall=syscall)
        applied, detail = ok > 0 and fail == 0, f"uclamp_min={value} on {ok} threads, {fail} refused"
    else:
        applied, detail = apply_uclamp_min(value, syscall=syscall, log=log)
    if log is not None:
        log(detail)
    return applied, detail
