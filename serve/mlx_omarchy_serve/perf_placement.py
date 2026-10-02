"""Unprivileged utilization clamp (uclamp_min) for serving processes.

w71 H204-H209 and PerfHold 2026-10-02 (receipts/2026-10-02-perf-hold):
the first-token cost under the stock schedutil governor is scheduler
placement — the MLX submit thread lands on the E cluster — and
uclamp_min=1024 on the serving process recovers the TTFT gain of the
performance governor without touching any governor (no root, no sysfs).

sched_setattr() with SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_UTIL_CLAMP_MIN
sets the boost hint on the CALLING THREAD only; every thread created
afterwards inherits it, so the serve must call this from its startup
thread before the HTTP server and decode threads spawn. Raising
uclamp_min on the calling thread needs no capability on these kernels
(probed: util-linux uclampset -m 1024 succeeds unprivileged on both
Apple Silicon hosts); any failure (EPERM, old kernel, seccomp) falls
back silently — serving continues at the stock clamp.

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


def apply_from_env(env=None, log=None, syscall=None):
    """Startup hook: honor MLX_OMARCHY_UCLAMP_MIN on the calling thread.

    Returns (applied, detail) and is safe to call unconditionally.
    """
    value = _clamp_from_env(env)
    if value is None:
        return False, f"{UCLAMP_ENV} off or invalid"
    applied, detail = apply_uclamp_min(value, syscall=syscall, log=log)
    if log is not None:
        log(detail)
    return applied, detail
