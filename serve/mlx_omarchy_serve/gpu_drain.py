"""Finish the queued GPU work before a SIGTERM ends the process (issue 339).

A process killed while one compute job runs and a second waits behind it
makes the kernel driver log a GPU page fault and time out; the machine and
the process recover. The same program that drains the queue first does not
fault. SIGKILL and an out-of-memory kill cannot run a handler and are not
covered.

install() puts the drain on SIGTERM. The handler restores the default action
first, so a second SIGTERM (a supervisor that is tired of waiting) ends the
process at once. After the drain it sends itself SIGTERM, so the exit status
is the one an unhandled SIGTERM gives and a parent sees nothing new.

The handler runs in the main thread between bytecodes, so a process that is
inside a long mx.eval drains when that call returns.
"""

import os
import signal
import sys
import threading


def _drain_gpu():
    import mlx.core as mx

    mx.synchronize()
    # mlx_lm generates on its own stream, not the default one.
    stream = getattr(sys.modules.get("mlx_lm.generate"), "generation_stream", None)
    if stream is not None:
        mx.synchronize(stream)


def install(drain=_drain_gpu):
    """Drain on SIGTERM. Returns False off the main thread (signal limit)."""
    if threading.current_thread() is not threading.main_thread():
        return False

    def handler(signum, _frame):
        signal.signal(signum, signal.SIG_DFL)
        try:
            drain()
        except BaseException as exc:
            print(f"gpu_drain: drain failed: {exc!r}", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, handler)
    return True
