"""gdb -batch -x bt_cpu.gdb.py --args <python> <args...>

Like count_cpu.gdb.py, but records the native caller chain of every
mlx::core::cpu::get_command_encoder(Stream) hit and prints a tally of the
distinct chains (one JSON line, CPU_BT)."""
import collections
import json
import os

import gdb

SYMBOL = "_ZN3mlx4core3cpu19get_command_encoderENS0_6StreamE"
gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")
gdb.execute("set confirm off")
gdb.execute("set print demangle on")
gdb.execute("handle SIGPIPE nostop noprint pass")
chains = collections.Counter()


class Hit(gdb.Breakpoint):
    def stop(self):
        names = []
        frame = gdb.newest_frame()
        while frame is not None and len(names) < 14:
            names.append(frame.name() or "??")
            frame = frame.older()
        chains[" <- ".join(names[1:])] += 1
        return False


Hit(SYMBOL)
gdb.execute(f"run {gdb.parameter('args') or ''} {os.environ.get('CPU_COUNT_REDIRECT', '')}")
print("CPU_BT " + json.dumps({"total": sum(chains.values()),
                              "chains": chains.most_common()}))
