"""gdb -batch -x pybt_cpu.gdb.py --args <python> <args...>

At the first hit of each distinct native caller chain of
mlx::core::cpu::get_command_encoder(Stream), print the Python stack of the
dispatching thread (taking the GIL through PyGILState_Ensure), then continue."""
import os

import gdb

SYMBOL = "_ZN3mlx4core3cpu19get_command_encoderENS0_6StreamE"
gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")
gdb.execute("set confirm off")
gdb.execute("handle SIGPIPE nostop noprint pass")
seen = set()
hits = [0]
SNIPPET = ("import sys,traceback,threading;NL=chr(10);"
           "sys.stderr.write('PYSTACK-BEGIN '+threading.current_thread().name+NL);"
           "traceback.print_stack(file=sys.stderr);"
           "sys.stderr.write('PYSTACK-END'+NL);sys.stderr.flush()")


class Hit(gdb.Breakpoint):
    def stop(self):
        hits[0] += 1
        names = []
        frame = gdb.newest_frame().older()
        while frame is not None and len(names) < 3:
            names.append(frame.name() or "??")
            frame = frame.older()
        key = " <- ".join(names)
        if key in seen:
            return False
        seen.add(key)
        print(f"HIT {hits[0]} {key}", flush=True)
        return True


Hit(SYMBOL)


def on_stop(event):
    if not isinstance(event, gdb.BreakpointEvent):
        return
    state = gdb.parse_and_eval("(int)PyGILState_Ensure()")
    gdb.parse_and_eval('(int)PyRun_SimpleString("%s")' % SNIPPET.replace('"', '\\"'))
    gdb.parse_and_eval(f"(void)PyGILState_Release({int(state)})")


gdb.events.stop.connect(on_stop)
gdb.execute(f"run {gdb.parameter('args') or ''} {os.environ.get('CPU_COUNT_REDIRECT', '')}")
while True:
    try:
        inferior = gdb.selected_inferior()
        if inferior.pid == 0:
            break
        gdb.execute("continue")
    except gdb.error:
        break
print(f"TOTAL {hits[0]}")
