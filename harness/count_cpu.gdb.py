"""gdb -batch -x count_cpu.gdb.py --args <python> <args...>
Counts calls to mlx::core::cpu::get_command_encoder(Stream) in the
chat/Laya/TTS server subprocess (set follow-fork-mode child), and
prints one JSON line. Redirections for the inferior come from
CPU_COUNT_REDIRECT.
"""
import json
import os
import gdb
SYMBOL = "_ZN3mlx4core3cpu19get_command_encoderENS0_6StreamE"
gdb.execute("set pagination off")
gdb.execute("set breakpoint pending on")
gdb.execute("set confirm off")
gdb.execute("set detach-on-fork off")
gdb.execute("set follow-fork-mode child")
gdb.execute("set print demangle on")
gdb.execute("handle SIGPIPE nostop noprint pass")
report = {"symbol": SYMBOL, "cpu_command_encoder_calls": 0,
          "resolved_while_running": False, "libmlx_loaded": False,
          "child_pid": None}
class Count(gdb.Breakpoint):
    def stop(self):
        report["cpu_command_encoder_calls"] += 1
        return False
bp = Count(SYMBOL)
def on_objfile(event):
    if event.new_objfile.filename.endswith("libmlx.so"):
        report["libmlx_loaded"] = True
    if report["libmlx_loaded"]:
        report["resolved_while_running"] = not bp.pending and len(bp.locations) > 0
gdb.events.new_objfile.connect(on_objfile)
def on_new_inferior(event):
    if event.inferior is not None:
        report["child_pid"] = int(event.inferior.pid)
        # Re-enable the breakpoint for the new inferior (parent bp is
        # shared across all inferiors because detach-on-fork off).
        bp.set_pending(True)
gdb.events.new_inferior.connect(on_new_inferior)
gdb.execute(f"run {gdb.parameter('args') or ''} {os.environ.get('CPU_COUNT_REDIRECT', '')}")
print("CPU_COUNT " + json.dumps(report))