#!/usr/bin/env python3
"""Run one qwen38 bench cell in-process and capture backend trace counters.

The omarchy trace counters are process-local, so the bench must run inside
this process (runpy) for the before/after delta to mean anything. Merges the
delta into the cell JSON the bench writes. Captures warmup+load too; the
census subtracts the constant via the depth slope, like the prior lanes'
NDJSON method.
"""
import json
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bsched_counters import snapshot  # noqa: E402


def main():
    out = sys.argv[1]
    bench = sys.argv[2]
    argv = sys.argv[3:]
    before = snapshot()
    sys.argv = [bench] + argv
    try:
        runpy.run_path(bench, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            raise
    after = snapshot()
    delta = {k: after[k] - before[k] for k in after}
    cell = json.load(open(out))
    cell["trace_delta"] = delta
    cell["wave"] = int(os.environ.get("MLX_OMARCHY_WAVE_SCHED", "0"))
    with open(out, "w") as f:
        json.dump(cell, f, indent=1)
    print(
        f"[census] {os.path.basename(out)} wave={cell['wave']} "
        f"dispatches={delta['vk_compute_dispatches']} "
        f"barriers_emitted={delta['barriers_emitted']} "
        f"skipped={delta['barriers_skipped']} "
        f"copies={delta['vk_buffer_copies']} fills={delta['vk_buffer_fills']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
