#!/usr/bin/env python3
"""Kit-side Neural Engine smoke for a Linux box, for Aurora's M4 rows.

Aurora's device-tree work lands an `apple,ane` node on their machines before
our driver is loaded, so this tool answers three questions in one run and is
meant to be run the moment the node appears:

1. Is the ANE device node there (`/dev/ane` by default)? A missing node is a
   clean one-line JSON answer and exit 0 — absent is a normal state before the
   node lands, not a failure.
2. With the node present, run one tiny fixed program (a 4-element integer add,
   PROGRAM below) once on the accelerator and once on the CPU, through the
   kit's `mlx`, and compare the results exactly.
3. A mismatch prints both sides and exits 1; a match exits 0.

Exit codes: 0 = node absent (clean) or program matches; 1 = mismatch or the
program could not run with the node present. Both results are one JSON line on
stdout so a lab row can quote them directly.

No hardware is needed to test this file: `--ane-node` points the detection at
any path, and tests inject `run_program`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ANE_NODE = "/dev/ane"

# One tiny known program: exact integer arithmetic, no floating-point drift,
# small enough that any backend can run it.
PROGRAM = {
    "op": "add",
    "a": [1, 2, 3, 4],
    "b": [5, 6, 7, 8],
    "expected": [6, 8, 10, 12],
}


def run_program(program: dict, device: str) -> list[int]:
    """Run `program` on `device` through the kit's mlx and return flat ints.

    `device` is "accelerator" (the public mx.gpu device, which is the ANE when
    the Omarchy backend loaded it) or "cpu" (mx.cpu). Raises on any backend
    error; the caller turns that into a failure line.
    """
    import mlx.core as mx

    a = mx.array(program["a"], dtype=mx.int32)
    b = mx.array(program["b"], dtype=mx.int32)
    with mx.stream(mx.cpu if device == "cpu" else mx.default_device()):
        out = mx.add(a, b)
        mx.eval(out)
    values = out.tolist()
    return [int(x) for x in values]


def check(node: str = ANE_NODE, runner=None) -> dict:
    """One smoke answer: absent, match, or mismatch (+why)."""
    if runner is None:
        runner = run_program  # resolved at call time so tests can inject it
    if not Path(node).exists():
        return {"ane": "absent", "node": node}
    program = dict(PROGRAM)
    try:
        accelerator_result = runner(program, "accelerator")
    except Exception as err:  # noqa: BLE001 — one line, named, exit 1
        return {
            "ane": "mismatch",
            "node": node,
            "program": program,
            "error": f"accelerator run failed: {type(err).__name__}: {err}",
        }
    cpu_result = runner(program, "cpu")
    result = {
        "ane": "match" if accelerator_result == cpu_result else "mismatch",
        "node": node,
        "program": program,
        "expected": program["expected"],
        "accelerator_result": accelerator_result,
        "cpu_result": cpu_result,
    }
    return result


def main(argv: list[str], node: str = ANE_NODE) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ane-node",
        default=node,
        help=f"ANE device node to probe (default {node})",
    )
    args = parser.parse_args(argv[1:])
    answer = check(args.ane_node)
    json.dump(answer, sys.stdout)
    sys.stdout.write("\n")
    return 0 if answer["ane"] in ("absent", "match") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
