#!/usr/bin/env python3
"""Kit-side Neural Engine smoke for a Linux box, for Aurora's M4 rows.

Aurora's device-tree work lands an `apple,ane` node on their machines, so this
tool answers, in one run, whether the real ANE path works on the kit they
already run.

It never touches MLX or `mx.gpu`: MLX has no ANE device, so an `mx.*` result
would claim a Neural Engine pass from a GPU run. The check itself is the
driver's own `omarchy-ane-smoke` (shipped by omarchy-ane-dkms), which picks
the ANEC program for the SoC it finds, runs it on the ANE, and compares every
result bit-for-bit against a CPU model. This tool adds the kit-side node
answer the card needs and nothing else: an absent node is a normal state
before the node lands, not a failure.

Answers, one JSON line on stdout:

- node absent: `{"ane": "absent", ...}` and exit 0 - a normal state before
  the node lands, not a failure.
- node present: the driver's smoke runs. `match` exits 0; `mismatch` exits 1
  with the driver's reason; `unavailable` exits 1 with the reason (no
  packaged program for this SoC, the driver is not bound, the smoke is not
  installed, or the run failed).

No hardware is needed to test this file: `--ane-node` points detection at any
path, and the tests inject the smoke runner.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

NODE_CANDIDATES = ("/dev/accel/accel0", "/dev/ane")
SMOKE = "omarchy-ane-smoke"
DT_COMPATIBLE = "sys/firmware/devicetree/base/compatible"
RUN_TIMEOUT_S = 120


def pick_node(candidates: tuple[str, ...] = NODE_CANDIDATES) -> str | None:
    """The first ANE device node that exists, else None."""
    for node in candidates:
        if Path(node).exists():
            return node
    return None


def chip(root: str = "/") -> str | None:
    """The SoC name from the device tree (apple,t8122 -> t8122), else None."""
    try:
        names = (Path(root) / DT_COMPATIBLE).read_bytes().decode("ascii", "replace").split("\0")
    except OSError:
        return None
    for name in names:
        if name.startswith("apple,t"):
            return name.split(",", 1)[1]
    return None


def run_smoke(timeout: int) -> dict:
    """One `omarchy-ane-smoke` call, parsed from its one JSON stdout line."""
    cmd = [shutil.which(SMOKE) or SMOKE, "--timeout", str(timeout)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60, check=False)
    lines = proc.stdout.strip().splitlines()
    try:
        return json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        err = proc.stderr.strip().splitlines()
        return {"available": False, "reason": err[-1] if err else "the smoke printed no JSON"}


def check(node: str | None = None, smoke_runner=None, timeout: int = RUN_TIMEOUT_S) -> dict:
    """One smoke answer: absent, match, mismatch, or unavailable (+why)."""
    if node is None:
        node = pick_node()
    if node is None:
        answer: dict[str, object] = {"ane": "absent", "nodes": list(NODE_CANDIDATES)}
        soc = chip()
        if soc:
            answer["chip"] = soc
        return answer
    answer: dict[str, object] = {"ane": "unavailable", "node": node}
    if smoke_runner is None:
        if shutil.which(SMOKE) is None:
            answer["error"] = f"{SMOKE} is missing. Reinstall omarchy-ane-dkms."
            return answer
        smoke_runner = run_smoke
    try:
        payload = smoke_runner(timeout)
    except Exception as err:  # noqa: BLE001 - one line, named, exit 1
        answer["error"] = f"{SMOKE} failed: {type(err).__name__}: {err}"
        return answer
    soc = payload.get("chip")
    if soc:
        answer["chip"] = soc
    if not payload.get("available", False):
        answer["error"] = payload.get("reason", "the driver smoke reports no available ANE")
        return answer
    errors = payload.get("errors", 0)
    hashes = payload.get("sha256") or []
    answer["ane"] = "match" if errors == 0 else "mismatch"
    answer["calls"] = len(hashes)
    answer["golden_sha256"] = payload.get("golden_sha256")
    if hashes:
        answer["sha256"] = hashes[0]
    if errors:
        answer["error"] = payload.get("reason", "one or more calls were not bit-exact")
    return answer


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument(
        "--ane-node",
        default=None,
        help="probe this node instead of the known candidates",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=RUN_TIMEOUT_S,
        help=f"seconds for the driver's smoke (default {RUN_TIMEOUT_S})",
    )
    args = parser.parse_args(argv[1:])
    answer = check(node=args.ane_node, timeout=args.timeout)
    json.dump(answer, sys.stdout)
    sys.stdout.write("\n")
    return 0 if answer["ane"] in ("absent", "match") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
