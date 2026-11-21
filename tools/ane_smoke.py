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

The node the tool probes is the node the driver is told to open: the driver
opens `/dev/accel/accelN`, so the index is forwarded in `ANE_DEVICE` and the
answer names the same node the result describes. A path the driver cannot be
pointed at is refused, not run.

Answers, one JSON line on stdout:

- node absent: `{"ane": "absent", ...}` and exit 0 - a normal state before
  the node lands, not a failure.
- node present: the driver's smoke runs. `match` exits 0; `mismatch` exits 1
  with the driver's reason; `unavailable` exits 1 with the reason (no
  packaged program for this SoC, the driver is not bound, the smoke is not
  installed, the run failed, or the driver's answer carried no per-call
  evidence).

The `ane` field is the answer; the exit code only says "absent or match".

No hardware is needed to test this file: `--ane-node` points detection at any
path, and the tests inject the smoke runner.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

NODE_CANDIDATES = ("/dev/accel/accel0", "/dev/ane")
SMOKE = "omarchy-ane-smoke"
DT_COMPATIBLE = "sys/firmware/devicetree/base/compatible"
RUN_TIMEOUT_S = 120
NO_JSON = "the driver smoke printed no JSON"
ACCEL_PREFIX = "/dev/accel/accel"


def pick_node(candidates: tuple[str, ...] | None = None) -> str | None:
    """The first ANE device node that exists, else None."""
    if candidates is None:
        candidates = NODE_CANDIDATES
    for node in candidates:
        if Path(node).exists():
            return node
    return None


def device_index(node: str) -> int | None:
    """The device index the driver opens for a node path, else None.

    The driver's libane enumerates /dev/accel/accelN and takes the Nth node
    it opens, so only that path shape can be forwarded to it.
    """
    if not node.startswith(ACCEL_PREFIX):
        return None
    suffix = node[len(ACCEL_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


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


def smoke_json(lines: list[str]) -> dict:
    """The driver's JSON answer: the last line that parses as an object."""
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {"available": False, "reason": NO_JSON}


def run_smoke(timeout: int, env: dict) -> dict:
    """One `omarchy-ane-smoke` call, parsed from its one JSON stdout line."""
    cmd = [shutil.which(SMOKE) or SMOKE, "--timeout", str(timeout)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout + 60,
        check=False,
        env=env,
    )
    payload = smoke_json(proc.stdout.splitlines())
    if payload.get("reason") == NO_JSON:
        err = proc.stderr.strip().splitlines()
        payload["reason"] = (
            err[-1] if err else f"{SMOKE} exited {proc.returncode} with no output"
        )
    return payload


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
    if node is not None and not Path(node).exists():
        return {"ane": "absent", "node": node, "nodes": [node]}
    answer: dict[str, object] = {"ane": "unavailable", "node": node}
    device = device_index(node)
    if device is None:
        answer["error"] = (
            f"the driver opens {ACCEL_PREFIX}N and cannot be told to use {node}"
        )
        return answer
    if smoke_runner is None:
        if shutil.which(SMOKE) is None:
            answer["error"] = f"{SMOKE} is missing. Reinstall omarchy-ane-dkms."
            return answer
        smoke_runner = run_smoke
    env = {**os.environ, "ANE_DEVICE": str(device)}
    try:
        payload = smoke_runner(timeout, env)
    except Exception as err:  # noqa: BLE001 - one line, named, exit 1
        answer["error"] = f"{SMOKE} failed: {type(err).__name__}: {err}"
        return answer
    if not isinstance(payload, dict):
        answer["error"] = "the driver smoke printed no JSON object"
        return answer
    soc = payload.get("chip")
    if soc:
        answer["chip"] = soc
    if not payload.get("available", False):
        answer["error"] = payload.get("reason", "the driver smoke reports no available ANE")
        return answer
    hashes = payload.get("sha256")
    answer["calls"] = len(hashes) if isinstance(hashes, list) else 0
    errors = payload.get("errors")
    if not isinstance(errors, int) or isinstance(errors, bool) or errors < 0:
        answer["error"] = payload.get(
            "reason", "the driver smoke reported no errors count"
        )
        return answer
    answer["errors"] = errors
    if not isinstance(hashes, list) or not hashes:
        answer["error"] = payload.get(
            "reason", "the driver smoke reported no calls"
        )
        return answer
    answer["ane"] = "match" if errors == 0 else "mismatch"
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
        help=(
            "probe this node instead of the known candidates; forwarded to "
            "the driver as ANE_DEVICE when it is an /dev/accel/accelN path"
        ),
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
