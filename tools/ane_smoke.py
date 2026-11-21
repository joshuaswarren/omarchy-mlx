#!/usr/bin/env python3
"""Kit-side Neural Engine smoke for a Linux box, for Aurora's M4 rows.

Aurora's device-tree work lands an `apple,ane` node on their machines, so this
tool answers, in one run, whether the real ANE path works on the kit they
already run. It never touches MLX or mx.gpu: MLX has no ANE device, so an
mx.* result would claim a Neural Engine pass from a GPU run. The program is
the omarchy-ane driver's own pinned H13 add fixture, run through the
`omarchy-ane-run` CLI on the driver node jw16 exposes (`/dev/accel/accel0`;
`/dev/ane` kept as a legacy candidate), compared bit-exact against the same
fp16 add computed on the CPU with numpy.

The fixture inputs here are chosen exact: every a+b is an exact fp16 value,
so hardware and CPU must agree regardless of rounding mode.

Answers, one JSON line on stdout:

- node absent: `{"ane": "absent", ...}` and exit 0 — a normal state before
  the node lands, not a failure.
- node present: the program runs once on the ANE and once on the CPU.
  `match` exits 0; `mismatch` exits 1 and names the first differing index;
  a missing runner or fixture, or a failed run, is `unavailable` and exits 1
  with the reason.

No hardware is needed to test this file: `--ane-node` points detection at any
path and tests inject the ANE runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

NODE_CANDIDATES = ("/dev/accel/accel0", "/dev/ane")
RUNNER = "omarchy-ane-run"
# The pinned H13 add fixture shipped by omarchy-ane-dkms.
H13_ANEC = "/usr/share/omarchy-ane/fixtures/h13-anec/add/program-0.anec"
H13_LANES, H13_PLANE = 64, 64  # 64 lanes x 64 fp16 values: the 16 KiB surface
SURFACE_BYTES = H13_LANES * H13_PLANE * 2
RUN_TIMEOUT_S = 60


def pick_node(candidates: tuple[str, ...] = NODE_CANDIDATES) -> str | None:
    """The first ANE device node that exists, else None."""
    for node in candidates:
        if Path(node).exists():
            return node
    return None


def fixture_inputs() -> tuple[list[float], list[float]]:
    """Two exact fp16 addends per lane position: half-integers, sums exact."""
    n = H13_LANES * H13_PLANE
    a = [1.0 + (i % 8) for i in range(n)]
    b = [0.5 + (i % 4) * 0.25 for i in range(n)]
    return a, b


def surface(values: list[float]) -> bytes:
    """The H13 surface layout: fp16 little-endian values, 64 per lane."""
    return b"".join(struct.pack("<e", v) for v in values)


def cpu_add(a: bytes, b: bytes) -> bytes:
    """The same add on CPU, bit-exact: fp16 in, fp16 out."""
    import numpy as np

    va = np.frombuffer(a, dtype="<f2")
    vb = np.frombuffer(b, dtype="<f2")
    return (va + vb).astype("<f2").tobytes()


def ane_add(runner: str, anec: str, a: bytes, b: bytes) -> bytes:
    """One omarchy-ane-run call: H13 add, inputs 0 and 1, output 0."""
    with tempfile.TemporaryDirectory(prefix="ane-smoke.") as work:
        path = Path(work)
        fa, fb, fy = path / "a.fp16", path / "b.fp16", path / "y.fp16"
        fa.write_bytes(a)
        fb.write_bytes(b)
        cmd = [
            runner,
            "--anec", anec,
            "--in", f"0={fa}",
            "--in", f"1={fb}",
            "--out", f"0={fy}",
        ]
        subprocess.run(cmd, check=True, timeout=RUN_TIMEOUT_S)
        return fy.read_bytes()


def check(node: str | None = None, anec: str = H13_ANEC, ane_runner=None) -> dict:
    """One smoke answer: absent, match, mismatch, or unavailable (+why)."""
    if node is None:
        node = pick_node()
    if node is None:
        return {"ane": "absent", "nodes": list(NODE_CANDIDATES)}
    runner_path = shutil.which(RUNNER)
    if ane_runner is None:
        if runner_path is None:
            return {
                "ane": "unavailable",
                "node": node,
                "error": f"{RUNNER} is not on PATH. Reinstall omarchy-ane-dkms.",
            }
        if not Path(anec).is_file():
            return {
                "ane": "unavailable",
                "node": node,
                "error": f"pinned fixture missing: {anec}. Reinstall omarchy-ane-dkms.",
            }
        ane_runner = ane_add
    else:
        runner_path = runner_path or RUNNER
    a, b = fixture_inputs()
    ab, bb = surface(a), surface(b)
    try:
        y = ane_runner(runner_path, anec, ab, bb)
    except Exception as err:  # noqa: BLE001 — one line, named, exit 1
        return {
            "ane": "unavailable",
            "node": node,
            "error": f"ANE run failed: {type(err).__name__}: {err}",
        }
    cpu = cpu_add(ab, bb)
    answer = {
        "ane": "match" if y == cpu else "mismatch",
        "node": node,
        "anec": anec,
        "ane_sha256": hashlib.sha256(y).hexdigest(),
        "cpu_sha256": hashlib.sha256(cpu).hexdigest(),
        "surface_bytes": SURFACE_BYTES,
    }
    if y != cpu:
        va = struct.unpack(f"<{H13_LANES * H13_PLANE}e", y)
        vc = struct.unpack(f"<{H13_LANES * H13_PLANE}e", cpu)
        first = next((i for i, (x, z) in enumerate(zip(va, vc)) if x != z), None)
        answer["first_mismatch_index"] = first
        if first is not None:
            answer["ane_value"] = va[first]
            answer["cpu_value"] = vc[first]
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
        "--anec",
        default=H13_ANEC,
        help=f"pinned ANEC add program (default {H13_ANEC})",
    )
    args = parser.parse_args(argv[1:])
    answer = check(node=args.ane_node, anec=args.anec)
    json.dump(answer, sys.stdout)
    sys.stdout.write("\n")
    return 0 if answer["ane"] in ("absent", "match") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
