#!/usr/bin/env python3
"""Writes the analysis outputs of the G13C gate into OUT: for each shape the tfcmp, tfulp, tfpair and tfseg outputs, the position-bucket table of the long shape, and the chip comparison.
usage: analysis.py OUT   (run from the scratch directory that holds the captures)"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
py = str(Path("venv/bin/python"))
shapes = {
    "short": ("cpu-ref-lse", {"tile1": "gate-<host>/tile1-short", "wide": "gate-<host>/wide-short", "plain": "gate-<host>/plain-short"}),
    "long": ("cpu-ref-long", {"tile1": "gate-<host>/tile1-long", "wide": "gate-<host>/wide-long", "plain": "gate-<host>/plain-long"}),
    "ctx2048": ("cpu-ctx-2048", {"tile1": "ctxgate-<host>/tile1-2048", "wide": "ctxgate-<host>/wide-2048", "plain": "ctxgate-<host>/plain-2048"}),
    "ctx4096": ("cpu-ctx-4096", {"tile1": "ctxgate-<host>/tile1-4096", "wide": "ctxgate-<host>/wide-4096", "plain": "ctxgate-<host>/plain-4096"}),
}


def run(cmd, name):
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out / name).write_text(result.stdout + result.stderr)


for shape, (ref, arms) in shapes.items():
    if not Path(ref).exists():
        print("skip", shape, "no reference")
        continue
    specs = [f"{k}={v}" for k, v in arms.items()]
    run([py, "tfcmp.py", ref, *specs], f"{shape}-tfcmp.out")
    run([py, "tfulp.py", ref, specs[0], specs[1]], f"{shape}-tfulp.out")
    run([py, "tfpair.py", ref, specs[1], specs[0]], f"{shape}-tfpair.out")
    run([py, "tfseg.py", ref, specs[1], specs[0]], f"{shape}-tfseg.out")

lines = []
ref = [np.load(f"cpu-ref-long/seg{s}.npz") for s in (0, 1)]
for lo, hi in ((0, 160), (160, 600)):
    for arm, path in (("tile1", "gate-<host>/tile1-long"), ("wide", "gate-<host>/wide-long")):
        x = [np.load(f"{path}/seg{s}.npz") for s in (0, 1)]
        n = flips = 0
        err = []
        for r, a in zip(ref, x):
            m = min(len(r["target_lp"]), len(a["target_lp"]))
            sel = (np.arange(m) >= lo) & (np.arange(m) < hi)
            flips += int((r["top_i"][:m, 0][sel] != a["top_i"][:m, 0][sel]).sum())
            n += int(sel.sum())
            err.append(np.abs(r["target_lp"][:m][sel] - a["target_lp"][:m][sel]))
        lines.append(f"positions {lo}-{hi - 1} {arm}: top-1 agreement {n - flips}/{n} ({100 * (n - flips) / n:.2f}%), mean abs log-prob error {np.concatenate(err).mean():.5f}")
(out / "long-position-buckets.out").write_text("\n".join(lines) + "\n")

pairs = [
    ("long segments, tiled arm", "lgate/tile1", "gate-<host>/tile1-long", 2),
    ("long segments, new arm", "lgate/new", "gate-<host>/wide-long", 2),
    ("ctx 2048, tiled arm", "ctxgate-m2/tile1-2048", "ctxgate-<host>/tile1-2048", 1),
    ("ctx 2048, new arm", "ctxgate-m2/new-2048", "ctxgate-<host>/wide-2048", 1),
    ("ctx 4096, tiled arm", "ctxgate-m2/tile1-4096", "ctxgate-<host>/tile1-4096", 1),
    ("ctx 4096, new arm", "ctxgate-m2/new-4096", "ctxgate-<host>/wide-4096", 1),
]
chip = ["Comparison of the captures of the two chips: np.array_equal on every array (ids, target_lp, top_i, top_v) of every segment, for the same stack on both Macs."]
for label, m2, jw, nseg in pairs:
    for s in range(nseg):
        a, b = np.load(f"{m2}/seg{s}.npz"), np.load(f"{jw}/seg{s}.npz")
        keys = ["ids", "target_lp", "top_i", "top_v"]
        equal = all(np.array_equal(a[k], b[k]) for k in keys)
        chip.append(f"{label} segment {s}: {'all four arrays equal' if equal else 'DIFFERENT'} ({len(a['target_lp'])} steps)")
chip.append("")
chip.append("Within one machine: the wide arm against the plain arm, and the tiled arm against its repeat, same comparison.")
within = [
    ("14-inch M2 Max long, new vs plain", "lgate/new", "lgate/plain", 2),
    ("14-inch M2 Max long, tile1 vs tile1b", "lgate/tile1", "lgate/tile1b", 2),
    ("14-inch M2 Max ctx 2048, new vs plain", "ctxgate-m2/new-2048", "ctxgate-m2/plain-2048", 1),
    ("14-inch M2 Max ctx 4096, new vs plain", "ctxgate-m2/new-4096", "ctxgate-m2/plain-4096", 1),
    ("16-inch M1 Max long, wide vs plain", "gate-<host>/wide-long", "gate-<host>/plain-long", 2),
    ("16-inch M1 Max short, wide vs plain", "gate-<host>/wide-short", "gate-<host>/plain-short", 4),
    ("16-inch M1 Max ctx 2048, wide vs plain", "ctxgate-<host>/wide-2048", "ctxgate-<host>/plain-2048", 1),
    ("16-inch M1 Max ctx 4096, wide vs plain", "ctxgate-<host>/wide-4096", "ctxgate-<host>/plain-4096", 1),
    ("16-inch M1 Max ctx 2048, tile1 vs tile1b", "ctxgate-<host>/tile1-2048", "ctxgate-<host>/tile1b-2048", 1),
    ("16-inch M1 Max ctx 4096, tile1 vs tile1b", "ctxgate-<host>/tile1-4096", "ctxgate-<host>/tile1b-4096", 1),
]
for label, x, y, nseg in within:
    for s in range(nseg):
        a, b = np.load(f"{x}/seg{s}.npz"), np.load(f"{y}/seg{s}.npz")
        equal = all(np.array_equal(a[k], b[k]) for k in ["ids", "target_lp", "top_i", "top_v"])
        chip.append(f"{label} segment {s}: {'all four arrays equal' if equal else 'DIFFERENT'} ({len(a['target_lp'])} steps)")
(out / "chip-comparison.out").write_text("\n".join(chip) + "\n")
print("done", math.floor(len(list(out.iterdir()))), "files")
