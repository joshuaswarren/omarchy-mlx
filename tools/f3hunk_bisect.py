#!/usr/bin/env python3
"""Jw16DecodeFuse3: minimal, byte-exact hunk-apply bisection for the
mlx-lm-rope-norm.patch hunk 1 failure on jw16. Runs ON jw16.
Builds candidate hunks in /tmp/f3h and dry-runs GNU patch on each."""
import subprocess, os

D = "/tmp/f3h"
os.makedirs(D, exist_ok=True)
src = "/var/tmp/fuse3-venv/lib/python3.14/site-packages/mlx_lm/models/qwen3_next.py"
text = open(src, encoding="utf-8").read()
lines = text.split("\n")
print("file line1-5:", lines[:5])
print("line1 repr:", repr(lines[0]))
print("line2 repr:", repr(lines[1]))
print("line3 repr:", repr(lines[2]))

def trial(name, ctx_idx, insert_after, ctx_lines):
    """ctx_lines: list of file line indices (0-based) as context, in order."""
    old0 = min(ctx_lines) + 1
    oldn = len(ctx_lines)
    body = []
    for i in ctx_lines:
        body.append(" " + lines[i] + "\n")
        if i == insert_after:
            body.append("+import os\n")
    p = "".join([f"--- a/{name}.txt\n", f"+++ b/{name}.txt\n",
                 f"@@ -{old0},{oldn} +{old0},{oldn + 1} @@\n"] + body)
    open(f"{D}/{name}.txt", "w", encoding="utf-8").write(
        "\n".join(lines[:6]) + "\n")
    open(f"{D}/{name}.patch", "w", encoding="utf-8").write(p)
    r = subprocess.run(["patch", "--dry-run", "--strip=1", "--forward",
                        "-i", f"{D}/{name}.patch"], cwd=D,
                       capture_output=True, text=True)
    print(f"trial {name}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()}")

# A: ctx = lines 0,2 (skip the empty line 1) insert after 2
trial("a", None, 2, [0, 2])
# B: ctx = lines 2,4 (import + dataclasses), insert after 2
trial("b", None, 2, [2, 4])
# C: ctx = line 2 only
trial("c", None, 2, [2])
# D: ctx = line 4 only, insert after 4
trial("d", None, 4, [4])
