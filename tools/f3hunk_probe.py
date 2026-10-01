#!/usr/bin/env python3
"""In-process verification on jw16: does the hunk old-side byte-match the
file? And does GNU patch apply it at fuzz 0 when written by the same
process? Plus locale probes."""
import subprocess, os, locale

D = "/tmp/f3h2"
os.makedirs(D, exist_ok=True)
src = "/var/tmp/fuse3-venv/lib/python3.14/site-packages/mlx_lm/models/qwen3_next.py"
text = open(src, encoding="utf-8").read()
lines = text.split("\n")

ctx = [0, 1, 2]  # lines 1..3
old_side = "".join(" " + lines[i] + "\n" for i in ctx)
file_head = text[: len(old_side)]
print("old_side == file_head:", old_side == file_head)
print("file_head bytes:", file_head.encode("utf-8"))
print("old_side bytes:", old_side.encode("utf-8"))

name = "k"
body = old_side + "+import os\n"
patch_text = (f"--- a/{name}.txt\n+++ b/{name}.txt\n"
              f"@@ -1,3 +1,4 @@\n{body}")
open(f"{D}/{name}.txt", "w", encoding="utf-8").write(text)
open(f"{D}/{name}.patch", "w", encoding="utf-8").write(patch_text)
r = subprocess.run(["patch", "--dry-run", "--strip=1", "--forward",
                    "-i", f"{name}.patch"], cwd=D, capture_output=True, text=True)
print("fuzz0 rc:", r.returncode, r.stdout.strip(), r.stderr.strip())

env = dict(os.environ, LC_ALL="C.UTF-8")
r = subprocess.run(["patch", "--dry-run", "--strip=1", "--forward", "--fuzz=0",
                    "-i", f"{name}.patch"], cwd=D, capture_output=True,
                   text=True, env=env)
print("C.UTF-8 rc:", r.returncode, r.stdout.strip(), r.stderr.strip())

r = subprocess.run(["patch", "--dry-run", "--strip=1", "--forward", "--ignore-whitespace",
                    "-i", f"{name}.patch"], cwd=D, capture_output=True, text=True)
print("ignore-ws rc:", r.returncode, r.stdout.strip(), r.stderr.strip())
print("locale:", locale.getpreferredencoding(), os.environ.get("LANG"), os.environ.get("LC_ALL"))
