#!/usr/bin/env python3
"""jw16: byte-verified patch application probe (fixed)."""
import os, hashlib, subprocess

D = "/tmp/f3h4"
os.makedirs(D, exist_ok=True)
ctx = [" # Copyright \xa9 2025 Apple Inc.\n",
       " \n",
       " from __future__ import annotations\n"]
patch_text = "--- a/k.txt\n+++ b/k.txt\n@@ -1,3 +1,4 @@\n" + "".join(ctx) + "+import os\n"
target = "".join(l[1:] for l in ctx) + "\nnext line\n"
p, t = f"{D}/k.patch", f"{D}/k.txt"
open(p, "w", encoding="utf-8").write(patch_text)
open(t, "w", encoding="utf-8").write(target)
print("patch sha:", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
print("target sha:", hashlib.sha256(open(t, "rb").read()).hexdigest()[:16])
print("old-side==target head:",
      open(t, encoding="utf-8").read()[: len(patch_text.splitlines(True)[3].join("")) or 0] is not None)
old_side = "".join(ctx)
tgt = open(t, encoding="utf-8").read()
stripped = "".join(l[1:] for l in old_side.splitlines(True))
print("ctx strip == target head:", tgt.startswith(stripped))
for extra in ([], ["--ignore-whitespace"], ["-l"]):
    r = subprocess.run(["patch", "--dry-run", "--strip=1", "--forward", "--fuzz=0"] + extra + ["-i", "k.patch"],
                       cwd=D, capture_output=True, text=True)
    print((" ".join(extra) or "plain"), "rc:", r.returncode, (r.stdout + r.stderr).strip().replace("\n", " | "))
