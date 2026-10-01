#!/usr/bin/env python3
"""jw16 filesystem readback integrity probe (btrfs suspicion)."""
import os, hashlib, subprocess

D = "/tmp/f3h3"
os.makedirs(D, exist_ok=True)
payload = ("--- a/k.txt\n+++ b/k.txt\n"
           "@@ -1,3 +1,4 @@\n"
           " # Copyright \xa9 2025 Apple Inc.\n"
           " \n"
           " from __future__ import annotations\n"
           "+import os\n")
p = f"{D}/k.patch"
open(p, "w", encoding="utf-8").write(payload)
h1 = hashlib.sha256(open(p, "rb").read()).hexdigest()
h2 = hashlib.sha256(open(p, "rb").read()).hexdigest()
subprocess.run(["sync"])
h3 = hashlib.sha256(open(p, "rb").read()).hexdigest()
h4 = subprocess.run(["sha256sum", p], capture_output=True, text=True).stdout.split()[0]
print("hash1", h1)
print("hash2", h2)
print("hash3", h3)
print("hash4", h4)
print("readback equal:", h1 == h2 == h3 == h4)
data = open(p, "rb").read()
print("repr head:", repr(data[:60]))
r = subprocess.run(["patch", "--dry-run", "--strip=1", "--forward", "--fuzz=0",
                    "-i", "k.patch"], cwd=D, capture_output=True, text=True)
print("patch rc:", r.returncode, r.stdout.strip(), r.stderr.strip())
