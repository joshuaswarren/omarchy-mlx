"""kernel_names.py [ID ...]: map ComputeKernel ids to names from compute.h at commit e57cb301 (the tree of the wheel the tickets run).
With no ids prints a few check values. Reads the enum with `git show` from ~/src/mlx-omarchy."""
import re
import subprocess
import sys
from pathlib import Path

src = subprocess.run(["git", "-C", str(Path.home() / "src/mlx-omarchy"), "show", "e57cb301:overlay/mlx/backend/omarchy/compute.h"],
                     capture_output=True, text=True, check=True).stdout
body = src.split("enum class ComputeKernel : uint16_t {", 1)[1].split("};", 1)[0]
body = re.sub(r"//[^\n]*", "", body)
body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
names, nxt = {}, 0
for item in body.split(","):
    item = item.strip()
    if not item:
        continue
    m = re.match(r"(\w+)\s*(?:=\s*(\d+))?$", item)
    if not m:
        continue
    if m.group(2):
        nxt = int(m.group(2))
    names[nxt] = m.group(1)
    nxt += 1
ids = [int(x) for x in sys.argv[1:]] or [0, 58, 455, 480, 514, 516]
for i in ids:
    print(i, names.get(i, "?"))
