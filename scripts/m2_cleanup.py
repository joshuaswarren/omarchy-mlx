#!/usr/bin/env python3
"""HwProbe M2 disk cleanup (allowlist script, per the orchestrator's disk rule).

Deletes exactly the two paths this lane created on the M2 and nothing else:
  /var/tmp/hwprobe-wheel   (private copy of the od prepared tree + built wheel)
  /var/tmp/hwprobe-venv    (private venv copy with the HwProbe wheel installed)

Refuses to run if either path is not exactly one of the allowlisted literals.
"""
import shutil
import sys

ALLOWED = {
    "/var/tmp/hwprobe-wheel",
    "/var/tmp/hwprobe-venv",
}

freed = []
for path in sorted(ALLOWED):
    try:
        total = shutil.disk_usage("/var/tmp").free
    except OSError:
        total = -1
    shutil.rmtree(path, ignore_errors=False)
    freed.append(path)
    print(f"removed {path} (free bytes before: {total})")

# Verify gone.
import os
for path in freed:
    if os.path.exists(path):
        print(f"ERROR: {path} still exists", file=sys.stderr)
        sys.exit(1)
print("cleanup complete")
