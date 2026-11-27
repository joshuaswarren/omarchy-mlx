#!/usr/bin/env python3
"""Live probe for PR 1044 core machinery with real files, no stubs."""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, "serve")
from mlx_omarchy_laya import convert  # noqa: E402

base = Path(tempfile.mkdtemp())
ok = True

# 1. _replace_file: failure inside the block leaves dst untouched, no litter.
dst = base / "weights.bin"
dst.write_bytes(b"OLD-WEIGHTS")
try:
    with convert._replace_file(dst) as f:
        f.write(b"NEW-PARTIAL")
        raise RuntimeError("simulated crash mid-write")
except RuntimeError:
    pass
after = dst.read_bytes()
litter = list(base.glob("*.part"))
ok &= after == b"OLD-WEIGHTS" and not litter
print("1. crash-mid-write: dst intact =", after == b"OLD-WEIGHTS", ", litter =", litter or "none")

# 2. _replace_file success path really replaces.
with convert._replace_file(dst) as f:
    f.write(b"NEW-WEIGHTS")
ok &= dst.read_bytes() == b"NEW-WEIGHTS"
print("2. success replace:", dst.read_bytes() == b"NEW-WEIGHTS")

# 3. _lock_out: second holder gets the named refusal.
src = base / "src.bin"; src.write_bytes(b"x" * 100)
with convert._lock_out(base):
    other = []
    def contender():
        try:
            with convert._lock_out(base):
                other.append("acquired")
        except SystemExit as e:
            other.append(str(e))
    t = threading.Thread(target=contender); t.start(); t.join()
    good = other and other[0].startswith("convert: another convert is writing")
    ok &= bool(good)
    print("3. concurrent lock:", other[0][:60])

# 4. _copy_source: stable source copies; mutated source aborts with the named message.
out4 = base / "copy.bin"
h = convert._sha256(src, "src.bin")
convert._copy_source(src, out4, "src.bin", h)
ok &= out4.read_bytes() == b"x" * 100
src.write_bytes(b"y" * 100)  # source mutated after snapshot
try:
    convert._copy_source(src, out4, "src.bin", h)
    print("4. mutated source: FAIL no error")
    ok = False
except SystemExit as e:
    good = "changed in the source snapshot" in str(e)
    ok &= good
    print("4. mutated source refused:", str(e)[:60])

print("ALL OK" if ok else "PROBE FAILED")
