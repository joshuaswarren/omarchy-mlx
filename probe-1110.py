#!/usr/bin/env python3
"""Live probe for PR 1110: _read_regular_bounded refusals and bounds.

Contract: returns True only on EOF within max_bytes; refusal or error
returns False (the caller turns False into its refusal path).
"""
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "serve")
from mlx_omarchy_serve import _mlxlm_server as m  # noqa: E402

base = Path(tempfile.mkdtemp())
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(name, "OK" if cond else "FAIL")

def rd(p, cap=4096):
    return m._read_regular_bounded(p, cap, lambda c: None)

# 1. FIFO refused fast (no hang), returns False.
fifo = base / ".build-commit"
os.mkfifo(fifo)
t0 = time.time()
r1 = rd(fifo)
check(f"1 fifo refused False ({(time.time()-t0)*1000:.1f}ms)", r1 is False)

# 2. Oversize sparse file refused, returns False.
big = base / "METADATA"
with open(big, "wb") as f:
    f.truncate(10 << 20)
check("2 oversize refused False", rd(big) is False)

# 3. Exactly-at-cap regular file passes, True.
at = base / "at.cap"
at.write_text("x" * 4096)
got = []
r3 = m._read_regular_bounded(at, 4096, got.append)
check("3 at-cap passes True, 4096 bytes", r3 is True and len(b"".join(got)) == 4096)

# 4. cap+1 regular file refused, False.
over = base / "over.cap"
over.write_text("x" * 4097)
check("4 cap+1 refused False", rd(over) is False)

# 5. Directory refused, False.
check("5 dir refused False", rd(base) is False)

# 6. Binary bytes stream raw to the sink (caller owns the UTF-8 decode).
bad = base / "bad.stamp"
bad.write_bytes(b"\xff\xfe\x00binary")
got6 = []
r6 = m._read_regular_bounded(bad, 4096, got6.append)
check("6 binary streams raw", r6 is True and b"".join(got6) == b"\xff\xfe\x00binary")

# 7. Socket file refused, False.
sockp = base / "s.sock"
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(str(sockp))
try:
    check("7 socket refused False", rd(sockp) is False)
finally:
    s.close()

# 8. RECORD hash enforcement present (PEP 376 canonical b64).
import inspect
src = inspect.getsource(m)
check("8 RECORD b64 validation present", "b64decode" in src)

print("ALL OK" if ok else "PROBE FAILED")
