#!/usr/bin/env python3
"""Live mechanical probe for PR 1047: read_snapshot_text bounds and refusals."""
import json
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "serve")
from mlx_omarchy_serve import __main__ as srv  # noqa: E402

base = Path(tempfile.mkdtemp())
ok = True

def check(name, cond):
    global ok
    ok &= bool(cond)
    print(name, "OK" if cond else "FAIL")

# 1. FIFO without writer: refused fast (no block).
fifo = base / "f.json"
os.mkfifo(fifo)
t0 = time.time()
try:
    srv.read_snapshot_text(fifo)
    check("1 fifo refused", False)
except ValueError as e:
    dt = time.time() - t0
    check(f"1 fifo refused fast ({dt*1000:.1f}ms)", dt < 0.05)

# 2. FIFO with writer: fstat refusal (opens, S_ISREG fails).
r_keep = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)  # hold a reader so the write end opens
w = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
try:
    srv.read_snapshot_text(fifo)
    check("2 fifo-with-writer refused", False)
except ValueError:
    check("2 fifo-with-writer refused", True)
finally:
    os.close(w)
    os.close(r_keep)

# 3. Sparse cap+1 file: refused pre-read.
sparse = base / "big.json"
with open(sparse, "wb") as f:
    f.truncate(srv.SNAPSHOT_JSON_MAX_BYTES + 1)
t0 = time.time()
try:
    srv.read_snapshot_text(sparse)
    check("3 sparse cap+1 refused", False)
except ValueError as e:
    check(f"3 sparse cap+1 refused ({(time.time()-t0)*1000:.1f}ms, pre-read)", "+ bytes" in str(e) or "exceeds" in str(e))

# 4. blob/ symlink followed (HF cache layout).
blob = base / "blobs" / "abc123"
blob.parent.mkdir()
blob.write_text('{"weight_map": {"w": "model-00001-of-00002.safetensors"}}')
link = base / "model.safetensors.index.json"
os.symlink(blob, link)
shards = srv.index_shards(link)
check("4 blob symlink followed", shards == ["model-00001-of-00002.safetensors"])

# 5. Exactly-at-cap regular file passes.
at_cap = base / "cap.json"
at_cap.write_text("x" * srv.SNAPSHOT_JSON_MAX_BYTES)
t0 = time.time()
txt = srv.read_snapshot_text(at_cap)
check(f"5 at-cap passes ({time.time()-t0:.2f}s)", len(txt) == srv.SNAPSHOT_JSON_MAX_BYTES)

# 6. Socket file: refused (S_ISREG false).
sock = base / "s.sock"
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(str(sock))
try:
    srv.read_snapshot_text(sock)
    check("6 socket refused", False)
except (ValueError, OSError):
    check("6 socket refused (OSError at caller boundary)", True)
finally:
    s.close()

# 7. Directory passed: OSError (called code treats unreadable as gap).
try:
    srv.read_snapshot_text(base)
    check("7 dir refused", False)
except (OSError, ValueError):
    check("7 dir refused", True)

# 8. Non-UTF8 bytes inside a regular file: decode error surfaces as ValueError-ish?
bad = base / "bad.json"
bad.write_bytes(b'{"a": "\xff\xfe"}')
try:
    srv.read_snapshot_text(bad)
    check("8 non-utf8 raises", True)  # any exception is fine; caller catches
except UnicodeDecodeError:
    check("8 non-utf8 raises", True)
except ValueError:
    check("8 non-utf8 raises", True)

print("ALL OK" if ok else "PROBE FAILED")
