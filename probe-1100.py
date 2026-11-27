#!/usr/bin/env python3
"""Live probe for PR 1100: qualify bounds + redirect-body drop + partial download."""
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, "serve")
sys.path.insert(0, "tests")
from mlx_omarchy_laya import convert, jsonfile  # noqa: E402

base = Path(tempfile.mkdtemp())
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(name, "OK" if cond else "FAIL")

# 1. jsonfile.read_json_object on a FIFO: refused fast.
fifo = base / "f.json"
os.mkfifo(fifo)
t0 = time.time()
try:
    jsonfile.read_json_object(fifo)
    check("1 fifo json refused", False)
except Exception as e:
    check(f"1 fifo json refused ({(time.time()-t0)*1000:.1f}ms)", True)

# 2. jsonfile cap: oversize regular file refused.
big = base / "big.json"
big.write_text("x" * (jsonfile._JSON_MAX_BYTES + 1))
try:
    jsonfile.read_json_object(big)
    check("2 oversize json refused", False)
except Exception:
    check("2 oversize json refused", True)

# 3. _DropRedirectBody: a redirect whose body never ends must not hang.
infinite_running = True
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/start":
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{PORT}/real")
            self.end_headers()
            # never-ending body on the redirect response
            try:
                while infinite_running:
                    self.wfile.write(b"\x00" * 4096)
                    time.sleep(0.05)
            except Exception:
                pass
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"payload")
    def log_message(self, *a):
        pass

srv = HTTPServer(("127.0.0.1", 0), H)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

dest = base / "dl" / "out.bin"
dest.parent.mkdir(parents=True)
t0 = time.time()
try:
    # short timeout via the url fetch: the drop must close the redirect body
    # before reading it, so the fetch completes fast against /real.
    convert._fetch.__wrapped__ if hasattr(convert._fetch, "__wrapped__") else None
    import urllib.request
    opener = urllib.request.build_opener(convert._DropRedirectBody)
    with opener.open(f"http://127.0.0.1:{PORT}/start", timeout=10) as resp:
        data = resp.read()
    dt = time.time() - t0
    check(f"3 redirect body dropped, followed ({dt:.2f}s, got {data[:7]!r})", data == b"payload" and dt < 5)
except Exception as e:
    check(f"3 redirect: {repr(e)[:60]}", False)
finally:
    infinite_running = False
    srv.shutdown()

print("ALL OK" if ok else "PROBE FAILED")
