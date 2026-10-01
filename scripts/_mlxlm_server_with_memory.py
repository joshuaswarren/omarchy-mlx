"""Adds GET /v1/internal/memory to the mlx_lm server.

Side-effect import. Patches mlx_lm.server.APIHandler.do_GET (the base
class of _GatedHandler defined inside mlx_omarchy_serve._mlxlm_server)
so that chat workers expose mx.get_active_memory / get_peak_memory /
get_cache_memory on /v1/internal/memory.

Used by PairGates to read backend peak memory from outside the worker
process without modifying the live wheel.
"""
import json

import mlx.core as mx
import mlx_lm.server as _server


_orig_do_get = _server.APIHandler.do_GET


def do_get(self):
    path = self.path.split("?", 1)[0]
    if path == "/v1/internal/memory":
        try:
            payload = {
                "active": int(mx.get_active_memory()),
                "peak": int(mx.get_peak_memory()),
                "cache": int(mx.get_cache_memory()),
            }
        except Exception as e:
            payload = {"error": str(e)}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        return
    return _orig_do_get(self)


_server.APIHandler.do_GET = do_get