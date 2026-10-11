"""H70 probe: why does a 1 GiB score chunk fail on a quiet 16 GB M1 (G13G)?

Steps, one process, each step prints a line:
  INFO     device_info memory keys and /proc/meminfo MemTotal and MemAvailable
  BLOCKS   allocate 1 GiB bf16 blocks one at a time (eval each) until an error; count; free all
  SDPA     non-causal bf16 SDPA at H=8, L=16384, D=128 alone in a fresh heap (the shape of the failing doctest), 3 runs
  AFTER    the BLOCKS count again after the SDPA runs (does the heap come back)
"""
import glob
import hashlib
import importlib.metadata as md
import json
import os

import mlx.core as mx


def meminfo(key):
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return -1


def libs():
    base = os.path.dirname(mx.__file__)
    res = {}
    for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))) + sorted(glob.glob(os.path.join(base, "libmlx*"))):
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                res[os.path.basename(p)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return res


GIB = 1 << 30
print("PROVENANCE mlx-omarchy", md.version("mlx-omarchy"), "libs", json.dumps(libs()), "load1", os.getloadavg()[0], "HK_SYSMEM", os.environ.get("HK_SYSMEM"), flush=True)
info = mx.device_info()
print("INFO", {k: v for k, v in info.items() if "mem" in k.lower() or "work" in k.lower() or k in ("device_name", "driver_info")},
      "MemTotal", meminfo("MemTotal"), "MemAvailable", meminfo("MemAvailable"), flush=True)


def blocks():
    held = []
    err = ""
    try:
        for _ in range(64):
            a = mx.zeros((GIB // 2,), dtype=mx.bfloat16)
            mx.eval(a)
            mx.synchronize()
            held.append(a)
    except Exception as exc:
        err = str(exc)[:200]
    n = len(held)
    del held
    mx.synchronize()
    return n, err


n, err = blocks()
print("BLOCKS", n, "GiB held before error:", err or "none (stopped at 64)", flush=True)

H, L, D = 8, 16384, 128
q = (mx.random.normal((1, H, L, D)) * 0.1).astype(mx.bfloat16)
k = (mx.random.normal((1, H, L, D)) * 0.1).astype(mx.bfloat16)
v = (mx.random.normal((1, H, L, D)) * 0.1).astype(mx.bfloat16)
mx.eval(q, k, v)
for i in range(3):
    try:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)
        mx.eval(out)
        mx.synchronize()
        print("SDPA run", i, "ok absmax", float(mx.abs(out).max()), flush=True)
        del out
    except Exception as exc:
        print("SDPA run", i, "ERROR", str(exc)[:200], flush=True)
        break
n2, err2 = blocks()
print("AFTER", n2, "GiB held before error:", err2 or "none (stopped at 64)", flush=True)
print("MemAvailable end", meminfo("MemAvailable"), flush=True)
