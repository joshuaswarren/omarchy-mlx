"""Time-resolved fp16 4096x4096 matmul loop (the ledger's GPU matmul cell) with per-second TFLOPS.
usage: mm_trace.py <seconds> [uclamp=0|1] ; prints one JSON line {tflops_total, per_sec:[...], eval_ms:[first..], n}"""
import json
import os
import sys
import time

secs = float(sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2] == "1":
    sys.path.insert(0, "/var/tmp/h276/src2/serve")
    from mlx_omarchy_serve.perf_placement import apply_from_env
    os.environ.setdefault("MLX_OMARCHY_UCLAMP_MIN", "1024")
    apply_from_env()
import mlx.core as mx

a = mx.random.normal((4096, 4096)).astype(mx.float16)
b = mx.random.normal((4096, 4096)).astype(mx.float16)
mx.eval(a, b)
FL = 2 * 4096**3
t0 = time.time()
ts = []
while time.time() - t0 < secs:
    mx.eval(a @ b)
    ts.append(time.time() - t0)
per_sec = []
for s in range(int(secs)):
    n = sum(1 for t in ts if s <= t < s + 1)
    per_sec.append(round(n * FL / 1e12, 3))
ev = [round((ts[i] - (ts[i - 1] if i else 0)) * 1000, 1) for i in range(len(ts))]
print(json.dumps({"tflops_total": round(len(ts) * FL / secs / 1e12, 3), "per_sec": per_sec, "n": len(ts), "eval_ms_head": ev[:6], "eval_ms_tail": ev[-6:]}))
