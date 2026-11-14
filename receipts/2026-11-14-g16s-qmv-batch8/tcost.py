# forward cost of a cached T-token pass (T=1..8) after a 400-token context; full logits
import os, sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as C
repo = sys.argv[1]
m, tok = load(repo)
ids = tok.encode("The river Danube flows through Bratislava and Vienna. " * 40)[:400]
c = C.make_prompt_cache(m); mx.eval(m(mx.array(ids, mx.uint32)[None], cache=c)); mx.eval([x.state for x in c])
trim = C.can_trim_prompt_cache(c)
os.environ["MLX_OMARCHY_FULL_LOGITS"] = "1"
print("device", mx.device_info().get("device_name"), "SMALL_T", os.environ.get("MLX_OMARCHY_QMV_SMALL_T"), flush=True)
for T in range(1, 9):
    x = mx.array(ids[:T], mx.uint32)[None]
    ts = []
    for r in range(12):
        snap = None if trim else [None if q.is_trimmable() else list(q.state) for q in c]
        t0 = time.perf_counter(); y = m(x, cache=c); mx.eval(y); ts.append(time.perf_counter() - t0)
        if trim: C.trim_prompt_cache(c, T)
        else:
            C.trim_prompt_cache([q for q in c if q.is_trimmable()], T)
            for q, s in zip(c, snap):
                if s is not None: q.state = s
    ts = sorted(ts[2:]); print(f"T={T} med {ts[len(ts)//2]*1000:.2f} ms min {ts[0]*1000:.2f}", flush=True)
