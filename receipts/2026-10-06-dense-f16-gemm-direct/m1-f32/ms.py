"""Linear-layer timing across m and n. usage: python ms.py <dtype> <nt|nn>
nt: x @ w.T with w (n, 4096); nn: x @ w with w (4096, n)."""
import json
import sys
import time

import mlx.core as mx

dt = getattr(mx, sys.argv[1])
orient = sys.argv[2]
res = {}
for n in (4096, 12288):
    w = mx.random.normal((n, 4096) if orient == "nt" else (4096, n)).astype(dt)
    for m in (32, 64, 128, 256, 512, 2048):
        x = mx.random.normal((m, 4096)).astype(dt)
        mx.eval(w, x)
        f = (lambda: x @ w.T) if orient == "nt" else (lambda: x @ w)
        mx.eval(f())
        reps = 0
        t0 = time.time()
        while time.time() - t0 < 2.0:
            mx.eval(f())
            reps += 1
        res[f"{sys.argv[1]}/{orient}/m{m}/n{n}"] = round((time.time() - t0) / reps * 1e3, 3)
print(json.dumps({"mlx": mx.__version__, "ms": res}))
