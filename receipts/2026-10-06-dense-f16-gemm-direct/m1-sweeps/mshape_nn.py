"""bf16/f32 linear-layer timing across m (x @ w, w 4096x4096 and 12288x4096),
for the widening break-even. usage: python mshape.py"""
import json
import time

import mlx.core as mx

res = {}
for dt_name, dt in (("bfloat16", mx.bfloat16), ("float32", mx.float32)):
    for n in (4096, 12288):
        w = mx.random.normal((4096, n)).astype(dt)
        for m in (32, 64, 128, 256, 512, 2048):
            x = mx.random.normal((m, 4096)).astype(dt)
            mx.eval(w, x)
            mx.eval(x @ w)
            reps = 0
            t0 = time.time()
            while time.time() - t0 < 2.0:
                mx.eval(x @ w)
                reps += 1
            dt_s = (time.time() - t0) / reps
            res[f"{dt_name}/m{m}/n{n}"] = round(dt_s * 1e3, 3)
print(json.dumps({"mlx": mx.__version__, "ms": res}))
