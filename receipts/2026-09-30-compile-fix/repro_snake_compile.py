"""Repro A: mx.compile of the Snake activation (reported as a hang)."""
import sys, time
sys.path.insert(0, sys.argv[1])  # <repo>/scripts, for mlx_provenance
from mlx_provenance import installed_provenance, provenance_line
import mlx.core as mx
print(provenance_line(installed_provenance()), flush=True)
snake = lambda x, a: x + (1 / a) * (mx.sin(a * x) ** 2)
cases = [(dt, s, "uniform") for dt in (mx.float32, mx.bfloat16)
         for s in ((1, 8, 64), (1, 512, 4096))] + [(mx.float32, (1, 8, 256), "ones")]
for dt, shape, alpha in cases:  # lazy inputs, as reported; "ones" = the reported line
    x = mx.random.normal(shape).astype(dt)
    a = (mx.ones((1, shape[1], 1)) if alpha == "ones"
         else mx.random.uniform(0.5, 2.0, (1, shape[1], 1))).astype(dt)
    t0 = time.perf_counter(); y = mx.compile(snake)(x, a); t1 = time.perf_counter()
    mx.eval(y); t2 = time.perf_counter()
    err = mx.abs(y.astype(mx.float32) - snake(x, a).astype(mx.float32)).max().item()
    print(f"{dt} {shape} alpha={alpha} trace={t1 - t0:.4f}s eval={t2 - t1:.4f}s "
          f"max_abs_err_vs_eager={err:.3g}", flush=True)
