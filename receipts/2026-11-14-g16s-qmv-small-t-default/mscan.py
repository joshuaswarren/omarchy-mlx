import os, sys, time, statistics
os.environ["MLX_OMARCHY_NO_MLXLM_WARNING"] = "1"
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
m, tok = load(sys.argv[1])
for M in (1, 2, 4, 8, 13, 16, 32, 64, 128):
    x = mx.array([[100 + i for i in range(M)]])
    xs = []
    for r in range(6):
        c = make_prompt_cache(m); s = time.perf_counter(); y = m(x, cache=c); mx.eval(y[:, -1]); xs.append((time.perf_counter() - s) * 1000)
    print("M", M, "ms med %.1f min %.1f" % (statistics.median(xs[1:]), min(xs[1:])), flush=True)
