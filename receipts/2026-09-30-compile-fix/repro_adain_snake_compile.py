"""Repro B: AdaIN + Snake under mx.compile ("eval an array without a primitive")."""
import sys
sys.path.insert(0, sys.argv[1])  # <repo>/scripts, for mlx_provenance
from mlx_provenance import installed_provenance, provenance_line
import mlx.core as mx
print(provenance_line(installed_provenance()), flush=True)
C, T, S = 256, 96, 128  # Kokoro generator.resblocks[0] sizes
w, b = mx.random.normal((2 * C, S)) * 0.05, mx.random.normal((2 * C,)) * 0.05
alpha = mx.random.uniform(0.5, 2.0, (1, C, 1))
def adain_snake(x, s):  # Kokoro AdaIN1d then Snake1D, alpha captured
    gamma, beta = mx.split(mx.expand_dims(mx.addmm(b, s, w.T), 2), 2, axis=1)
    xn = (x - mx.mean(x, axis=2, keepdims=True)) / mx.sqrt(mx.var(x, axis=2, keepdims=True) + 1e-5)
    y = (1 + gamma) * xn + beta
    return y + (1 / alpha) * (mx.sin(alpha * y) ** 2)
x, s = mx.random.normal((1, C, T)), mx.random.normal((1, S)); mx.eval(w, b, alpha, x, s)
ref = adain_snake(x, s); mx.eval(ref)
y = mx.compile(adain_snake)(x, s); mx.eval(y)  # raises on the unfixed wheel
print("ok max_abs_err_vs_eager", mx.abs(y - ref).max().item(), flush=True)
