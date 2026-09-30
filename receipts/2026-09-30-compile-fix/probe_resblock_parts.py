"""Narrow the no-primitive failure inside Kokoro's AdaINResBlock1.

usage: probe_resblock_parts.py <repo>/scripts <kokoro-pack-dir>
Compiles sub-functions of generator.resblocks[0] and compares with eager.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from mlx_provenance import installed_provenance, provenance_line  # noqa: E402
import mlx.core as mx  # noqa: E402

print(provenance_line(installed_provenance()), flush=True)
from mlx_audio.tts.utils import load_model  # noqa: E402

blk = load_model(Path(sys.argv[2])).decoder.generator.resblocks[0]
c1, c2, n1, n2 = blk.convs1[0], blk.convs2[0], blk.adain1[0], blk.adain2[0]
a1 = blk.alpha1[0]
print(f"alpha dtype={a1.dtype} shape={a1.shape} conv w dtype={c1.weight_v.dtype}",
      flush=True)


def snake(x, a):
    return x + (1 / a) * (mx.sin(a * x) ** 2)


def conv(c, x):
    return c(x.swapaxes(2, 1), mx.conv1d).swapaxes(2, 1)


parts = {
    "adain": lambda x, s: n1(x, s),
    "snake_captured_alpha": lambda x, s: snake(x, a1),
    "adain_snake": lambda x, s: snake(n1(x, s), a1),
    "conv": lambda x, s: conv(c1, x),
    "snake_conv": lambda x, s: conv(c1, snake(x, a1)),
    "adain_snake_conv": lambda x, s: conv(c1, snake(n1(x, s), a1)),
    "one_iteration": lambda x, s: conv(c2, snake(n2(conv(c1, snake(n1(x, s), a1)), s),
                                             blk.alpha2[0])) + x,
    "whole_block": blk,
}
x = mx.random.normal((1, 256, 96))
s = mx.random.normal((1, 128))
mx.eval(x, s)
for name, fn in parts.items():
    ref = fn(x, s)
    mx.eval(ref)
    try:
        y = mx.compile(fn)(x, s)
        mx.eval(y)
        err = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        print(f"{name}: ok max_abs_err_vs_eager={err:.3g}", flush=True)
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"{name}: FAIL {type(exc).__name__}: {str(exc).splitlines()[0]}"
              f" at {Path(tb.filename).name}:{tb.lineno}", flush=True)
