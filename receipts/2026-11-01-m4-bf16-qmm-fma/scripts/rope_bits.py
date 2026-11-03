"""Dump or compare mx.fast.rope outputs bit for bit, and time decode RoPE.

usage: rope_bits.py dump|compare <file.npz>
compare exits 1 on any differing or missing case, before the timing.
The timing is host wall time per q+k pair (200 pairs, one eval).
Cases: bf16 and f16, base and freqs, traditional and not, decode (T=1) and
prefill-like (T=17) shapes, several offsets.
"""
import sys
import time

import prov

prov.check()
import mlx.core as mx
import numpy as np

mx.random.seed(3)
out = {}
cases = []
for dt in (mx.bfloat16, mx.float16):
    for freqs in (False, True):
        for trad in (False, True):
            for heads, t in ((16, 1), (4, 1), (8, 17)):
                for off in (0, 37, 4093):
                    cases.append((dt, freqs, trad, heads, t, off))
f = (1.0 / (10000.0 ** (mx.arange(0, 128, 2) / 128.0))).astype(mx.float32)
for i, (dt, freqs, trad, heads, t, off) in enumerate(cases):
    x = mx.random.normal((1, heads, t, 128)).astype(dt)
    if freqs:
        y = mx.fast.rope(x, 128, traditional=trad, base=None, scale=1.0, offset=off, freqs=1.0 / f)
    else:
        y = mx.fast.rope(x, 128, traditional=trad, base=10000.0, scale=1.0, offset=off)
    mx.eval(y)
    out[f"c{i}"] = np.array(y.view(mx.uint16))

mode, path = sys.argv[1], sys.argv[2]
if mode == "dump":
    np.savez(path, **out)
    print(f"dumped {len(out)} cases")
else:
    ref = np.load(path)
    bad = [k for k in out if not np.array_equal(out[k], ref[k])]
    for k in bad[:8]:
        print("MISMATCH", cases[int(k[1:])])
    print(f"{len(out) - len(bad)}/{len(out)} cases bit-identical")
    if bad or set(ref.files) != set(out):
        print("COMPARE FAILED", flush=True)
        sys.exit(1)

for dt in (mx.bfloat16, mx.float16):
    for freqs in (False, True):
        q = mx.random.normal((1, 16, 1, 128)).astype(dt)
        k = mx.random.normal((1, 4, 1, 128)).astype(dt)
        mx.eval(q, k)
        kw = dict(base=None, freqs=1.0 / f) if freqs else dict(base=10000.0)
        run = lambda: mx.eval([r for i in range(200) for r in (
            mx.fast.rope(q, 128, traditional=False, scale=1.0, offset=i, **kw),
            mx.fast.rope(k, 128, traditional=False, scale=1.0, offset=i, **kw))])
        run()
        t0 = time.perf_counter(); run(); dt_ = time.perf_counter() - t0
        print(f"{str(dt):14s} freqs={freqs!s:5s} q+k pair {dt_ / 200 * 1e6:6.1f} us")
