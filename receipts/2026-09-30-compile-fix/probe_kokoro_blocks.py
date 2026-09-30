"""Compile each Kokoro decoder block alone and compare with eager.

usage: probe_kokoro_blocks.py <repo>/scripts <kokoro-pack-dir> [block ...]
Needs mlx_audio (Kokoro) importable. Prints one line per block.
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from mlx_provenance import installed_provenance, provenance_line  # noqa: E402
import mlx.core as mx  # noqa: E402

print(provenance_line(installed_provenance()), flush=True)
from mlx_audio.tts.utils import load_model  # noqa: E402

model = load_model(Path(sys.argv[2]))
dec = model.decoder
gen = dec.generator
T = 96
s = mx.random.normal((1, 128))


def channels(blk):
    return getattr(blk, "dim_in", None) or blk.convs1[0].weight_v.shape[2]


blocks = {
    "gen.resblocks[0]": gen.resblocks[0],
    "gen.noise_res[0]": gen.noise_res[0],
    "dec.encode": dec.encode,
    "dec.decode[0]": dec.decode[0],
    "dec.decode[3]": dec.decode[3],
}
wanted = sys.argv[3:] or list(blocks)
for name in wanted:
    blk = blocks[name]
    x = mx.random.normal((1, channels(blk), T))
    mx.eval(x, s)
    ref = blk(x, s)
    mx.eval(ref)
    try:
        t0 = time.perf_counter()
        y = mx.compile(blk)(x, s)
        t1 = time.perf_counter()
        mx.eval(y)
        t2 = time.perf_counter()
        err = mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max().item()
        print(f"{name} x={x.shape} ok trace={t1 - t0:.3f}s eval={t2 - t1:.3f}s "
              f"max_abs_err_vs_eager={err:.3g} ref_dtype={ref.dtype}", flush=True)
    except Exception as exc:  # report and continue with the next block
        tb = traceback.extract_tb(exc.__traceback__)[-1]
        print(f"{name} x={x.shape} FAIL {type(exc).__name__}: "
              f"{str(exc).splitlines()[0]} at {tb.filename}:{tb.lineno}",
              flush=True)
