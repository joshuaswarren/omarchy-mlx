#!/usr/bin/env python3
"""w9C: barriers, dispatches and submits per decode step. Loads a model, runs a 512-token prompt twice with 9 and with 41 greedy tokens;
the difference over 32 steps is the per-step count (prefill cancels). Counters: libmlx trace snapshot (barriers_emitted / barriers_skipped).
usage: w9c_barcount.py MODEL_DIR OUT.json
"""
import ctypes
import json
import pathlib
import sys

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.utils import load

FIELDS = ("gpu_primitive_dispatches", "vk_submissions", "vk_buffer_copies", "vk_buffer_fills",
          "vk_compute_dispatches", "omarchy_finalize_calls", "commit_calls_with_work",
          "commit_calls_noop", "barriers_emitted", "barriers_skipped")


class Snapshot(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in FIELDS]


lib = ctypes.CDLL(str(pathlib.Path(mx.__file__).parent / "lib" / "libmlx.so"))
lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
if ctypes.sizeof(Snapshot) != int(lib.mlx_omarchy_trace_snapshot_abi_size()):
    raise SystemExit("trace snapshot layout differs from libmlx")
lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(Snapshot)]
snap = Snapshot()


def read():
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(snap))
    return {n: getattr(snap, n) for n in FIELDS}


model, tok = load(sys.argv[1])
ids = tok.encode("The history of France begins in antiquity. ")
ids = (ids * (512 // len(ids) + 1))[:512]
prompt = mx.array(ids)


def run(n):
    from mlx_lm.models.cache import make_prompt_cache
    before = read()
    it = generate_step(prompt, model, max_tokens=n, prompt_cache=make_prompt_cache(model))
    for _ in range(n):
        next(it)
    mx.synchronize()
    after = read()
    return {k: after[k] - before[k] for k in FIELDS}


run(9)
a, b = run(9), run(41)
per = {k: round((b[k] - a[k]) / 32, 2) for k in FIELDS}
json.dump({"mx": mx.__version__, "per_decode_step": per, "run9": a, "run41": b}, open(sys.argv[2], "w"), indent=1)
print(json.dumps(per))
