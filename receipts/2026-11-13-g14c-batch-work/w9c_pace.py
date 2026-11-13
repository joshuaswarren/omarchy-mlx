#!/usr/bin/env python3
"""w9C: sustained greedy decode with a 60 Hz frame-pacing probe (submit-latency, issue #19) running beside it.
Loads a model, prefills a tiled 512-token prompt, decodes N tokens timed,
reads the libmlx trace counters for submits per step, and runs the probe for the same window.
usage: w9c_pace.py MODEL_DIR OUT.json PROBE_BIN [TOKENS=600]
OUT.json: decode_tps, submits_per_step, ids_sha (first 64 ids), probe (n, p50, p99, max, mean ms, count over 2x frame)
"""
import ctypes
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
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


def submits():
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(snap))
    return snap.vk_submissions


out, probe_bin = sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 600
model, tok = load(sys.argv[1])
ids = tok.encode("The history of France begins in antiquity. ")
prompt = mx.array((ids * (512 // len(ids) + 1))[:512])


def decode(count):
    it = generate_step(prompt, model, max_tokens=count, prompt_cache=make_prompt_cache(model))
    got = [next(it)[0] for _ in range(count)]
    mx.synchronize()
    return [int(t) for t in got]


decode(16)
probe_s = 6
s0 = submits()
t0 = time.perf_counter()
probe = subprocess.Popen([probe_bin, str(probe_s), "--rate", "60"], stdout=subprocess.PIPE, text=True)
time.sleep(0.5)
got = decode(n)
elapsed = time.perf_counter() - t0
s1 = submits()
text = probe.communicate(timeout=60)[0].strip().splitlines()
dec_tps = round(n / (elapsed - 0.5), 2)
json.dump({"mx": mx.__version__, "decode_tps": dec_tps, "n": n,
           "submits_per_step": round((s1 - s0) / (n + 0.0), 3),
           "ids_sha": hashlib.sha256(json.dumps(got[:64]).encode()).hexdigest()[:16],
           "probe": text[-1] if text else ""}, open(out, "w"), indent=1)
print(json.dumps({"decode_tps": dec_tps, "submits_per_step": round((s1 - s0) / n, 3), "probe": text[-1] if text else ""}))
