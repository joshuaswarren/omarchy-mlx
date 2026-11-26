#!/usr/bin/env python3
"""Dispatch census for one rope-norm arm (run once per arm, the arm is set by the environment).

Loads the model, then runs the fixed 512-token prompt twice, fully drained, with 9 and with 41 greedy
tokens. The difference of the two runs over 32 decode steps is the per-decode-step cost; prefill cancels.
Counted per step: calls of mx.fast.rope_rms_norm (wrapped here) and the libmlx trace counters
gpu_primitive_dispatches and vk_compute_dispatches. Prints one JSON line.

Usage: <venv python> fold_dispatch_probe.py MODEL_DIR
"""
import ctypes
import hashlib
import json
import os
import pathlib
import random
import sys

import mlx.core as mx
from mlx_lm.generate import generate_step, generation_stream
from mlx_lm.utils import load

FIELDS = ("gpu_primitive_dispatches", "vk_submissions", "vk_buffer_copies", "vk_buffer_fills",
          "vk_compute_dispatches", "omarchy_finalize_calls", "commit_calls_with_work",
          "commit_calls_noop", "barriers_emitted", "barriers_skipped")


def counter_reader():
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

    return read


def main(model_dir):
    read = counter_reader()
    calls = [0]
    orig = mx.fast.rope_rms_norm

    def counted(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    mx.fast.rope_rms_norm = counted
    model, _ = load(model_dir)
    vocab = getattr(getattr(model, "args", None), "vocab_size", None) or 30000
    rng = random.Random(292)
    prompt = mx.array([rng.randrange(1000, vocab - 1) for _ in range(512)])

    def run(n):
        before, c0 = read(), calls[0]
        toks = [int(t) for t, _ in generate_step(prompt, model, max_tokens=n, prefill_step_size=512)]
        mx.synchronize(generation_stream)
        mx.synchronize()
        after = read()
        return toks, {k: after[k] - before[k] for k in ("gpu_primitive_dispatches", "vk_compute_dispatches")} | {
            "rope_rms_norm_calls": calls[0] - c0}

    t9, d9 = run(9)
    t41, d41 = run(41)
    per_step = {k: (d41[k] - d9[k]) / 32 for k in d9}
    print(json.dumps({
        "env": {k: os.environ.get(k) for k in ("MLX_OMARCHY_ROPE_NORM_FUSE", "MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE")},
        "default_device": str(mx.default_device()),
        "run9": d9, "run41": d41, "per_decode_step": per_step,
        "tokens9_sha256": hashlib.sha256(",".join(map(str, t9)).encode()).hexdigest(),
        "tokens41_sha256": hashlib.sha256(",".join(map(str, t41)).encode()).hexdigest(),
    }), flush=True)
    print("M2_INFERENCE_DRAIN_OK", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        raise SystemExit("usage: fold_dispatch_probe.py MODEL_DIR")
    main(sys.argv[1])
