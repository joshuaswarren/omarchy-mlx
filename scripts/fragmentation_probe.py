#!/usr/bin/env python3
"""Per-token fragmentation attribution at the Python/MLX boundary.

Wraps every boundary call mlx_lm 0.31.3's generate_step makes into MLX
(eval, async_eval, array.item, tolist, bool/float/index/numpy conversion),
samples the backend trace counters (trace.h snapshot, ctypes) around each
crossing, and dumps one NDJSON event per crossing plus a marker per token.

Run on llvmpipe (x86 dev box):

  MLX_DISABLE_COMPILE=1 MLX_OMARCHY_ALLOW_NON_APPLE=1 \
  MLX_OMARCHY_GPU_PROFILE=/tmp/p.jsonl \
  python3 scripts/fragmentation_probe.py \
    --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/<hash> \
    --prompt "The quick brown fox jumps over the lazy dog." \
    --max-tokens 48 --events /tmp/events.jsonl --markers /tmp/m.jsonl

The counters make each token's window attributable:
  gpu_primitive_dispatches  = gpu::eval calls (one per primitive)
  omarchy_finalize_calls    = finalize points (throttle + graph end)
  commit_calls_with_work    = real batches submitted
  commit_calls_noop         = commits that found nothing pending
  vk_submissions            = QueueSubmit calls
  vk_compute_dispatches     = compute dispatches recorded
"""

import argparse
import ctypes
import json
import pathlib
import sys
import time


def make_counter_reader():
    import mlx.core  # import loads libmlx into this process

    so = pathlib.Path(mlx.core.__file__).parent / "lib" / "libmlx.so"

    class Snapshot(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in (
                "gpu_primitive_dispatches",
                "vk_submissions",
                "vk_buffer_copies",
                "vk_buffer_fills",
                "vk_compute_dispatches",
                "omarchy_finalize_calls",
                "commit_calls_with_work",
                "commit_calls_noop",
            )
        ]

    lib = ctypes.CDLL(str(so))
    lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(Snapshot)]
    lib.mlx_omarchy_trace_snapshot.restype = None
    snap = Snapshot()

    def read():
        lib.mlx_omarchy_trace_snapshot(ctypes.byref(snap))
        return {name: getattr(snap, name) for name, _ in Snapshot._fields_}

    return read


def caller_site(skip=2):
    frame = sys._getframe(skip)
    for _ in range(6):
        if frame is None:
            return "?"
        name = frame.f_code.co_filename
        if "fragmentation_probe" not in name and "importlib" not in name:
            return f"{pathlib.Path(name).name}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


def install_boundary_hooks(events, read):
    import mlx.core as mx

    def wrap(obj, attr, kind):
        orig = getattr(obj, attr, None)
        if orig is None:
            return

        def wrapped(*a, **k):
            before = read()
            t0 = time.monotonic_ns()
            try:
                return orig(*a, **k)
            finally:
                after = read()
                events.append(
                    {
                        "t": t0,
                        "us": (time.monotonic_ns() - t0) / 1e3,
                        "kind": kind,
                        "at": caller_site(),
                        "d": {n: after[n] - before[n] for n in after},
                    }
                )

        setattr(obj, attr, wrapped)

    wrap(mx, "eval", "eval")
    wrap(mx, "async_eval", "async_eval")
    wrap(mx.array, "item", "item")
    wrap(mx.array, "tolist", "tolist")
    wrap(mx.array, "__bool__", "bool")
    wrap(mx.array, "__float__", "float")
    wrap(mx.array, "__index__", "index")
    wrap(mx.array, "__array__", "numpy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events", default="events.jsonl")
    ap.add_argument("--markers", default="markers.jsonl")
    ap.add_argument("--prim-dump", default="")
    args = ap.parse_args()

    read = make_counter_reader()

    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    import mlx.core as mx  # noqa: F401 - ensure libmlx is loaded

    if args.prim_dump:
        so = pathlib.Path(mx.__file__).parent / "lib" / "libmlx.so"
        lib = ctypes.CDLL(str(so))
        lib.mlx_omarchy_prim_reset.argtypes = []
        lib.mlx_omarchy_prim_reset()
    events = []
    markers = open(args.markers, "w", encoding="utf-8")
    install_boundary_hooks(events, read)

    mx.random.seed(args.seed)

    def mark(phase, n=-1):
        markers.write(
            json.dumps({"t": time.monotonic_ns(), "p": phase, "n": n}) + "\n"
        )
        markers.flush()

    mark("load_start")
    model, tokenizer = load(args.model)
    mark("load_done")

    from mlx_lm.generate import generate_step

    prompt_ids = tokenizer.encode(args.prompt)
    sampler = make_sampler(temp=0.0)

    mark("prefill_start")
    per_token = []
    if args.prim_dump:
        so = pathlib.Path(mx.__file__).parent / "lib" / "libmlx.so"
        lib = ctypes.CDLL(str(so))
        lib.mlx_omarchy_prim_dump.argtypes = [ctypes.c_char_p]
        lib.mlx_omarchy_prim_reset.argtypes = []
        lib.mlx_omarchy_prim_reset()
    read()  # drain any load-time counters from the deltas below
    start = read()
    for token, _logprobs in generate_step(
        mx.array(prompt_ids),
        model,
        max_tokens=args.max_tokens,
        sampler=sampler,
    ):
        snap = read()
        per_token.append({"t": time.monotonic_ns(), "c": snap})
        mark("tok", len(per_token))
    end = read()
    mark("decode_done")
    markers.close()

    total = {n: end[n] - start[n] for n in end}
    events = events[-(args.max_tokens * 8):]  # decode-phase crossings only

    with open(args.events, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    def summarize(lo, hi):
        t0 = per_token[lo]["t"]
        t1 = per_token[hi]["t"] if hi < len(per_token) else time.monotonic_ns()
        win = [e for e in events if t0 <= e["t"] < t1]
        out = {}
        for e in win:
            key = (e["kind"], e["at"])
            cell = out.setdefault(
                key, {"n": 0, "us": 0.0, "d": {n: 0 for n in e["d"]}}
            )
            cell["n"] += 1
            cell["us"] += e["us"]
            for n, v in e["d"].items():
                cell["d"][n] += v
        return out

    print(f"prompt_tokens={len(prompt_ids)} generated={len(per_token)}")
    print("TOTALS (decode phase):")
    for n, v in total.items():
        print(f"  {n}: {v}")

    print("\nPER-TOKEN (boundary crossings in window, counter deltas):")
    header = (
        "tok | item ev aev | prims subs fin wk noop | disp | us_item us_async"
    )
    print(header)
    for i in range(1, len(per_token)):
        s = summarize(i - 1, i)
        items = sum(v["n"] for (k, _), v in s.items() if k == "item")
        evals = sum(v["n"] for (k, _), v in s.items() if k == "eval")
        aevals = sum(v["n"] for (k, _), v in s.items() if k == "async_eval")
        d = {n: 0 for n in per_token[i]["c"]}
        for (_, _), v in s.items():
            for n, x in v["d"].items():
                d[n] += x
        us_item = sum(
            v["us"] for (k, _), v in s.items() if k == "item"
        )
        us_async = sum(
            v["us"] for (k, _), v in s.items() if k == "async_eval"
        )
        print(
            f"{i:3d} | {items:4d} {evals:2d} {aevals:3d} "
            f"| {d['gpu_primitive_dispatches']:5d} {d['vk_submissions']:4d} "
            f"{d['omarchy_finalize_calls']:3d} {d['commit_calls_with_work']:3d} "
            f"{d['commit_calls_noop']:4d} | {d['vk_compute_dispatches']:4d} "
            f"| {us_item:8.1f} {us_async:8.1f}"
        )

    print("\nATTRIBUTION (decode phase, by kind and call site):")
    for (kind, at), v in sorted(summarize(0, len(per_token)).items()):
        print(
            f"  {kind:10s} {at:40s} n={v['n']:4d} us={v['us']:10.1f} "
            f"prims={v['d']['gpu_primitive_dispatches']:6d} "
            f"subs={v['d']['vk_submissions']:5d}"
        )

    if args.prim_dump:
        so = pathlib.Path(mx.__file__).parent / "lib" / "libmlx.so"
        lib = ctypes.CDLL(str(so))
        lib.mlx_omarchy_prim_dump.argtypes = [ctypes.c_char_p]
        lib.mlx_omarchy_prim_dump(args.prim_dump.encode())
        print(f"\nprimitive name counts: {args.prim_dump}")


if __name__ == "__main__":
    main()
