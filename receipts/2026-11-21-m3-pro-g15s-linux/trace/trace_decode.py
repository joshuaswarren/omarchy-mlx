"""Dispatch trace of one greedy decode on the release wheel.

Loads a model, generates N greedy tokens and prints the backend trace counter
deltas (MlxOmarchyTraceSnapshot via the wheel's libmlx) around the run, the
default device and a sha256 of the token ids. Run it under count_cpu.gdb.py to
count CPU-stream primitive dispatches in the same process.

usage: trace_decode.py MODEL_DIR N_TOKENS [--cpu-control]
"""
import ctypes, hashlib, importlib.metadata, json, sys

FIELDS = ("gpu_primitive_dispatches", "vk_submissions", "vk_buffer_copies",
          "vk_buffer_fills", "vk_compute_dispatches", "omarchy_finalize_calls",
          "commit_calls_with_work", "commit_calls_noop", "barriers_emitted",
          "barriers_skipped")


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in FIELDS]


def reader():
    import mlx.core  # noqa: F401  loads libmlx
    so = importlib.metadata.distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so")
    lib = ctypes.CDLL(str(so))
    lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
    assert int(lib.mlx_omarchy_trace_snapshot_abi_size()) == ctypes.sizeof(Snap), "trace ABI size mismatch"
    fn = lib.mlx_omarchy_trace_snapshot
    fn.argtypes = [ctypes.POINTER(Snap)]
    fn.restype = None

    def read():
        s = Snap()
        fn(ctypes.byref(s))
        return {n: int(getattr(s, n)) for n in FIELDS}
    return read


def main():
    model_dir, n = sys.argv[1], int(sys.argv[2])
    import mlx.core as mx
    read = reader()
    if "--cpu-control" in sys.argv:
        a = mx.arange(4096, dtype=mx.float32)
        mx.eval(mx.add(a, a, stream=mx.cpu))
        print("TRACE " + json.dumps({"arm": "cpu-control", "device": str(mx.default_device())}))
        return
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(model_dir)
    ids = tok.encode("Explain how a CPU cache works in two sentences.")
    before = read()
    out = []
    for t, _ in generate_step(mx.array(ids), model, prompt_cache=make_prompt_cache(model)):
        mx.eval(t)
        out.append(int(t))
        if len(out) >= n:
            break
    after = read()
    delta = {k: after[k] - before[k] for k in FIELDS}
    print("TRACE " + json.dumps({"arm": "decode", "model": model_dir.rstrip("/").split("/")[-1],
                                 "device": str(mx.default_device()), "tokens": len(out),
                                 "token_sha256": hashlib.sha256(json.dumps(out).encode()).hexdigest(),
                                 "delta": delta}))


main()
