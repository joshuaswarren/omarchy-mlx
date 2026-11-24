"""Dispatch trace of one long-prompt greedy decode on a release wheel (run under count_cpu.gdb.py to count CPU-stream dispatches).

usage: trace_split.py MODEL_DIR CTX N_TOKENS EXPECT_WHEEL REPO_ROOT TEXT_FILE [--cpu-control]
The prompt is the first CTX tokens of TEXT_FILE, so decode attention runs at CTX + 1 keys and up. Prints the scripts/mlx_provenance.py
line (refuses, exit 3, unless the installed wheel matches), then the trace counter deltas (MlxOmarchyTraceSnapshot) around the
N_TOKENS decode and the sha256 of the token ids. --cpu-control runs one explicit CPU-stream add instead (the counter's positive control).
"""
import ctypes, hashlib, importlib.metadata, json, sys
from pathlib import Path

FIELDS = ("gpu_primitive_dispatches", "vk_submissions", "vk_buffer_copies", "vk_buffer_fills", "vk_compute_dispatches",
          "omarchy_finalize_calls", "commit_calls_with_work", "commit_calls_noop", "barriers_emitted", "barriers_skipped")


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in FIELDS]


model_dir, ctx, n, wheel, repo, text = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6]
sys.path.insert(0, str(repo / "scripts"))
import mlx_provenance as mp  # noqa: E402
prov = mp.installed_provenance(expect_wheel=wheel)
print(mp.provenance_line(prov), flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    sys.exit(3)
import mlx.core as mx  # noqa: E402
lib = ctypes.CDLL(str(importlib.metadata.distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so")))
lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
assert int(lib.mlx_omarchy_trace_snapshot_abi_size()) == ctypes.sizeof(Snap)
lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(Snap)]


def read():
    s = Snap()
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return {f: int(getattr(s, f)) for f in FIELDS}


if "--cpu-control" in sys.argv:
    a = mx.arange(4096, dtype=mx.float32)
    mx.eval(mx.add(a, a, stream=mx.cpu))
    print("TRACE " + json.dumps({"arm": "cpu-control", "device": str(mx.default_device())}))
    sys.exit(0)
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
model, tok = load(model_dir)
ids = tok.encode(open(text).read())[:ctx]
gen = generate_step(mx.array(ids), model, prompt_cache=make_prompt_cache(model))
first, _ = next(gen)  # prefill and the first token outside the window
mx.eval(first)
before, out = read(), []
for t, _ in gen:
    mx.eval(t)
    out.append(int(t))
    if len(out) >= n:
        break
after = read()
print("TRACE " + json.dumps({"arm": "decode", "model": Path(model_dir).name, "ctx": ctx, "device": str(mx.default_device()), "tokens": len(out),
                             "token_sha256": hashlib.sha256(json.dumps(out).encode()).hexdigest(),
                             "delta": {k: after[k] - before[k] for k in FIELDS}}))
