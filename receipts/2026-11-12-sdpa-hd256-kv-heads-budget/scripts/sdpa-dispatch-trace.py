# Dispatch trace for the width-256 value-phase route: MLX_OMARCHY_TRACE_DISPATCH=1 prints one "[rtmod] DISPATCH kernel=<id> count=<n> gx gy gz"
# line per dispatch on stderr. This script runs one engaged SDPA call per arm (old pass 1 = "0", variant = "1") in a child process each,
# so the trace of one arm cannot leak into the other, and writes both traces plus the per-arm set of (kernel id, grid) pairs.
# A CPU-stream add is the negative control (no DISPATCH expected); a GPU add is the positive control (DISPATCH expected).
import os, subprocess, sys, json, collections

CTX = int(os.environ.get("SDPA_CTX", "480"))
KVH = int(os.environ.get("SDPA_KVH", "2"))
QH = int(os.environ.get("SDPA_QH", "8"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mlx_provenance import installed_provenance
_PROV = installed_provenance()
print("PROVENANCE " + json.dumps(_PROV, sort_keys=True), flush=True)
if _PROV["verified"] == "mismatch":
    raise SystemExit("provenance refusal: " + str(_PROV["mismatch"]))

CHILD = r'''
import os, sys
import mlx.core as mx
mode = sys.argv[1]
if mode == "cpu-add":
    sys.stderr.write("[trace-begin]\n"); sys.stderr.flush()
    a = mx.ones((256,), stream=mx.cpu); mx.eval(mx.add(a, a, stream=mx.cpu))
    sys.stderr.write("[trace-end]\n"); sys.stderr.flush()
elif mode == "gpu-add":
    a = mx.ones((256,), stream=mx.cpu); mx.eval(a); sys.stderr.write("[trace-begin]\n"); sys.stderr.flush(); mx.eval(mx.add(a, a)); sys.stderr.write("[trace-end]\n"); sys.stderr.flush()
else:
    N, QH, KVH = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    mx.random.seed(3)
    cap = N + 64
    q = mx.random.normal((1, QH, 1, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, KVH, cap, 256)).astype(mx.bfloat16)[:, :, :N, :]
    v = mx.random.normal((1, KVH, cap, 256)).astype(mx.bfloat16)[:, :, :N, :]
    mx.eval(q, k, v)
    sys.stderr.write("[trace-begin]\n"); sys.stderr.flush()
    mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=256 ** -0.5))
    sys.stderr.write("[trace-end]\n"); sys.stderr.flush()
'''

def run(mode, flag=None):
    env = dict(os.environ, MLX_OMARCHY_TRACE_DISPATCH="1", MLX_OMARCHY_SDPA_DECODE_SPLIT="1")
    if flag is not None:
        env["MLX_OMARCHY_SDPA_SPLIT_PV"] = flag
    r = subprocess.run([sys.executable, "-c", CHILD, mode, str(CTX), str(QH), str(KVH)], env=env, capture_output=True, text=True)
    lines = r.stderr.splitlines()
    if r.returncode != 0:  # a failed child must say why
        print("CHILD-FAILED " + mode + " rc=" + str(r.returncode) + " stderr-tail: " + " | ".join(lines[-4:]), flush=True)
    if "[trace-begin]" in lines and "[trace-end]" in lines:  # keep only the engaged call, not the buffer setup
        lines = lines[lines.index("[trace-begin]") + 1: lines.index("[trace-end]")]
    disp = [l for l in lines if "DISPATCH" in l]
    return r.returncode, disp

# The trace prints raw lines; keys below are the distinct dispatches, so a different route shows as a different set.
out = {}
for name, mode, flag in (("arm0-old-pass1", "sdpa", "0"), ("arm1-variant", "sdpa", "1"), ("control-cpu-add", "cpu-add", None), ("control-gpu-add", "gpu-add", None)):
    rc, disp = run(mode, flag)
    out[name] = {"rc": rc, "dispatches": len(disp), "distinct": sorted(set(l.split("count=")[0] + " " + " ".join(l.split()[-3:]) for l in disp))}
    print(name, rc, len(disp), flush=True)
    for l in disp:
        print("  " + l)
print("TRACE-JSON " + json.dumps(out, sort_keys=True))
ok = out["control-gpu-add"]["dispatches"] > 0 and out["control-cpu-add"]["dispatches"] == 0 and out["arm1-variant"]["distinct"] != out["arm0-old-pass1"]["distinct"]
print("FUSE-DONE" if ok else "TRACE-NOT-DISCRIMINATING")
