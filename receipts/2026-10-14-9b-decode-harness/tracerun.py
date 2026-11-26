"""tracerun.py <hf_repo> <ntok>: load the model offline, run one prefill and <ntok> greedy decode tokens on the default device.

Used by runprov.sh under MLX_OMARCHY_TRACE_DISPATCH=1 (kernel ids on stderr) and under gdb (CPU command-encoder count).
Prints one TRACERUN line: device, wheel version, route env, number of generated tokens, digest of the generated token ids.
"""
import hashlib
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import mlx.core as mx
from mlx_lm import load, stream_generate

repo, ntok = sys.argv[1], int(sys.argv[2])
model, tok = load(repo)
text = "Answer in one clear sentence: why seasons change on Earth."
ids = [r.token for r in stream_generate(model, tok, prompt=text, max_tokens=ntok)]
print("TRACERUN device=%s mlx=%s route_direct_env=%s tokens=%d digest=%s" % (
    mx.default_device(), mx.__version__, os.environ.get("MLX_OMARCHY_GATHER_QMM_DIRECT", "<unset>"),
    len(ids), hashlib.sha256(repr(ids).encode()).hexdigest()[:16]), flush=True)
