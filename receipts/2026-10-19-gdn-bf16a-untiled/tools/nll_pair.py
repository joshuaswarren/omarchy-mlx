"""nll_pair.py MODEL_DIR TEXT_FILE OUT.npz [--prefixes P] [--first F] [--steps 64] [--tokens 512] [--expect-libmlx HEX16]

Teacher-forced decode of a real model on a fixed text (the H90 method with the structure the pre-registration fixes). The text file is tokenized with
the model's tokenizer. Prefix p (p = first .. first + P - 1) is the token window ids[o : o + T] with o = p * (T + STEPS + 1): prefilled in one call, then
STEPS decode steps (batch 1, one token each, the true next tokens fed) whose logits score the true next token: nll[p, i] = logsumexp(logits) - logits[target].
Also saved: top1[p, i] (argmax token of each step's logits), and the sha256 of the float32 logits of the last step of each prefix.
The route is chosen by the caller's environment (MLX_OMARCHY_GATHER_QMM_WORD, MLX_OMARCHY_GATHER_QMM_SUB). Prints the provenance line first.
The decode path under test is m == 1; the prefill route is the same in every arm.
"""
import argparse
import glob
import hashlib
import importlib.metadata as md
import json
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("text")
ap.add_argument("out")
ap.add_argument("--prefixes", type=int, default=64)
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--expect-libmlx", default="")
a = ap.parse_args()
T, STEPS, P, F = a.tokens, a.steps, a.prefixes, a.first


def lib_hashes():
    base = os.path.dirname(mx.__file__)
    out = {}
    for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))) + sorted(glob.glob(os.path.join(base, "libmlx*"))):
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                out[os.path.basename(p)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


libs = lib_hashes()
print("PROVENANCE mlx-omarchy", md.version("mlx-omarchy"), "device", mx.device_info().get("device_name"), "libs", json.dumps(libs),
      "WORD", os.environ.get("MLX_OMARCHY_GATHER_QMM_WORD", "unset"), "SUB", os.environ.get("MLX_OMARCHY_GATHER_QMM_SUB", "unset"),
      "DIRECT", os.environ.get("MLX_OMARCHY_GATHER_QMM_DIRECT", "unset"), "load1", os.getloadavg()[0], flush=True)
if a.expect_libmlx and set(v for k, v in libs.items() if k == "libmlx.so") != {a.expect_libmlx}:
    print("LIBMLX MISMATCH, wanted", a.expect_libmlx)
    sys.exit(4)
with open(a.text, "rb") as fh:
    raw = fh.read()
print("TEXT sha256", hashlib.sha256(raw).hexdigest(), "bytes", len(raw), flush=True)
model, tok = load(a.model)
ids = tok.encode(raw.decode("utf-8"))
need = (F + P) * (T + STEPS + 1)
assert len(ids) >= need, (len(ids), need)
print("IDS", len(ids), "needed", need, "prefix windows", F, "to", F + P - 1, flush=True)
nll = np.zeros((P, STEPS), dtype=np.float64)
top1 = np.zeros((P, STEPS), dtype=np.int64)
last_sha = []
step_times = []
for pi in range(P):
    off = (F + pi) * (T + STEPS + 1)
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array(ids[off: off + T])[None], cache=cache))
    mx.synchronize()
    for i in range(STEPS):
        x = mx.array([[ids[off + T + i]]])
        t0 = time.perf_counter()
        logits = model(x, cache=cache)[0, -1].astype(mx.float32)
        target = ids[off + T + i + 1]
        val = mx.logsumexp(logits) - logits[target]
        am = mx.argmax(logits)
        mx.eval(val, am)
        mx.synchronize()
        step_times.append(time.perf_counter() - t0)
        nll[pi, i] = float(val.item())
        top1[pi, i] = int(am.item())
    last_sha.append(hashlib.sha256(np.array(logits).tobytes()).hexdigest()[:16])
    if pi % 8 == 7 or pi == P - 1:
        print(f"prefix {F + pi} running mean_nll {nll[:pi + 1].mean():.5f} step_median_ms {statistics.median(step_times) * 1e3:.2f}", flush=True)
np.savez(a.out, nll=nll, top1=top1, last_sha=np.array(last_sha), first=F)
digest = hashlib.sha256(top1.tobytes()).hexdigest()[:16]
print(f"NLL mean {nll.mean():.5f} over {nll.size} positions; top1_digest {digest}; last_logits_digest "
      f"{hashlib.sha256(''.join(last_sha).encode()).hexdigest()[:16]}; step_median_ms {statistics.median(step_times) * 1e3:.2f}", flush=True)
