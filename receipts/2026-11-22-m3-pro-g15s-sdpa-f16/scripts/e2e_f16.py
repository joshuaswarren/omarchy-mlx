"""End-to-end f16 decode of one model in one process: speed, route, logits.

usage: e2e_f16.py MODEL OUT.json OUT.npz EXPECT_WHEEL REPO_ROOT LABEL TEXT_FILE

The model's floating parameters are cast to float16 (model.set_dtype), so
activations, the KV cache and every single-query attention call are f16.
The arm is chosen by the caller (MLX_OMARCHY_SDPA_F16_NO_COOPMAT).

For each context (prompt) length in CTX (environment E2E_CTX, comma-separated;
default 64,1100,3000), tokens taken from TEXT_FILE:
  - speed: one warmup and RUNS timed greedy generations of GEN tokens
    (mlx_lm.stream_generate, generation_tps), token-id sha per run (must
    not change within the process);
  - route: Vulkan compute dispatches per generated token over the timed
    runs (trace snapshot ABI);
  - logits: teacher-forced decode of the next FORCE tokens of the text,
    one token per step on the prompt cache (single-query attention every
    step); the full-vocabulary log-softmax of every step goes to the npz
    as float32 for the cross-arm comparison (cmp_logits.py).
"""
import ctypes
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

mdl, out_json, out_npz, expect_wheel, repo_root, label, text_file = sys.argv[1:8]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp  # noqa: E402

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
pline = mp.provenance_line(prov)
print(pline, flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    print("PROVENANCE_REFUSAL", file=sys.stderr)
    sys.exit(3)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

lib = ctypes.CDLL(os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so"))


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "primitives", "submissions", "copies", "fills", "compute", "finalize",
        "commit_work", "commit_noop", "barriers", "barriers_skipped")]


lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
assert lib.mlx_omarchy_trace_snapshot_abi_size() == ctypes.sizeof(Snap)


def snap():
    s = Snap()
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return s


CTX = tuple(int(x) for x in os.environ.get("E2E_CTX", "64,1100,3000").split(","))
GEN = 128
RUNS = 3
FORCE = 48

model, tok = load(mdl)
model.set_dtype(mx.float16)
mx.eval(model.parameters())
text_ids = tok.encode(open(text_file).read())
assert len(text_ids) >= max(CTX) + FORCE, len(text_ids)
greedy = make_sampler(temp=0.0)

rows, arrays = [], {}
for ctx in CTX:
    prompt = text_ids[:ctx]
    sha0, tps, disp, subs, gen_tokens = None, [], 0, 0, 0
    for r in range(RUNS + 1):
        ids, last = [], None
        s0 = snap()
        for resp in stream_generate(model, tok, prompt, max_tokens=GEN, sampler=greedy):
            ids.append(int(resp.token))
            last = resp
        mx.synchronize()
        s1 = snap()
        h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        if r == 0:
            sha0, text = h, tok.decode(ids)
            continue
        if h != sha0:
            print(json.dumps({"error": "ids changed within process", "ctx": ctx}), flush=True)
            sys.exit(4)
        tps.append(round(last.generation_tps, 3))
        disp += s1.compute - s0.compute
        subs += s1.submissions - s0.submissions
        gen_tokens += len(ids)

    # teacher-forced decode logits
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)
    mx.eval(logits)
    lps = []
    for t in text_ids[ctx:ctx + FORCE]:
        lp = logits[0, -1].astype(mx.float32)
        lp = lp - mx.logsumexp(lp)
        mx.eval(lp)
        lps.append(np.array(lp))
        logits = model(mx.array([[t]]), cache=cache)
        mx.eval(logits)
    arrays[f"ctx{ctx}"] = np.stack(lps)

    row = {"ctx": ctx, "gen": GEN, "decode_tps": tps, "decode_tps_median": statistics.median(tps),
           "ids_sha": sha0, "vk_compute_dispatches_per_token": round(disp / gen_tokens, 2),
           "submissions_per_token": round(subs / gen_tokens, 3),
           "forced_steps": FORCE, "text": " ".join(text.split())[:120]}
    rows.append(row)
    print(label, json.dumps(row), flush=True)

np.savez_compressed(out_npz, **arrays)
peak = getattr(mx, "get_peak_memory", None)
json.dump({"label": label, "provenance": pline, "model": Path(mdl).name, "dtype": "float16 (set_dtype)",
           "env_MLX_OMARCHY_SDPA_F16_NO_COOPMAT": os.environ.get("MLX_OMARCHY_SDPA_F16_NO_COOPMAT"),
           "env_MLX_OMARCHY_BATCH_WORK": os.environ.get("MLX_OMARCHY_BATCH_WORK"),
           "peak_gpu_gib": round(peak() / 2**30, 3) if peak else None, "rows": rows},
          open(out_json, "w"), indent=1)
print("DONE", label, flush=True)
