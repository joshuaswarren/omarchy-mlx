"""Long-context greedy decode of one build with its default routes (every MLX_OMARCHY_SDPA variable unset), one process.

usage: ab_decode.py MODEL OUT.json OUT.npz EXPECT_WHEEL REPO_ROOT TEXT_FILE

For each context length in CTX (environment CTX; default 443,1024,2048,4000), the prompt is the first CTX tokens of TEXT_FILE:
  - one warm-up generation, then RUNS timed generations of GEN greedy tokens (mlx_lm.stream_generate generation_tps and prompt_tps);
    the token ids must not change within the process (exit 4); Vulkan compute dispatches per generated token (trace snapshot ABI);
  - teacher-forced decode of the next FORCE tokens of the text on the prompt cache (512-token chunked prefill, one query token per step):
    the full-vocabulary log-softmax of every step goes to OUT.npz (float32) for the cross-build comparison (cmp_ab.py).
Prints the scripts/mlx_provenance.py line and refuses (exit 3) unless the installed wheel matches EXPECT_WHEEL.
"""
import ctypes
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

mdl, out_json, out_npz, expect_wheel, repo_root, text_file = sys.argv[1:7]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp  # noqa: E402

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
pline = mp.provenance_line(prov)
print(pline, flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    sys.exit(3)
assert not any(k.startswith("MLX_OMARCHY_SDPA") for k in os.environ), "the SDPA variables must be unset"

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


def compute_dispatches():
    s = Snap()
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return s.compute


CTX = tuple(int(x) for x in os.environ.get("CTX", "443,1024,2048,4000").split(","))
GEN, RUNS, FORCE = 128, 2, 32
model, tok = load(mdl)
text_ids = tok.encode(open(text_file).read())
assert len(text_ids) >= max(CTX) + FORCE
greedy = make_sampler(temp=0.0)

rows, arrays = [], {}
for ctx in CTX:
    prompt = text_ids[:ctx]
    sha0, tps, pf, disp = None, [], [], 0
    for r in range(RUNS + 1):
        ids, last = [], None
        d0 = compute_dispatches()
        for resp in stream_generate(model, tok, prompt, max_tokens=GEN, sampler=greedy):
            ids.append(int(resp.token))
            last = resp
        mx.synchronize()
        h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
        if r == 0:
            sha0 = h
            continue
        if h != sha0:
            print(json.dumps({"error": "ids changed within process", "ctx": ctx}), flush=True)
            sys.exit(4)
        tps.append(round(last.generation_tps, 3))
        pf.append(round(last.prompt_tps, 1))
        disp += compute_dispatches() - d0
    cache = make_prompt_cache(model)
    for i in range(0, ctx, 512):
        logits = model(mx.array(prompt[i:i + 512])[None], cache=cache)
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
    del cache, logits
    mx.clear_cache()
    row = {"ctx": ctx, "gen": GEN, "decode_tps": tps, "decode_tps_median": statistics.median(tps), "prefill_tps": pf,
           "ids_sha": sha0, "vk_compute_dispatches_per_token": round(disp / (RUNS * GEN), 2), "forced_steps": FORCE}
    rows.append(row)
    print("AB", json.dumps(row), flush=True)
np.savez_compressed(out_npz, **arrays)
json.dump({"provenance": pline, "model": Path(mdl).name, "rows": rows, "load1_end": os.getloadavg()[0]}, open(out_json, "w"), indent=1)
print("DONE", flush=True)
