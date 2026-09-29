#!/usr/bin/env python3
"""Last-logits patch kernel-level check (Jw16PrefillGap3 W2).

For each T in argv (default 512 2048):
  arm FULL  (MLX_OMARCHY_FULL_LOGITS=1): logits_full = model(ids, cache)  [1,T,V]
  arm LAST  (env unset):                 logits_last = model(ids, cache)  [1,1,V]
Each arm runs in a FRESH SUBPROCESS (mx.compile traces / lazily-evaluated state
cannot leak across arms). Comparisons, per T:
  sha256(uint16 bits of logits_full[0,-1,:])  vs  sha256(uint16 bits of logits_last[0,0,:])
  -> LAST-ROW BITWISE: must be identical (same quantized-head GEMM row)
  argmax equal + both rows finite
  head-level isolation: y1 = as_linear(hidden[:, -1:, :]) vs y_t = as_linear(hidden)[:, -1, :]
  -> HEAD M=1 vs M=T row bitwise (isolates qmm row-invariance from the model path)
  LAST arm twice -> determinism
Exit 0 iff every check passes.

usage: pg3_lmhead_check.py MODEL_DIR T [T ...]
"""
import hashlib
import json
import os
import subprocess
import sys

model_dir = sys.argv[1]
Ts = [int(t) for t in sys.argv[2:]] or [512, 2048]

CHILD = r"""
import hashlib, json, os, sys
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model_dir, arm, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
if arm == "FULL":
    os.environ["MLX_OMARCHY_FULL_LOGITS"] = "1"
prompts_path = os.path.expanduser("~/bench-scripts/qwen38-2b-prompts.jsonl")
prompts = [json.loads(l)["text"] for l in open(prompts_path) if l.strip()]
model, tok = load(model_dir)
text = " ".join(prompts)
base = tok.encode(text)
ids = list(base)
while len(ids) < T:
    ids = ids + tok.encode(" " + text)
ids = ids[:T]
x = mx.array(ids)[None]

# model-level (what the serving path executes)
cache = make_prompt_cache(model)
logits = model(x, cache=cache)
sha = hashlib.sha256(np.array(logits[0, -1, :].view(mx.uint16)).tobytes()).hexdigest()[:16]
shape = list(logits.shape)
am = mx.argmax(logits[0, -1, :])
finite = bool(mx.all(mx.isfinite(logits)))
mx.eval(am)
out = {"arm": arm, "T": T, "shape": shape, "lastrow_sha": sha,
       "argmax": int(am), "finite": finite,
       "argmax_sha": hashlib.sha256(np.array(am).tobytes()).hexdigest()[:16]}
import tempfile
rowpath = os.path.join(tempfile.gettempdir(), f"pg3_lastrow_{arm}_{T}.npy")
np.save(rowpath, np.array(logits[0, -1, :].view(mx.uint16)))
out["rowpath"] = rowpath

# head-level isolation: M=1 vs M=T row of the quantized head
cache = make_prompt_cache(model)
lm = getattr(model, "language_model", model)   # qwen3_5 wraps TextModel
hidden = lm.model(x, cache=cache)              # inner transformer, [1,T,H]
mx.eval(hidden)
y_m1 = lm.model.embed_tokens.as_linear(hidden[:, -1:, :])   # [1,1,V]
y_mt = lm.model.embed_tokens.as_linear(hidden)[:, -1, :]    # [V]
s1 = hashlib.sha256(np.array(y_m1.view(mx.uint16)).tobytes()).hexdigest()[:16]
st = hashlib.sha256(np.array(y_mt.view(mx.uint16)).tobytes()).hexdigest()[:16]
out["head_m1_sha"] = s1
out["head_mt_sha"] = st
out["head_bitwise_equal"] = bool(np.array_equal(np.array(y_m1[0, 0].view(mx.uint16)),
                                                np.array(y_mt.view(mx.uint16))))
out["head_finite"] = bool(mx.all(mx.isfinite(y_m1)) and mx.all(mx.isfinite(y_mt)))
print(json.dumps(out))
"""

def run(arm, T, extra_env=None):
    env = dict(os.environ)
    env.pop("MLX_OMARCHY_FULL_LOGITS", None)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([sys.executable, "-c", CHILD, model_dir, arm, str(T)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return {"arm": arm, "T": T, "error": r.stderr.strip()[-400:]}
    return json.loads(r.stdout.strip().splitlines()[-1])

summary = {"model": model_dir}
ok = True
for T in Ts:
    full = run("FULL", T, {"MLX_OMARCHY_FULL_LOGITS": "1"})
    last = run("LAST", T)
    last2 = run("LAST", T)
    row = {
        "full": full, "last": last, "last2": last2,
        "shape_full": full.get("shape"), "shape_last": last.get("shape"),
        "lastrow_bitwise_equal": full.get("lastrow_sha") == last.get("lastrow_sha"),
        "argmax_equal": full.get("argmax") == last.get("argmax"),
        "both_finite": bool(full.get("finite") and last.get("finite")),
        "head_bitwise_equal": last.get("head_bitwise_equal"),
        "head_finite": last.get("head_finite"),
        "deterministic": last.get("lastrow_sha") == last2.get("lastrow_sha"),
    }
    # Contract stats (kernel-flags.md class: reduction-order rounding allowed
    # while no argmax flip is possible): max |full-m1| in bf16 quanta of the
    # reference argmax logit, and the reference top1-top2 margin.
    try:
        import numpy as _np
        fr = _np.fromfile(full.get("rowpath", ""), dtype="<u2").astype(_np.uint32) << 16
        lr = _np.fromfile(last.get("rowpath", ""), dtype="<u2").astype(_np.uint32) << 16
        f32 = fr.astype(_np.uint32).view(_np.float32) if False else fr.view(_np.float32)
        l32 = lr.view(_np.float32)
        d = _np.abs(f32 - l32)
        am = int(full.get("argmax"))
        ref = float(f32[am])
        exp = _np.frexp(ref)[1] if ref != 0 else 0
        quantum = _np.ldexp(1.0, exp - 8)  # bf16 mantissa: 8 stored bits
        quanta = float(d.max() / quantum) if quantum > 0 else float("inf")
        srt = _np.sort(f32)[-2:]
        margin = float(srt[1] - srt[0])
        row["max_abs_delta"] = float(d.max())
        row["delta_quanta_at_ref_precision"] = quanta
        row["top1_top2_margin"] = margin
        row["no_flip_possible"] = bool(d[am] < margin)
    except Exception as e:  # noqa: BLE001 - stats are advisory, rows may be missing
        row["contract_stats_error"] = repr(e)[:200]
    contract_ok = row.get("no_flip_possible") or (
        row.get("delta_quanta_at_ref_precision", 1e9) <= 2.0)
    passed = (row["argmax_equal"] and row["both_finite"]
              and row["head_finite"] and row["deterministic"]
              and contract_ok
              and full.get("shape", [0])[:2] == [1, T]
              and last.get("shape", [0])[:2] == [1, 1]
              and full.get("shape", [0, 0, 0])[2] == last.get("shape", [0, 0, 0])[2])
    row["PASS"] = bool(passed)
    ok = ok and bool(passed)
    summary[f"T{T}"] = row
    print(f"T={T} full={full.get('lastrow_sha')} last={last.get('lastrow_sha')} "
          f"bitwise={row['lastrow_bitwise_equal']} argmax={full.get('argmax')}=={last.get('argmax')} "
          f"max|d|={row.get('max_abs_delta')} quanta={row.get('delta_quanta_at_ref_precision')} "
          f"margin={row.get('top1_top2_margin')} no_flip={row.get('no_flip_possible')} "
          f"det={row['deterministic']} PASS={row['PASS']}", flush=True)
print(json.dumps(summary, indent=1))
sys.exit(0 if ok else 4)
