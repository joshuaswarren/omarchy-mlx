"""docs/numerics-gate.md capture for one model and one arm: teacher-forced decode, one token per step, over a long text.

usage: numerics_gate.py MODEL ARM OUT.npz EXPECT_WHEEL REPO_ROOT TEXT_FILE

ARM is bf16 (the model as shipped, the installed wheel's default routes, every MLX_OMARCHY_SDPA variable unset), fp32
(model.set_dtype(float32) on the omarchy wheel: every activation and attention in float32 on the GPU) or cpuref (the
docs/numerics-gate.md reference: stock upstream MLX on the CPU stream, every floating parameter in float32, in a venv with no
omarchy package, built by ref_setup.sh; EXPECT_WHEEL is then ignored and the stock mlx version is recorded instead). The first PREFIX tokens of TEXT_FILE are
prefilled; then each of the next STEPS tokens is fed one at a time on the prompt cache, so step i attends PREFIX + i + 1 keys and
the decode attention routes run at every cache length from PREFIX + 1 to PREFIX + STEPS. Per scored step the npz stores the
argmax id, the top-1 and top-2 logits, and the log-prob of the next text token (the target). With --validate (bf16 only) the
same text is also scored by chunked prefill, and the S=1 teacher-forced perplexity must match the prefill perplexity
(numerics-gate.md harness validation). Prints the scripts/mlx_provenance.py line and refuses (exit 3) unless the installed wheel
matches EXPECT_WHEEL.
"""
import json
import math
import os
import sys
from pathlib import Path

mdl, arm, out_npz, expect_wheel, repo_root, text_file = sys.argv[1:7]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp  # noqa: E402

if arm == "cpuref":
    import importlib.metadata as md
    assert not any("omarchy" in d.metadata["Name"].lower() for d in md.distributions()), "the reference venv holds an omarchy package"
    pline = "reference: stock mlx " + md.version("mlx") + " mlx-cpu " + md.version("mlx-cpu") + " mlx-lm " + md.version("mlx-lm") + " (CPU stream, float32)"
else:
    prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
    pline = mp.provenance_line(prov)
    if prov["verified"] != "match" or prov["version_match"] is not True:
        print(pline, flush=True)
        sys.exit(3)
print(pline, flush=True)
assert not any(k.startswith("MLX_OMARCHY_SDPA") or k == "MLX_OMARCHY_FULL_LOGITS" for k in os.environ), "the SDPA and full-logits variables must be unset"

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

PREFIX = int(os.environ.get("GATE_PREFIX", "128"))  # a later segment of the same scored range starts after a longer chunked prefill
STEPS = int(os.environ.get("GATE_STEPS", "5120"))
model, tok = load(mdl)
if arm in ("fp32", "cpuref"):
    if arm == "cpuref":
        mx.set_default_device(mx.cpu)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
else:
    assert arm == "bf16", arm
ids = tok.encode(open(text_file).read())
assert len(ids) >= PREFIX + STEPS + 1, len(ids)
cache = make_prompt_cache(model)
for c0 in range(0, PREFIX, 512):
    logits = model(mx.array(ids[c0:min(c0 + 512, PREFIX)])[None], cache=cache)
    mx.eval(logits)
argmax, top1, top2, target_lp = [], [], [], []
for i in range(STEPS):
    row = logits[0, -1].astype(mx.float32)
    lse = mx.logsumexp(row)
    top = mx.topk(row, 2)
    am = mx.argmax(row)
    t = ids[PREFIX + i]
    tl = row[t] - lse
    mx.eval(top, am, tl)
    top = np.sort(np.array(top))[::-1]
    argmax.append(int(am)); top1.append(float(top[0])); top2.append(float(top[1])); target_lp.append(float(tl))
    logits = model(mx.array([[t]]), cache=cache)
    if (i + 1) % 512 == 0:
        print(f"step {i + 1}/{STEPS}", file=sys.stderr, flush=True)
mx.eval(logits)
res = {"argmax": np.array(argmax, np.int64), "top1": np.array(top1), "top2": np.array(top2), "target_lp": np.array(target_lp),
       "target": np.array(ids[PREFIX:PREFIX + STEPS], np.int64), "keys": PREFIX + 1 + np.arange(STEPS)}
ppl_tf = math.exp(-float(np.mean(target_lp)))
info = {"arm": arm, "model": Path(mdl).name, "prefix": PREFIX, "steps": STEPS, "ppl_teacher_forced": ppl_tf, "provenance": pline}
if "--validate" in sys.argv:
    del cache
    mx.clear_cache()
    cache = make_prompt_cache(model)
    lps = []
    os.environ["MLX_OMARCHY_FULL_LOGITS"] = "1"  # the packaged last-logits patch keeps one row on cached calls unless this is set
    for s in range(0, PREFIX + STEPS, 512):
        chunk = ids[s:min(s + 512, PREFIX + STEPS)]
        lg = model(mx.array(chunk)[None], cache=cache)[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        nxt = mx.array(ids[s + 1:s + 1 + len(chunk)])
        g = mx.take_along_axis(lp[: nxt.shape[0]], nxt[:, None], axis=-1)[:, 0]
        mx.eval(g)
        lps.extend(np.array(g).tolist())
    del os.environ["MLX_OMARCHY_FULL_LOGITS"]
    # positions PREFIX-1 .. PREFIX+STEPS-2 predict the scored targets
    pre = np.array(lps[PREFIX - 1:PREFIX - 1 + STEPS])
    info["ppl_prefill_same_targets"] = math.exp(-float(pre.mean()))
    res["prefill_target_lp"] = pre
np.savez_compressed(out_npz, **res)
json.dump(info, open(out_npz[:-4] + ".json", "w"), indent=1)
print("GATE " + json.dumps(info), flush=True)
