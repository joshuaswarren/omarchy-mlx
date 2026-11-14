# end-to-end: mlx_lm.stream_generate greedy, plain vs MLX_OMARCHY_PROMPT_LOOKUP=K (patched tree in ./pb)
import os, sys, time, json, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pb"))
import mlx.core as mx
import mlx_lm; from mlx_lm import load, stream_generate
assert mlx_lm.__file__.startswith(sys.path[0]), mlx_lm.__file__
repo = sys.argv[1]; K = sys.argv[2]; maxt = int(sys.argv[3]) if len(sys.argv) > 3 else 256
draft = sys.argv[4] if len(sys.argv) > 4 else None
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.py")).read())
model, tok = load(repo); dm = load(draft)[0] if draft else None
def run(p, lookup, use_draft=False):
    if lookup: os.environ["MLX_OMARCHY_PROMPT_LOOKUP"] = K
    else: os.environ.pop("MLX_OMARCHY_PROMPT_LOOKUP", None)
    out = []; last = None
    kw = {"draft_model": dm, "num_draft_tokens": int(K)} if use_draft else {}
    for r in stream_generate(model, tok, p, max_tokens=maxt, **kw):
        out.append(r.token); last = r
    return out, last.generation_tps
for name, msg in PROMPTS.items():
    p = enc(tok, msg)
    run(p[:64] if isinstance(p, list) else p, False); run(p, not draft, bool(draft))
    b, bt = run(p, False); s, st = run(p, not draft, bool(draft))
    print("E2E", json.dumps({"prompt": name, "base_tps": round(bt, 1), "spec_tps": round(st, 1), "speedup": round(st / bt, 3),
          "identical": b == s, "gen": len(s), "sha": hashlib.sha256(json.dumps(s).encode()).hexdigest()[:12]}), flush=True)
