import os, sys, time, json, hashlib
import mlx, mlx.core as mx
import mlx_lm; from mlx_lm import load, stream_generate
HERE=os.path.dirname(os.path.abspath(__file__))
repo=sys.argv[1]; draft=sys.argv[2]; K=int(sys.argv[3]); maxt=int(sys.argv[4])
only=sys.argv[5].split(",") if len(sys.argv)>5 else None
exec(open(os.path.join(HERE,"prompts.py")).read())
print("MLX",mlx.__file__,flush=True); print("MLXLM",mlx_lm.__file__,flush=True)
model,tok=load(repo); dm=load(draft)[0]
def run(p,use_draft,adapt):
    if use_draft: os.environ["MLX_OMARCHY_SPEC_ADAPT"]="1" if adapt else "0"
    out=[]; last=None
    kw={"draft_model":dm,"num_draft_tokens":K} if use_draft else {}
    for r in stream_generate(model,tok,p,max_tokens=maxt,**kw):
        out.append(r.token); last=r
    return out,(last.generation_tps if last else 0.0)
for name,msg in PROMPTS.items():
    if only and name not in only: continue
    p=enc(tok,msg)
    run(p,False,False)                    # warmup
    b,bt=run(p,False,False)               # plain greedy baseline
    f,ft=run(p,True,False)                # fixed-k spec
    a,at=run(p,True,True)                 # adaptive spec
    print("E2E",json.dumps({"prompt":name,"base_tps":round(bt,1),
      "fixed_tps":round(ft,1),"fixed_su":round(ft/bt,3),"fixed_id":b==f,
      "adapt_tps":round(at,1),"adapt_su":round(at/bt,3),"adapt_id":b==a,
      "gen":len(a),"sha":hashlib.sha256(json.dumps(a).encode()).hexdigest()[:12]}),flush=True)
