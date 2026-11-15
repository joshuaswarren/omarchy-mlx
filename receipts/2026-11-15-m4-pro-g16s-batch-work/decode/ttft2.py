"""ttft.py <repo> <label>: TTFT after idle gaps, prefill 2k and decode for one model, one process.
TTFT = time from the call to the first streamed token of a short chat prompt (prefill ~30 tok + 1 token)."""
import os, sys, time, statistics, json
os.environ["MLX_OMARCHY_NO_MLXLM_WARNING"] = "1"
import mlx.core as mx
from mlx_lm import load, stream_generate
repo, label = sys.argv[1], sys.argv[2]
gaps = [float(g) for g in os.environ.get("GAPS", "0,0.02,0.1,0.5,2").split(",")]
reps = int(os.environ.get("REPS", "5"))
m, tok = load(repo)
short = tok.apply_chat_template([{"role": "user", "content": "Name three primary colours."}], add_generation_prompt=True)
long2k = tok.apply_chat_template([{"role": "user", "content": "Summarize the following text. " + ("The quick brown fox jumps over the lazy dog near the river bank while the farmer counts his sheep. " * 100)}], add_generation_prompt=True)
p = tok.apply_chat_template([{"role": "user", "content": "Explain in three sentences why the sky is blue."}], add_generation_prompt=True)
for _ in range(2):
    for r in stream_generate(m, tok, short, max_tokens=2): pass
def ttft():
    t = time.perf_counter()
    for r in stream_generate(m, tok, short, max_tokens=1): pass
    return (time.perf_counter() - t) * 1000
res = {"label": label, "repo": repo}
for g in gaps:
    xs = []
    for _ in range(reps):
        time.sleep(g); xs.append(ttft())
    res[f"ttft_ms_gap{g}"] = round(statistics.median(xs), 2)
pf = []
for _ in range(2):
    for r in stream_generate(m, tok, long2k, max_tokens=1): pw = r
    pf.append(pw.prompt_tps)
res["pf2k_tps"] = round(max(pf), 1); res["pf2k_tok"] = pw.prompt_tokens
dec = []
for _ in range(2):
    for r in stream_generate(m, tok, p, max_tokens=128): last = r
    dec.append(last.generation_tps)
res["decode_tps"] = round(max(dec), 2)
res["peak_mem_gb"] = round(mx.get_peak_memory() / 1e9, 2)
print("JSON", json.dumps(res), flush=True)
