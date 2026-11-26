"""One model, one process: decode cell + repeated pure-prefill cell (qwen38 protocol: greedy, temp 0, same prompts file).
Decode: prompts[:limit] x passes, TTFT excluded, decode_tok_rate = (generated-1)/decode_s per sample (n = limit*passes).
Prefill: one untimed prefill of the same shape, then PF_REPS timed prefills (fresh cache, no token generated) of exactly PF_TOKENS tokens.
Works unchanged on Linux (mlx_omarchy wheel) and macOS (upstream mlx)."""
import argparse, hashlib, json, os, platform, statistics, sys, time

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--prompts", required=True)
p.add_argument("--label", default="")
p.add_argument("--out", required=True)
p.add_argument("--decode-new", type=int, default=64)
p.add_argument("--decode-limit", type=int, default=5)
p.add_argument("--decode-passes", type=int, default=2)
p.add_argument("--pf-tokens", type=int, default=512)
p.add_argument("--pf-reps", type=int, default=5)
p.add_argument("--warmup", type=int, default=2)
a = p.parse_args()

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache


def summ(v):
    return {"median": round(statistics.median(v), 3), "mean": round(statistics.fmean(v), 3),
            "stdev": round(statistics.stdev(v), 3) if len(v) > 1 else 0.0,
            "min": round(min(v), 3), "max": round(max(v), 3), "n": len(v)}


def main():
    prompts = [json.loads(l)["text"] for l in open(a.prompts) if l.strip()]
    corpus_sha = hashlib.sha256(open(a.prompts, "rb").read()).hexdigest()
    t_load = time.perf_counter()
    model, tok = load(a.model)
    load_s = time.perf_counter() - t_load
    for _ in range(a.warmup):
        generate(model, tok, prompt=prompts[0], max_tokens=8, verbose=False)
    recs = []
    for ps in range(a.decode_passes):
        for i, text in enumerate(prompts[: a.decode_limit]):
            ids = tok.encode(text)
            cache = make_prompt_cache(model)
            t0 = time.perf_counter()
            it = generate_step(mx.array(ids), model, prompt_cache=cache)
            first = next(it)[0]
            mx.eval(first)
            ttft = time.perf_counter() - t0
            out = [int(first)]
            t1 = time.perf_counter()
            for tk, _ in it:
                out.append(int(tk))
                mx.eval(tk)
                if len(out) >= a.decode_new:
                    break
            dt = time.perf_counter() - t1
            recs.append({"pass": ps, "i": i, "in": ids, "out": out, "ttft_s": ttft,
                         "decode_tok_s": (len(out) - 1) / dt if len(out) > 1 and dt > 0 else 0.0})
    digest = hashlib.sha256(json.dumps([[r["pass"], r["i"], r["in"], r["out"]] for r in recs]).encode()).hexdigest()
    text = " ".join(prompts[: a.decode_limit])
    ids = tok.encode(text)
    while len(ids) < a.pf_tokens:
        ids = ids + tok.encode(" " + text)
    ids = mx.array(ids[: a.pf_tokens])[None]
    mx.eval(model(ids, cache=make_prompt_cache(model)))  # untimed: pipeline/shape warmup
    pf = []
    for _ in range(a.pf_reps):
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        mx.eval(model(ids, cache=cache))
        pf.append(a.pf_tokens / (time.perf_counter() - t0))
    res = {
        "label": a.label, "model": os.path.basename(a.model.rstrip("/")), "load_s": round(load_s, 2),
        "decode_tok_s": summ([r["decode_tok_s"] for r in recs]), "ttft_s": summ([r["ttft_s"] for r in recs]),
        "prefill_tok_s": summ(pf), "prefill_all": [round(x, 2) for x in pf], "pf_tokens": a.pf_tokens,
        "decode_digest": digest, "corpus_sha256": hashlib.sha256(open(a.prompts, "rb").read()).hexdigest(),
        "first_outputs": [r["out"][:16] for r in recs[: a.decode_limit]],
        "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2) if hasattr(mx, "get_peak_memory") else None,
        "env": {"platform": platform.platform(), "python": platform.python_version(), "backend": str(mx.default_device()),
                "load1_end": os.getloadavg()[0]},
    }
    import mlx_lm
    from importlib import metadata as im
    res["env"]["mlx_lm"] = getattr(mlx_lm, "__version__", "?")
    for d in ("mlx-omarchy", "mlx"):
        try:
            res["env"]["mlx_dist"] = d + " " + im.version(d)
            break
        except im.PackageNotFoundError:
            pass
    json.dump(res, open(a.out, "w"), indent=1)
    print("DECODE %.2f tok/s  PREFILL %.2f tok/s  digest %s" % (res["decode_tok_s"]["median"], res["prefill_tok_s"]["median"], digest[:16]))


main()
