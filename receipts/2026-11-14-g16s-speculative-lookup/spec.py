"""Greedy speculative decoding A/B: baseline greedy vs prompt-lookup / draft-model drafting.
Usage: python spec.py <repo> [--k K] [--ngram N] [--draft REPO] [--max 256] [--prompts set]
Output: per prompt baseline tok/s, spec tok/s, acceptance, identical(bool)."""
import os, sys, time, json, argparse, hashlib
os.environ.setdefault("MLX_OMARCHY_NO_MLXLM_WARNING", "1")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import cache as C

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("--k", type=int, default=3)
ap.add_argument("--ngram", type=int, default=3); ap.add_argument("--draft", default=None)
ap.add_argument("--max", type=int, default=256); ap.add_argument("--only", default=None)
ap.add_argument("--reps", type=int, default=1); ap.add_argument("--nobase", action="store_true")
ap.add_argument("--minn", type=int, default=1); ap.add_argument("--adapt", type=float, default=0.0,
    help="draft only while the running hit rate of the first draft token is >= this (0 = always)")
args = ap.parse_args()

CODE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_code.py")).read()
TEXT = ("The river Danube rises in the Black Forest in Germany and flows for about 2,850 kilometres "
        "through ten countries before it empties into the Black Sea in Romania and Ukraine. "
        "Bratislava, Vienna, Budapest and Belgrade all sit on its banks. In the Middle Ages the river "
        "was a trade route for salt, wine and timber, and today it carries barges, cruise ships and "
        "freight between the North Sea and the Black Sea through the Rhine-Main-Danube canal. ") * 3
PROMPTS = {
  "code": "Rewrite the following Python code so that every function has a docstring and type hints. "
          "Return the full code only.\n\n" + CODE,
  "quote": "Repeat the following text exactly, then add one sentence summarizing it.\n\n" + TEXT,
  "chat": "Explain in three sentences why the sky is blue.",
}

model, tok = load(args.repo)
draft = load(args.draft)[0] if args.draft else None

def enc(p):
    m = [{"role": "user", "content": p}]
    try:
        return tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, add_generation_prompt=True)

eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

def prefill(mdl, cache, ids):
    x = mx.array(ids, mx.uint32)
    while x.size > 1:
        n = min(512, x.size - 1)
        mdl(x[:n][None], cache=cache); mx.eval([c.state for c in cache]); x = x[n:]
    return x  # last token, not yet fed

def baseline(ids):
    cache = C.make_prompt_cache(model)
    y = prefill(model, cache, ids)
    out = []; t0 = None
    while len(out) < args.max:
        logits = model(y[None], cache=cache)
        y = mx.argmax(logits[0, -1:], axis=-1).astype(mx.uint32)
        mx.eval(y)
        if t0 is None: t0 = time.perf_counter(); first = True
        t = y.item(); out.append(t)
        if t in eos: break
    dt = time.perf_counter() - t0
    return out, (len(out) - 1) / dt if len(out) > 1 else 0.0, {}

def ngram_draft(hist, k, nmax, nmin=1):
    for n in range(nmax, nmin - 1, -1):
        if len(hist) < n + 1: continue
        pat = hist[-n:]
        # search most recent earlier occurrence
        for i in range(len(hist) - n - 1, -1, -1):
            if hist[i:i + n] == pat:
                d = hist[i + n:i + n + k]
                if d: return d
    return []

def snapshot(cache):
    return [None if c.is_trimmable() else [s for s in c.state] for c in cache]

def restore(cache, snap):
    for c, s in zip(cache, snap):
        if s is not None: c.state = s

def spec(ids):
    cache = C.make_prompt_cache(model)
    trim_ok = C.can_trim_prompt_cache(cache)
    y = prefill(model, cache, ids)
    hist = list(ids)
    dcache = None
    if draft is not None:
        dcache = C.make_prompt_cache(draft); dy = prefill(draft, dcache, ids)
        dtrim = C.can_trim_prompt_cache(dcache)
    out = []; stats = {"passes": 0, "drafted": 0, "accepted": 0, "replays": 0}
    t0 = None; ema = 1.0
    # first token via a normal step so the timer starts as in the baseline
    logits = model(y[None], cache=cache)
    if dcache is not None: draft(y[None], cache=dcache)
    y = mx.argmax(logits[0, -1:], axis=-1).astype(mx.uint32); mx.eval(y)
    t0 = time.perf_counter(); t = y.item(); out.append(t); hist.append(t)
    while len(out) < args.max and t not in eos:
        k = min(args.k, args.max - len(out))
        if draft is not None:
            dsnap = None if dtrim else snapshot(dcache)
            ds = []; dyy = mx.array([t], mx.uint32)
            for _ in range(k):
                dl = draft(dyy[None], cache=dcache)
                dyy = mx.argmax(dl[0, -1:], axis=-1).astype(mx.uint32); ds.append(dyy)
            d = mx.concatenate(ds).tolist()
        else:
            d = ngram_draft(hist, k, args.ngram, args.minn)
            if args.adapt and (not d or ema < args.adapt):
                # plain step; score the draft's first token against it for free
                logits = model(mx.array([t], mx.uint32)[None], cache=cache)
                g0 = mx.argmax(logits[0, -1:], axis=-1).item()
                stats["plain"] = stats.get("plain", 0) + 1
                if d: ema = 0.8 * ema + 0.2 * (d[0] == g0)
                out.append(g0); hist.append(g0); t = g0
                continue
        snap = None if trim_ok else snapshot(cache)
        x = mx.array([t] + d, mx.uint32)
        os.environ["MLX_OMARCHY_FULL_LOGITS"] = "1"  # verify needs every position
        logits = model(x[None], cache=cache)
        os.environ.pop("MLX_OMARCHY_FULL_LOGITS", None)
        g = mx.argmax(logits[0], axis=-1).tolist()
        assert len(g) == len(d) + 1, logits.shape
        stats["passes"] += 1; stats["drafted"] += len(d)
        a = 0
        try:
            while a < len(d) and d[a] == g[a]: a += 1
        except IndexError:
            print("DBG2", a, len(d), len(g), repr(d)[:200], repr(g)[:200], logits.shape, flush=True); raise
        stats["accepted"] += a
        if d: ema = 0.8 * ema + 0.2 * (a > 0)
        new = d[:a] + [g[a]]
        rej = len(d) - a
        if rej:
            if trim_ok:
                C.trim_prompt_cache(cache, rej)
            else:
                # roll back the recurrent state and replay the accepted prefix
                C.trim_prompt_cache([c for c in cache if c.is_trimmable()], len(d) + 1)
                restore(cache, snap)
                model(mx.array([t] + d[:a], mx.uint32)[None], cache=cache)
                stats["replays"] += 1
        if dcache is not None and dtrim:
            # draft cache holds t + d[:k-1]; keep t + d[:a]
            C.trim_prompt_cache(dcache, max(len(d) - 1 - a, 0))
            if a == len(d):  # last draft token not yet seen by the draft model
                draft(mx.array([d[-1]], mx.uint32)[None], cache=dcache)
        elif dcache is not None:
            # recurrent draft: roll back and replay t + d[:a] in one pass
            C.trim_prompt_cache([c for c in dcache if c.is_trimmable()], len(d))
            restore(dcache, dsnap)
            draft(mx.array([t] + d[:a], mx.uint32)[None], cache=dcache)
        for tt in new:
            out.append(tt); hist.append(tt)
            if tt in eos or len(out) >= args.max: break
        t = out[-1]
    mx.eval([c.state for c in cache])
    dt = time.perf_counter() - t0
    return out, (len(out) - 1) / dt, stats

res = []
for name, p in PROMPTS.items():
    if args.only and name not in args.only.split(","): continue
    ids = enc(p)
    for r in range(args.reps):
        # warm both paths once on the first rep
        if r == 0:
            baseline(ids[:64]); spec(ids[:64])
        b = baseline(ids) if not args.nobase else (None, 0, {})
        s = spec(ids)
        same = (b[0] == s[0]) if b[0] is not None else None
        line = {"prompt": name, "prompt_tokens": len(ids), "gen": len(s[0]), "base_tps": round(b[1], 1),
                "spec_tps": round(s[1], 1), "speedup": round(s[1] / b[1], 3) if b[1] else None,
                "identical": same, **s[2],
                "sha": hashlib.sha256(json.dumps(s[0]).encode()).hexdigest()[:12]}
        if same is False:
            i = next(i for i, (u, v) in enumerate(zip(b[0], s[0])) if u != v) if any(u != v for u, v in zip(b[0], s[0])) else min(len(b[0]), len(s[0]))
            line["first_diff"] = i
        print("RESULT", json.dumps(line), flush=True)
        res.append(line)
print("TEXT(last):", tok.decode(s[0])[:200].replace("\n", " "))
