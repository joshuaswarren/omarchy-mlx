#!/usr/bin/env python3
"""Teacher-forced harness for docs/numerics-gate.md (checks 2 and 3, harness check).

Built on receipts/2026-10-28-qmm-small-t-g13s/qwen38_chunked_gate.py; records use the
tools/dfuse/compare_tf9.py format (ten prompts x 512 measurements).

Text: the GPL-3 licence text kept in the tree (tfref-text.txt, sha256 3972dc97...), as in
receipts/2026-10-21-gdn-bf16a-g14c-ratio1. It is tokenized with each model's own tokenizer
(no special tokens) and cut into ten non-overlapping 513-token segments, evenly spaced from
token 0 to the end. Each segment is one prompt: the model's BOS token when its tokenizer has
one, then the segment; the 512 positions after the first segment token are scored, so every
model gets 5,120 teacher-forced positions. Every segment starts from a fresh cache.

  gate_tf.py text     --model DIR --text FILE --out teacher.json
  gate_tf.py measure  --model DIR --teacher teacher.json --chunk N [--cpu] --out FILE | --outdir DIR --label L
  gate_tf.py validate --model DIR --teacher teacher.json --out validate.json
  gate_tf.py merge    --model DIR --parts a.json b.json ... --out ref.json
GPU processes import prov.py and refuse (exit 3) unless the loaded wheel's RECORD hashes
match and its stamp equals EXPECT_COMMIT. --cpu is the reference: stock upstream MLX on the
CPU stream with every floating parameter in float32 (the CPU quantized matmul dequantizes
in fp32), and it refuses to run if an omarchy distribution is installed.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import math
import os
import sys
import time

# Teacher forcing needs logits at every position.
os.environ.setdefault("MLX_OMARCHY_FULL_LOGITS", "1")

SEGMENTS, SEG_LEN = 10, 513
ROUTE_ENV = ("MLX_OMARCHY_FUSED_CHAIN", "MLX_OMARCHY_FUSED_GEMV", "MLX_OMARCHY_FUSED_TRIO",
             "MLX_OMARCHY_NO_MATMUL_FMA", "MLX_OMARCHY_NO_QMM_FMA_BF16", "MLX_OMARCHY_NO_COOPMAT",
             "HK_APPLE9_COOPMAT", "AGX_APPLE9_MAX_SHARED", "MLX_OMARCHY_BATCH_WORK")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ids_sha(prompts):
    return hashlib.sha256(json.dumps([[p["prompt_ids"], p["tokens"]] for p in prompts]).encode()).hexdigest()


def identity(cpu):
    if cpu:
        names = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
        if any("omarchy" in n for n in names):
            sys.exit("REFUSED: the reference venv must not contain an omarchy distribution")
        return {"reference": "stock upstream MLX, CPU stream, float32",
                "mlx": names.get("mlx"), "mlx_cpu": names.get("mlx-cpu"), "mlx_lm": names.get("mlx-lm")}
    import prov  # noqa: E402  (probes dir)
    p = prov.check()
    return {"provenance": prov.mlx_provenance.provenance_line(p), "mlx_lm": md.version("mlx-lm")}


def text(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    raw = open(a.text, encoding="utf-8").read()
    ids = tok.encode(raw, add_special_tokens=False)
    n = len(ids)
    step = (n - SEG_LEN) // (SEGMENTS - 1)
    if step < SEG_LEN:
        sys.exit(f"text too short for {SEGMENTS} non-overlapping segments: {n} tokens")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    prompts = []
    for i in range(SEGMENTS):
        s = ids[i * step:i * step + SEG_LEN]
        prompts.append({"prompt_index": i, "start": i * step, "prompt_ids": bos + s[:1], "tokens": s[1:]})
    parts = os.path.abspath(a.model).split(os.sep)
    repo = next((p[len("models--"):].replace("--", "/") for p in parts if p.startswith("models--")), parts[-1])
    out = {"model": repo, "snapshot": parts[-1], "text_sha256": sha256_file(a.text),
           "text_tokens": n, "segment_step": step, "bos": bos, "prompt_sha256": ids_sha(prompts),
           "prompts": prompts}
    json.dump(out, open(a.out, "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "prompts"}))


def load_model(path, cpu):
    import mlx.core as mx
    from mlx_lm import load
    if cpu:
        mx.set_default_device(mx.cpu)
    model, _ = load(path)
    if cpu:
        model.set_dtype(mx.float32)
    return model


def measure_rows(model, ids, toks, chunk):
    """Teacher-forced pass over ids + toks in fixed chunks; one record per teacher token."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    seq = ids + toks[:-1]
    want = range(len(ids) - 1, len(ids) - 1 + len(toks))
    cache = make_prompt_cache(model)
    pred, gaps, mags, chosen = [], [], [], []
    for start in range(0, len(seq), chunk):
        part = seq[start:start + chunk]
        logits = model(mx.array(part)[None], cache=cache)[0].astype(mx.float32)
        if logits.shape[0] != len(part):
            sys.exit(f"REFUSED: model returned {logits.shape[0]} logit rows for a {len(part)}-token pass")
        lo, hi = max(start, want.start), min(start + len(part), want.stop)
        if lo >= hi:
            mx.eval(logits)
            continue
        rows = logits[lo - start:hi - start]
        # Top two without mx.topk: the M4 driver rejects the partition pipeline at
        # vocabulary width. The largest value, its index, and the largest value with that
        # one index masked out give the same two numbers (a tie gives a gap of 0).
        am = mx.argmax(rows, axis=-1)
        m1 = mx.max(rows, axis=-1)
        m2 = mx.max(mx.where(mx.arange(rows.shape[-1])[None, :] == am[:, None], -mx.inf, rows), axis=-1)
        lse = mx.logsumexp(rows, axis=-1)
        teach = mx.array(toks[lo - want.start:hi - want.start])
        lp = mx.take_along_axis(rows, teach[:, None], axis=-1)[:, 0] - lse
        mx.eval(am, m1, m2, lp)
        v1, v2 = m1.tolist(), m2.tolist()
        pred += am.tolist()
        gaps += [x - y for x, y in zip(v1, v2)]
        mags += [abs(x) for x in v1]
        chosen += lp.tolist()
    if not all(math.isfinite(v) for v in chosen + gaps):
        sys.exit("REFUSED: non-finite logits")
    return {"predicted": pred, "top2_gaps": gaps, "top2_magnitudes": mags, "chosen_logprobs": chosen}


def measure(a):
    ident = identity(a.cpu)
    model = load_model(a.model, a.cpu)
    tch = json.load(open(a.teacher, encoding="utf-8"))
    t0 = time.time()
    prompts = []
    only = {int(i) for i in a.only.split(",")} if a.only else None
    for p in tch["prompts"]:
        if only is not None and p["prompt_index"] not in only:
            continue
        row = measure_rows(model, p["prompt_ids"], p["tokens"], a.chunk)
        prompts.append({"prompt_index": p["prompt_index"], "tokens": p["tokens"], **row})
    lps = [v for p in prompts for v in p["chosen_logprobs"]]
    rec = {"prompt_sha256": tch["prompt_sha256"], "model": tch["model"], "chunk": a.chunk,
           "device": "cpu" if a.cpu else "gpu", "commit": os.environ.get("EXPECT_COMMIT", ""),
           "env": {k: os.environ[k] for k in ROUTE_ENV if k in os.environ},
           "seconds": round(time.time() - t0, 1), **ident, "prompts": prompts}
    out = a.out or os.path.join(a.outdir, f"{a.label}-{rec['commit'] or 'ref'}-{time.time_ns()}.json")
    json.dump(rec, open(out, "w"))
    print(json.dumps({"out": os.path.basename(out), "positions": len(lps),
                      "ppl": math.exp(-sum(lps) / len(lps)), "seconds": rec["seconds"]}), flush=True)


def merge(a):
    """Joins reference parts (one process per segment group) into one record."""
    parts = [json.load(open(p, encoding="utf-8")) for p in a.parts]
    out = dict(parts[0])
    assert all(p["prompt_sha256"] == out["prompt_sha256"] for p in parts)
    out["prompts"] = sorted((p for d in parts for p in d["prompts"]), key=lambda p: p["prompt_index"])
    out["seconds"] = max(d["seconds"] for d in parts)
    out["parts"] = len(parts)
    assert [p["prompt_index"] for p in out["prompts"]] == list(range(SEGMENTS))
    json.dump(out, open(a.out, "w"))
    lps = [v for p in out["prompts"] for v in p["chosen_logprobs"]]
    print(json.dumps({"out": os.path.basename(a.out), "positions": len(lps), "ppl": math.exp(-sum(lps) / len(lps))}))


def validate(a):
    """numerics-gate.md harness check: S=1 teacher-forced PPL equals one-pass prefill PPL."""
    ident = identity(False)
    model = load_model(a.model, False)
    tch = json.load(open(a.teacher, encoding="utf-8"))
    res = {"model": tch["model"], "commit": os.environ.get("EXPECT_COMMIT", ""), **ident,
           "env": {k: os.environ[k] for k in ROUTE_ENV if k in os.environ}, "segments": [0, 1]}
    for name, chunk in (("s1", 1), ("one_pass", 100000)):
        lps = []
        for p in tch["prompts"][:2]:
            lps += measure_rows(model, p["prompt_ids"], p["tokens"], chunk)["chosen_logprobs"]
        res[f"{name}_positions"] = len(lps)
        res[f"{name}_ppl"] = math.exp(-sum(lps) / len(lps))
    res["rel_diff"] = abs(res["s1_ppl"] / res["one_pass_ppl"] - 1)
    res["threshold"] = 1e-2
    res["passed"] = res["rel_diff"] <= res["threshold"]
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("model", "s1_ppl", "one_pass_ppl", "rel_diff", "passed")}), flush=True)
    if not res["passed"]:
        sys.exit(5)


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("text", "measure", "validate", "merge"))
ap.add_argument("--model", required=True)
ap.add_argument("--text")
ap.add_argument("--teacher")
ap.add_argument("--chunk", type=int, default=1)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--out")
ap.add_argument("--outdir")
ap.add_argument("--label", default="arm")
ap.add_argument("--only", default="", help="measure: comma-separated segment indices (parallel reference parts)")
ap.add_argument("--parts", nargs="*", default=[], help="merge: part files")
a = ap.parse_args()
{"text": text, "measure": measure, "validate": validate, "merge": merge}[a.mode](a)
