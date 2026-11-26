#!/usr/bin/env python3
"""Teacher-forced numerics gate for MLX_OMARCHY_QMV_SMALL_T on the parity-contract 2B model.

docs/parity-contract.md pins SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b; docs/numerics-gate.md
judges a route change against an independent reference (an fp32 CPU forward of the same weights
and text), not against the kernel it replaces. The prompts are the ten contract prompts, taken as
token ids from receipts-work/vocab-prune/out/contract-ctl-r1.json (pass 0).

The short-pass route only runs on passes of 2..T rows, and a one-pass prefill never has one, so
every arm reads the teacher-forced text in fixed chunks (--chunk). Records use the
tools/dfuse/compare_tf9.py format: ten prompts x 512 measurements.

  qwen38_chunked_gate.py teacher   --model DIR --record JSON --out teacher.json
  qwen38_chunked_gate.py measure   --model DIR --teacher teacher.json --chunk 4 [--cpu] --out arm.json
  qwen38_chunked_gate.py validate  --model DIR --teacher teacher.json --out validate.json
  qwen38_chunked_gate.py merge     --model DIR --parts a.json b.json ... --out ref.json
The route is selected by MLX_OMARCHY_QMV_SMALL_T in the environment of each process.
"""
import argparse
import hashlib
import json
import math
import os
import time

# Teacher forcing needs logits at every position; omarchy's mlx-lm last-logits patch keeps only
# the last one on cached passes unless this is set (models/qwen3_5.py).
os.environ["MLX_OMARCHY_FULL_LOGITS"] = "1"

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

NEW = int(os.environ.get("GATE_NEW_TOKENS", "512"))  # smoke tests only; the gate is 512
LIMIT = int(os.environ.get("GATE_PROMPTS", "10"))


def contract_prompts(record_path):
    rec = json.load(open(record_path, encoding="utf-8"))
    rows = {int(r["prompt_idx"]): (r["input_ids"] if isinstance(r["input_ids"], list) else json.loads(r["input_ids"]))
            for r in rec["per_prompt"] if str(r["pass"]) == "0"}
    assert sorted(rows) == list(range(10)), sorted(rows)
    return [rows[i] for i in range(10)], rec["protocol"].get("prompts_corpus_sha256")


def ids_sha(prompts):
    return hashlib.sha256(json.dumps(prompts).encode()).hexdigest()


def teacher(a):
    model, _ = load(a.model)
    prompts, corpus = contract_prompts(a.record)
    out = []
    for i, ids in enumerate(prompts[:LIMIT]):
        cache = make_prompt_cache(model)
        logits = model(mx.array(ids)[None], cache=cache)
        toks = []
        for _ in range(NEW):
            t = mx.argmax(logits[0, -1]).item()
            toks.append(t)
            logits = model(mx.array([[t]]), cache=cache)
        out.append({"prompt_index": i, "prompt_ids": ids, "tokens": toks})
    json.dump({"record": os.path.abspath(a.record), "record_sha256": sha256(a.record),
               "contract_corpus_sha256": corpus, "prompt_sha256": ids_sha(prompts),
               "small_t": os.environ.get("MLX_OMARCHY_QMV_SMALL_T", ""), "prompts": out},
              open(a.out, "w"))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def measure_rows(model, ids, toks, chunk):
    """Teacher-forced pass over ids + toks in fixed chunks; one record per teacher token."""
    seq = ids + toks[:-1]
    want = range(len(ids) - 1, len(ids) - 1 + len(toks))
    cache = make_prompt_cache(model)
    pred, gaps, mags, chosen = [], [], [], []
    for start in range(0, len(seq), chunk):
        part = seq[start:start + chunk]
        logits = model(mx.array(part)[None], cache=cache)[0].astype(mx.float32)
        lo = max(start, want.start)
        hi = min(start + len(part), want.stop)
        if lo >= hi:
            mx.eval(logits)
            continue
        rows = logits[lo - start:hi - start]
        top2 = mx.topk(rows, 2, axis=-1)  # the two largest, in no promised order
        lse = mx.logsumexp(rows, axis=-1)
        teach = mx.array(toks[lo - want.start:hi - want.start])
        lp = mx.take_along_axis(rows, teach[:, None], axis=-1)[:, 0] - lse
        am = mx.argmax(rows, axis=-1)
        mx.eval(top2, lp, am)
        t2 = top2.tolist()
        pred += am.tolist()
        gaps += [max(r) - min(r) for r in t2]
        mags += [abs(max(r)) for r in t2]
        chosen += lp.tolist()
    return {"predicted": pred, "top2_gaps": gaps, "top2_magnitudes": mags, "chosen_logprobs": chosen}


def measure(a):
    if a.cpu:
        mx.set_default_device(mx.cpu)
    model, _ = load(a.model)
    if a.cpu:
        # numerics-gate.md reference: an fp32 CPU forward of the same weights, run with stock
        # upstream MLX (the parity contract's reference version). The CPU quantized matmul
        # dequantizes in fp32; --dequantize expands the weights up front instead (~7 GB).
        if a.dequantize:
            from mlx_lm.utils import dequantize_model
            model = dequantize_model(model)
        model.set_dtype(mx.float32)
    tch = json.load(open(a.teacher, encoding="utf-8"))
    t0 = time.time()
    prompts = []
    only = {int(i) for i in a.only.split(",")} if a.only else None
    for p in tch["prompts"]:
        if only is not None and p["prompt_index"] not in only:
            continue
        row = measure_rows(model, p["prompt_ids"], p["tokens"], a.chunk)
        prompts.append({"prompt_index": p["prompt_index"], "tokens": p["tokens"], **row})
    json.dump({"prompt_sha256": tch["prompt_sha256"], "chunk": a.chunk, "device": "cpu" if a.cpu else "gpu",
               "small_t": os.environ.get("MLX_OMARCHY_QMV_SMALL_T", ""), "seconds": round(time.time() - t0, 1),
               "mlx": mx.__version__, "prompts": prompts}, open(a.out, "w"))


def validate(a):
    """numerics-gate.md harness check: S=1 teacher-forced PPL equals one-pass prefill PPL."""
    model, _ = load(a.model)
    tch = json.load(open(a.teacher, encoding="utf-8"))
    res = {}
    for name, chunk in (("s1", 1), ("one_pass", 100000)):
        lps = []
        for p in tch["prompts"][:2]:
            lps += measure_rows(model, p["prompt_ids"], p["tokens"], chunk)["chosen_logprobs"]
        res[name] = math.exp(-sum(lps) / len(lps))
    res["rel_diff"] = abs(res["s1"] / res["one_pass"] - 1)
    json.dump(res, open(a.out, "w"))
    print(json.dumps(res))


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("teacher", "measure", "validate", "merge"))
ap.add_argument("--model", required=True)
ap.add_argument("--record")
ap.add_argument("--teacher")
ap.add_argument("--chunk", type=int, default=4)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--dequantize", action="store_true")
ap.add_argument("--only", default="", help="comma-separated prompt indices (parallel CPU parts)")
ap.add_argument("--parts", nargs="*", default=[], help="merge: part files")
ap.add_argument("--out", required=True)
a = ap.parse_args()
def merge(a):
    parts = [json.load(open(p, encoding="utf-8")) for p in a.parts]
    out = dict(parts[0])
    out["prompts"] = sorted((p for d in parts for p in d["prompts"]), key=lambda p: p["prompt_index"])
    out["seconds"] = max(d["seconds"] for d in parts)
    assert [p["prompt_index"] for p in out["prompts"]] == list(range(10))
    json.dump(out, open(a.out, "w"))


{"teacher": teacher, "measure": measure, "validate": validate, "merge": merge}[a.mode](a)
