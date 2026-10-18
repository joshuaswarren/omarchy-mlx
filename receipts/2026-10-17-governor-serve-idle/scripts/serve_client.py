#!/usr/bin/env python3
"""Decode rate of a local mlx_lm server from two non-streamed requests; prints ONE JSON line.

usage: serve_client.py PORT TOKENS PROMPT
TOKENS <= 16: one request (a warm-up), prints its token count and wall time.
Otherwise two requests with the same prompt, temperature 0: a short one (TOKENS // 4) and a long one (TOKENS). Both must end with
finish_reason "length" and report completion_tokens equal to max_tokens in the usage block. decode_tps is
(n_long - n_short) / (t_long - t_short) over the wall times of the whole requests, so the prompt processing, the request overhead and the
way the server chunks its stream all cancel. text_sha16 hashes the message of the long request (content and any reasoning fields).
Exit 1 when a request fails or a count differs.
"""
import hashlib
import json
import sys
import time
import urllib.request

port, tokens, prompt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def ask(n):
    body = json.dumps({"model": "default_model", "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"content-type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    wall = time.monotonic() - t0
    choice = data["choices"][0]
    return {"n": data["usage"]["completion_tokens"], "finish": choice.get("finish_reason"), "wall": wall,
            "message": json.dumps(choice.get("message"), sort_keys=True)}


try:
    if tokens <= 16:
        r = ask(tokens)
        out = {"n": r["n"], "finish": r["finish"], "wall_s": round(r["wall"], 4)}
        ok = r["finish"] == "length" and r["n"] == tokens
    else:
        short, long_ = ask(tokens // 4), ask(tokens)
        ok = (short["finish"] == "length" and long_["finish"] == "length" and short["n"] == tokens // 4 and long_["n"] == tokens
              and long_["wall"] > short["wall"])
        out = {"n_short": short["n"], "n_long": long_["n"], "wall_short_s": round(short["wall"], 4), "wall_long_s": round(long_["wall"], 4),
               "finish": long_["finish"], "text_sha16": hashlib.sha256(long_["message"].encode()).hexdigest()[:16]}
        if long_["wall"] > short["wall"]:
            out["decode_tps"] = round((long_["n"] - short["n"]) / (long_["wall"] - short["wall"]), 4)
except Exception as exc:  # noqa: BLE001 - the one line says why
    print(json.dumps({"error": repr(exc)[:300]}))
    sys.exit(1)
print(json.dumps(out, sort_keys=True))
sys.exit(0 if ok else 1)
