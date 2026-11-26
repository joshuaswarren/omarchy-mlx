#!/usr/bin/env python3
"""servegen_inproc.py ARM MODEL_DIR PAIRS TOKENS PROMPT: the mlx_lm.server generator driven in-process, no HTTP, no sockets.

ARM G  stock mlx_lm.server stack (ResponseGenerator + BatchGenerator, decode and prompt concurrency 1, prompt cache 0) with the
       serve package's uclamp placement on the calling thread (the same call the shim makes).
ARM S  the same stack through the repo serve shim (mlx_omarchy_serve._mlxlm_server.main): context cap, speech yield gate, TTFT wrap,
       uclamp placement. ONLY the HTTP server is replaced: server_module._run_http_server is swapped for the driver below, so the shim
       installs every hook exactly as in the real server and then calls this driver where it would bind the port.
ARMS GM, GL, GS  arm G with ONE candidate cost removed: GM no mask array for the sliding-window batch cache when it would be all True,
       GL no per-token logprob read in the server loop (a stub replaces r.logprobs), GS the new thread-local stream bench_decode's path
       uses instead of the default stream. See install_flip(); the ids digest must stay equal to arm G's.
The driver sends one warm-up request (8 tokens; the model loads on the first request) and then PAIRS request pairs: a TOKENS//4 token
request and a TOKENS token request, same prompt, greedy. Per pair it prints ONE JSON line:
  ts_tps    (n - 1) / (time of the last token - time of the first token) of the long request, consumer-side stamps (what bench_decode measures)
  wall_tps  (n_long - n_short) / (wall_long - wall_short) over the whole requests (what serve_client.py and the streaming client measure)
  ids_sha16 bench_decode.ids_digest over the exact token ids of the long request; text_sha16 sha256 of the concatenated Response.text
Exit 1 when a request does not end with finish_reason "length" and the exact token count.
"""
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
arm, model_dir, pairs, tokens, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ARMS = ("G", "S", "GM", "GL", "GS")
if arm not in ARMS:
    raise SystemExit(f"ARM must be one of {ARMS}")
sys.path.insert(0, HERE)
from mlx_provenance import ProvenanceRefusal, installed_provenance, provenance_line  # noqa: E402

try:
    prov = installed_provenance(expect_wheel=None)
    if prov["verified"] == "mismatch":
        raise ProvenanceRefusal(prov["mismatch"])
except ProvenanceRefusal as exc:
    print(f"REFUSING TO EMIT NUMBERS: {exc}", file=sys.stderr)
    raise SystemExit(3)
print(provenance_line(prov), flush=True)

import mlx_lm.server as server_module  # noqa: E402
from bench_decode import ids_digest  # noqa: E402

PROMPT_TEXT = prompt


def request_args(cli, n):
    sm = server_module
    return sm.GenerationArguments(
        model=sm.ModelDescription(model="default_model", draft="default_model", adapter=None),
        sampling=sm.SamplingArguments(temperature=0.0, top_p=cli.top_p, top_k=cli.top_k, min_p=cli.min_p,
                                      xtc_probability=0.0, xtc_threshold=0.1),
        logits=sm.LogitsProcessorArguments(logit_bias=None, repetition_penalty=0.0, repetition_context_size=20,
                                           presence_penalty=0.0, presence_context_size=20,
                                           frequency_penalty=0.0, frequency_context_size=20),
        stop_words=[], max_tokens=n, num_draft_tokens=cli.num_draft_tokens, logprobs=False, top_logprobs=-1, seed=None,
        chat_template_kwargs=None)


def one(rg, n):
    req = server_module.CompletionRequest("chat", "", [{"role": "user", "content": PROMPT_TEXT}], None, None)
    t0 = time.monotonic_ns()
    _ctx, it = rg.generate(req, request_args(rg.cli_args, n))
    ids, texts, stamps, finish = [], [], [], None
    for gen in it:
        stamps.append(time.monotonic_ns())
        ids.append(gen.token)
        texts.append(gen.text)
        finish = gen.finish_reason or finish
    wall = (time.monotonic_ns() - t0) / 1e9
    return {"n": len(ids), "finish": finish, "wall": wall, "stamps": stamps, "ids": ids, "text": "".join(texts)}


def driver(host, port, response_generator, server_class=None, handler_class=None):
    rg = response_generator
    ok = True
    try:
        warm = one(rg, 8)
        if warm["n"] != 8 or warm["finish"] != "length":
            print(json.dumps({"arm": arm, "error": f"warm-up n={warm['n']} finish={warm['finish']}"}))
            os._exit(1)
        table = subprocess.run([sys.executable, os.path.join(HERE, "thread_table.py"), str(os.getpid())],
                               capture_output=True, text=True, check=False).stdout.strip()
        print(f"ARM {arm} {table[:300]}", flush=True)
        for p in range(1, pairs + 1):
            short, long_ = one(rg, tokens // 4), one(rg, tokens)
            good = (short["finish"] == "length" and long_["finish"] == "length" and short["n"] == tokens // 4
                    and long_["n"] == tokens and long_["wall"] > short["wall"])
            ok = ok and good
            st = long_["stamps"]
            out = {"arm": arm, "pair": p, "n_short": short["n"], "n_long": long_["n"], "finish": long_["finish"],
                   "wall_short_s": round(short["wall"], 4), "wall_long_s": round(long_["wall"], 4),
                   "ids_sha16": ids_digest(long_["ids"]),
                   "text_sha16": hashlib.sha256(long_["text"].encode()).hexdigest()[:16]}
            if len(st) > 1:
                out["ts_tps"] = round((len(st) - 1) / ((st[-1] - st[0]) / 1e9), 4)
            if long_["wall"] > short["wall"]:
                out["wall_tps"] = round((long_["n"] - short["n"]) / (long_["wall"] - short["wall"]), 4)
            print(json.dumps(out, sort_keys=True), flush=True)
    finally:
        rg.stop_and_join()
    print(f"FLIP {arm} make_mask returned None {FLIP_STATS['none']} times, delegated {FLIP_STATS['delegated']} times", flush=True)
    os._exit(0 if ok else 1)


FLIP_STATS = {"none": 0, "delegated": 0}


def install_flip(which):
    """Remove ONE candidate cost from the stock stack (arms GM, GL, GS); the token ids must stay equal to arm G."""
    import mlx.core as mx

    if which == "GS":
        # server.py:660 takes mx.default_stream(...); generate.py:227 (bench_decode's path) uses a new thread-local stream.
        class _MX:
            def __getattr__(self, name):
                return getattr(mx, name)

            @staticmethod
            def default_stream(device=None):
                return mx.new_thread_local_stream(device or mx.default_device())

        server_module.mx = _MX()
    elif which == "GL":
        # server.py:867 reads r.logprobs[r.token].item() for every token; give the server a stub instead of the device array.
        from mlx_lm.generate import BatchGenerator as BG

        class _NoLogprob:
            def __getitem__(self, _i):
                return self

            def item(self):
                return 0.0

        real_next = BG.next

        def next_(self):
            prompt_responses, gen_responses = real_next(self)
            for r in gen_responses:
                r.logprobs = _NoLogprob()
            return prompt_responses, gen_responses

        BG.next = next_
    elif which == "GM":
        # cache.py:1447-1470 BatchRotatingKVCache.make_mask always builds a mask array; RotatingKVCache (cache.py:557-585, bench_decode's
        # path) returns None at N == 1 while the window is not binding. Return None only when the mask would be all True: one row
        # without left padding (checked once per cache), no rotation, no right-padding lengths, every key inside the window.
        from mlx_lm.models import cache as cache_mod

        rot = cache_mod.BatchRotatingKVCache
        real_make_mask = rot.make_mask

        def make_mask(self, N, window_size=None, return_array=False):
            if (N == 1 and not return_array and not self.rotated and self._lengths is None
                    and self._idx == self._offset and self._idx < self.max_size and self._offset < (window_size or self.max_size)):
                nopad = getattr(self, "_flip_nopad", None)
                if nopad is None:
                    nopad = self._flip_nopad = not any(v > 0 for v in self.left_padding.tolist())
                if nopad:
                    FLIP_STATS["none"] += 1
                    return None
            FLIP_STATS["delegated"] += 1
            return real_make_mask(self, N, window_size, return_array)

        rot.make_mask = make_mask


if arm in ("GM", "GL", "GS"):
    install_flip(arm)
driver.__signature__ = inspect.signature(server_module._run_http_server)
server_module._run_http_server = driver
sys.argv = ["mlx_lm.server", "--model", model_dir, "--host", "127.0.0.1", "--port", "0", "--max-tokens", "512",
            "--decode-concurrency", "1", "--prompt-concurrency", "1", "--prompt-cache-size", "0"]
if arm == "S":
    from mlx_omarchy_serve import _mlxlm_server as shim

    shim.main()
else:
    from mlx_omarchy_serve.perf_placement import apply_from_env

    applied, detail = apply_from_env()
    print(f"stock: uclamp placement {'applied' if applied else 'skipped'} ({detail})", file=sys.stderr, flush=True)
    server_module.main()
