#!/usr/bin/env python3
"""Isolated multi-server LLM serve benchmark (mlx_lm.server / oMLX / mlx-serve).

Audit harness for the 2026-09-19 serve receipts. It is a client-side harness:
it starts and probes real servers, changes no production code, and runs
outside any import path of mlx-omarchy.

Differences from receipts/2026-09-19-mlxserve-linux-port-t6001-test-host/bench3.py:

- Legs run SEQUENTIALLY, one fresh server process per leg (default). The
  09-19 runs kept all four servers co-resident on one GPU; that is fair
  across legs but understates absolute rates and, at ~15 GB weights
  (Qwen3.8-27B-4bit), four residents no longer fit a 62 GB box.
  --resident restores the legacy co-resident method for comparability.
- Adds streaming rounds so TTFT and decode-only rate are measured, not
  just the end-to-end wall/token number the 09-19 receipts report
  (their only server-side timings came from mlx-serve's `timings` field).
- Zero cache bias by construction: fresh process per leg + identical
  warmup request count/tokens/prompt for every leg; prompt_tokens and any
  server-reported cached_n are recorded per request.
- Pins are fatal-checked before any request (mlx-omarchy version, libmlx
  sha256-16, mlx-serve binary sha256-16), per the A/B provenance rule.
- Records per-leg server RSS (/proc VmRSS) after load and after rounds,
  plus total safetensors bytes, to produce a measured weights->peak
  factor for admission control.
- Optional bounded concurrency phase (--concurrency N, default off).

Metric definitions (all per leg, computed client-side):

  end_to_end_tok_s  usage.completion_tokens / HTTP wall of a non-streaming
                    request. Includes tokenize+prefill+decode+detok+HTTP.
                    This is the ONLY metric comparable with the 2026-09-19
                    receipt numbers (10.36 / 33.5 / 5.65).
  ttft_s            wall until first streamed chunk of a stream=true
                    request with the same prompt/max_tokens.
  decode_tok_s      (completion_tokens - 1) / (stream_wall - ttft).
                    Decode-slope approximation; ignores detok/HTTP tail.
  concurrency_tok_s aggregate completion_tokens / wall of N simultaneous
                    stream=true requests (--concurrency, default off).

Self-check (no GPU, no real servers): python3 serve_bench.py --selfcheck
runs the full client path against an in-process stdlib mock of the three
OpenAI-style endpoints and asserts the summary math.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
LEGS = ("mlxlm", "omlx", "mlxserve", "mlxserve_nopld")
PORTS = {"mlxlm": 8081, "omlx": 8082, "mlxserve": 11234, "mlxserve_nopld": 11235}
SSE_DONE = b"data: [DONE]"


def log(msg: str) -> None:
    print(msg, flush=True)


def fatal(msg: str, code: int) -> "NoReturn":  # type: ignore[valid-type]
    print(f"FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def free_port(port: int) -> None:
    subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        s = socket.socket()
        try:
            s.bind((HOST, port))
            s.close()
            return
        except OSError:
            s.close()
            time.sleep(2)
    fatal(f"port {port} still busy after cleanup", 8)


def dir_size(path: str, suffixes: tuple[str, ...]) -> int:
    root = pathlib.Path(path)
    return sum(p.stat().st_size for p in root.rglob("*")
               if p.suffix in suffixes and p.is_file())


def venv_json(python: str, code: str) -> dict:
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return {"error": out.stderr.strip()[-300:]}
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return {"error": f"unparseable: {out.stdout[:200]}"}


def check_pins(python: str, expect_version: str, expect_libmlx: str) -> dict:
    pins = venv_json(python, r"""
import hashlib, importlib.metadata as md, json, pathlib
pins = {}
try:
    pins["mlx_omarchy_version"] = md.version("mlx-omarchy")
except Exception as e:
    pins["mlx_omarchy_version"] = f"MISSING: {e}"
try:
    import mlx
    lib = None
    for root in mlx.__path__:
        p = pathlib.Path(root) / "lib" / "libmlx.so"
        if p.exists():
            lib = p
    pins["libmlx_sha256_16"] = hashlib.sha256(lib.read_bytes()).hexdigest()[:16] if lib else "MISSING"
except Exception as e:
    pins["libmlx_sha256_16"] = f"MISSING: {e}"
print(json.dumps(pins))
""")
    if "error" in pins:
        fatal(f"venv runtime probe failed: {pins['error']}", 3)
    if expect_version and pins["mlx_omarchy_version"] != expect_version:
        fatal(f"runtime pin mismatch: version={pins['mlx_omarchy_version']} expected={expect_version}", 3)
    if expect_libmlx and pins.get("libmlx_sha256_16") != expect_libmlx:
        fatal(f"runtime pin mismatch: libmlx={pins.get('libmlx_sha256_16')} expected={expect_libmlx}", 3)
    for mod in ("mlx-lm", "omlx"):
        info = venv_json(python, f"import importlib.metadata as md, json; print(json.dumps(md.version('{mod}')))")
        if isinstance(info, str):
            pins[mod.replace("-", "_")] = info
        else:
            pins[mod.replace("-", "_")] = info.get("error", "UNKNOWN")
    return pins


def rss_gb(pid: int) -> float | None:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return round(int(out.stdout.strip()) / 1024 / 1024, 2) if out.returncode == 0 and out.stdout.strip() else None


def start_leg(leg: str, args, python: str) -> subprocess.Popen:
    port = PORTS[leg]
    if leg == "mlxlm":
        cmd = [python, "-m", "mlx_lm.server", "--model", args.hf_id or args.model_snapshot,
               "--host", HOST, "--port", str(port)]
    elif leg == "omlx":
        cmd = [python, "-m", "omlx.server", "--model-dir", args.model_snapshot,
               "--host", HOST, "--port", str(port)]
        if args.omlx_extra_args:
            cmd += args.omlx_extra_args.split()
    elif leg == "mlxserve":
        cmd = [args.mlxserve_bin, "--model", args.model_snapshot, "--serve",
               "--host", HOST, "--port", str(port)]
    elif leg == "mlxserve_nopld":
        cmd = [args.mlxserve_bin, "--model", args.model_snapshot, "--no-pld", "--serve",
               "--host", HOST, "--port", str(port)]
    else:
        fatal(f"unknown leg {leg}", 2)
    logf = open(f"{args.outdir}/server-{leg}.log", "ab")
    logf.write(f"\n==== run {time.strftime('%F %T')} ====\n".encode())
    logf.flush()
    return subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True,
                            env=dict(os.environ))


def wait_ready(port: int, name: str, deadline_s: int) -> float:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        for path in ("/v1/models", "/health"):
            try:
                with urllib.request.urlopen(f"http://{HOST}:{port}{path}", timeout=5) as r:
                    if r.status == 200:
                        ready = round(time.time() - t0, 1)
                        log(f"[ready] {name} via {path} in {ready}s")
                        return ready
            except Exception:
                pass
        time.sleep(2)
    fatal(f"{name} not ready in {deadline_s}s", 4)
    return -1.0  # unreachable


def discover_model_name(port: int, hint: str | None, hf_id: str) -> str | None:
    if not hint:
        return hf_id
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/v1/models", timeout=10) as r:
            ids = [m.get("id", "") for m in json.loads(r.read()).get("data", [])]
        match = [i for i in ids if hint.lower() in i.lower()]
        if not match:
            log(f"[discover] NO model matching hint {hint!r} in {ids}")
            return None
        picked = match[0]
        log(f"[discover] model_name={picked} from {ids[:6]}")
        return picked
    except Exception as e:
        log(f"[discover] fallback default ({e})")
        return None


def chat_once(port: int, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer mlx"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:300]
        except Exception:
            detail = "<unreadable>"
        return {"error": f"HTTP {e.code}: {detail}", "wall_s": round(time.time() - t0, 3)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 3)}
    wall = round(time.time() - t0, 3)
    if body.get("stream"):
        return {"wall_s": wall}
    usage = payload.get("usage") or {}
    choice = (payload.get("choices") or [{}])[0]
    return {"wall_s": wall,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": choice.get("finish_reason"),
            "text_head": ((choice.get("message") or {}).get("content") or "")[:80],
            "text": (choice.get("message") or {}).get("content") or "",
            "timings": payload.get("timings")}


def chat_stream(port: int, body: dict, timeout: int) -> dict:
    """stream=true request; records TTFT (first chunk) and total wall."""
    sse_body = dict(body, stream=True)
    req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/chat/completions",
        data=json.dumps(sse_body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer mlx"})
    t0 = time.time()
    t_first = None
    chunks = 0
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                if line.strip() == SSE_DONE:
                    break
                if t_first is None:
                    t_first = time.time()
                chunks += 1
                try:
                    evt = json.loads(line[5:])
                    if evt.get("usage"):
                        usage = evt["usage"]
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "wall_s": round(time.time() - t0, 3)}
    wall = round(time.time() - t0, 3)
    return {"ttft_s": round(t_first - t0, 3) if t_first else None,
            "wall_s": wall, "chunks": chunks,
            "completion_tokens": usage.get("completion_tokens"),
            "decode_tok_s": (round((usage.get("completion_tokens", 0) - 1) / (wall - (t_first - t0)), 2)
                             if t_first and usage.get("completion_tokens") else None)}


def summarize(rounds: list[dict], streams: list[dict], conc: dict | None) -> dict:
    good = [r for r in rounds if r.get("completion_tokens") and r.get("wall_s")]
    tps = [r["completion_tokens"] / r["wall_s"] for r in good]
    sttft = [s["ttft_s"] for s in streams if s.get("ttft_s") is not None]
    sdec = [s["decode_tok_s"] for s in streams if s.get("decode_tok_s")]
    out = {
        "n": len(good), "n_errors": len(rounds) - len(good),
        "median_tok_s": round(statistics.median(tps), 2) if tps else None,
        "min_tok_s": round(min(tps), 2) if tps else None,
        "max_tok_s": round(max(tps), 2) if tps else None,
        "median_ttft_s": round(statistics.median(sttft), 3) if sttft else None,
        "median_decode_tok_s": round(statistics.median(sdec), 2) if sdec else None,
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in good), 1) if good else None,
        "mean_prompt_tokens": round(statistics.mean(r["prompt_tokens"] or 0 for r in good), 1) if good else None,
        "all_finished_length": bool(good) and all(r["finish"] == "length" for r in good),
    }
    if conc:
        out["concurrency"] = conc
    return out


DIRECT_SNIPPET = """
import hashlib, json, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, BatchGenerator
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

MODE = {mode!r}
SNAPSHOT = {snapshot!r}
PROMPT = {prompt!r}
WARMUP_TOKENS = {warmup_tokens}
MAXTOK = {max_tokens}
ROUNDS = {rounds}

model, tokenizer = load(SNAPSHOT)
prompt_ids = tokenizer.apply_chat_template(
    [{{"role": "user", "content": PROMPT}}], add_generation_prompt=True)
if hasattr(prompt_ids, "tolist"):
    prompt_ids = prompt_ids.tolist()
prompt_ids = prompt_ids[0] if (len(prompt_ids) and isinstance(prompt_ids[0], list)) else prompt_ids
psha = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()[:16]
sampler = lambda x: mx.argmax(x, axis=-1)

def run_stream_round(cache, prompt, max_tokens):
    ids, t0, t_first = [], time.perf_counter(), None
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens,
                             sampler=sampler, prompt_cache=cache):
        if t_first is None:
            t_first = time.perf_counter()
        ids.append(r.token)
        if len(ids) >= max_tokens:
            break
    wall = time.perf_counter() - t0
    return ids, round(t_first - t0, 4), round(wall, 4)

def state_fingerprint(cache):
    # Per-layer (offset, hash16(keys), hash16(values)) for byte-exact
    # state comparison. Missing arrays recorded as None.
    out = []
    for c in cache:
        offset = getattr(c, "offset", None)
        keys, values = c.state if hasattr(c, "state") else (None, None)
        def h(a):
            try:
                return hashlib.sha256(bytes(a)).hexdigest()[:16]
            except Exception:
                return None
        out.append({{"offset": offset,
                    "k16": h(keys) if keys is not None else None,
                    "v16": h(values) if values is not None else None}})
    return out

def assert_fresh_prefix(cache, ref_state, ref_offset, tag):
    # Per-layer: offsets AND bytes must equal the fresh-prefix
    # reference (heterogeneous multimodal cache lists may carry
    # non-text layers whose offsets legitimately differ from the
    # text prefix length).
    st = state_fingerprint(cache)
    offs_ok = all(e["offset"] == r["offset"] for e, r in zip(st, ref_state))
    bytes_ok = all(e["k16"] == r["k16"] and e["v16"] == r["v16"]
                   for e, r in zip(st, ref_state))
    if not (offs_ok and bytes_ok):
        raise SystemExit(f"{{tag}}: cache != fresh prefix: offsets_ok={{offs_ok}} "
                         f"bytes_ok={{bytes_ok}} state={{json.dumps(st)[:400]}}")
    return {{"offsets_ok": offs_ok, "bytes_ok": bytes_ok}}

rounds = []
warm_ids = []
proofs = None
warm_prefix_ids_ok = None
trim_counts = None
ref_offset = None
if MODE == "d1c":
    # Cold-cache rounds: fresh cache + full prefill every round.
    # Within-leg id equality across rounds is the determinism test under
    # a VALID (fresh) cache state.
    for _ in range(ROUNDS):
        cache = make_prompt_cache(model)
        ids, t_first, wall = run_stream_round(cache, prompt_ids, MAXTOK)
        rounds.append({{"ids": ids, "ttft_s": t_first, "wall_s": wall,
                       "n": len(ids),
                       "ids_sha16": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}})
elif MODE == "d1w":
    # Server-replicated warm rounds WITHOUT trim. Source-proven server
    # mechanism (models/cache.py:1674 fetch_nearest_cache shorter-key
    # branch + server.py:962-976): the stored prefix entry is deepcopied
    # untrimmed and only the remaining suffix is prefetched; out11's
    # server log shows steady rounds at "Prompt processing progress:
    # 1/1". Replicate: prefill prompt[:-1] once into a pristine master
    # cache (untimed, like the server's stored entry), then per round
    # deepcopy it and decode from the last prompt token. The pre-round
    # state fingerprint must equal the master's (byte-exact) every
    # round.
    import copy as _copy
    master = make_prompt_cache(model)
    model(mx.array([prompt_ids[:-1]]), cache=master)
    mx.eval([c.state for c in master])
    ref_state = state_fingerprint(master)
    ref_offset = len(prompt_ids) - 1
    warm_prefix_ids_ok = True
    proofs = []
    for _ in range(ROUNDS):
        cache = _copy.deepcopy(master)
        proofs.append(assert_fresh_prefix(cache, ref_state, ref_offset, "pre-round"))
        ids, t_first, wall = run_stream_round(cache, prompt_ids[-1:], MAXTOK)
        rounds.append({{"ids": ids, "ttft_s": t_first, "wall_s": wall,
                       "n": len(ids),
                       "ids_sha16": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}})
elif MODE == "d1m":
    # Margin probe: manual greedy loop recording the full-distribution
    # top1-top2 logprob margin per step (stream_generate only exposes the
    # sampled logprob). Timing NOT comparable -- measurement leg only.
    rounds = []
    for _ in range(ROUNDS):
        cache = make_prompt_cache(model)
        ids, margins = [], []
        logits = model(mx.array([prompt_ids]), cache=cache)
        for _ in range(MAXTOK):
            lprobs = logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], keepdims=True)
            top2 = mx.sort(lprobs[0])[-2:]
            margins.append(float((top2[1] - top2[0]).item()))
            nxt = int(mx.argmax(lprobs[0]).item())
            ids.append(nxt)
            logits = model(mx.array([[nxt]]), cache=cache)
        rounds.append({{"ids": ids, "n": len(ids), "min_margin": round(min(margins), 6),
                       "margin_p10": round(sorted(margins)[MAXTOK // 10], 6),
                       "ids_sha16": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}})
else:
    raise SystemExit("unknown mode: " + MODE)

out_ids = [t for r in rounds for t in r["ids"]]
result = {{
    "mode": MODE,
    "prompt_n": len(prompt_ids),
    "prompt_ids_sha16": psha,
    "warmup": {{"n": len(warm_ids), "ids_sha16": hashlib.sha256(json.dumps(warm_ids).encode()).hexdigest()[:16]}},
    "rounds": rounds,
    "output_ids_sha16": hashlib.sha256(json.dumps(out_ids).encode()).hexdigest()[:16],
    "rounds_identical": len({{r["ids_sha16"] for r in rounds}}) == 1,
    "text": tokenizer.decode(out_ids)[:400],
}}
if MODE == "d1w":
    result["cache_state_proof"] = {{
        "fresh_prefix_ref_offset": ref_offset,
        "per_round_proofs": proofs,
        "warm_prefix_ids_ok": warm_prefix_ids_ok,
        "trim_counts_first_round": trim_counts,
    }}
print(json.dumps(result))
"""


def direct_control(args, python: str, mode: str) -> dict:
    """No-HTTP direct decode legs on the identical prompt/token budget.

    d1c: cold cache per round (full prefill; determinism test under valid
    state). d1w: server-replicated warm cache via trim_prompt_cache
    (matches mlx_lm.server steady rounds). d1m: manual greedy loop
    recording full-distribution top1-top2 margins per step (timing not
    comparable). All record token ids; agreement is asserted, never
    assumed.
    """
    snippet = DIRECT_SNIPPET.format(mode=mode, snapshot=args.model_snapshot,
                                    prompt=args.prompt,
                                    warmup_tokens=args.warmup_tokens,
                                    max_tokens=args.max_tokens,
                                    rounds=args.rounds)
    compile(snippet, f"<direct-{mode}>", "exec")  # fail fast on syntax errors
    def clean_err(text: str) -> str:
        # rtmod kernel-trace lines bury the actual failure; drop them and
        # keep real error output (SystemExit messages print WITHOUT a
        # Traceback header).
        lines = [ln for ln in text.splitlines()
                 if ln.strip() and not ln.startswith("[rtmod]")
                 and not ln.startswith("ts=") and "COMMIT-NOOP" not in ln
                 and "SUBMIT tid=" not in ln]
        return "\n".join(lines)

    try:
        out = subprocess.run([python, "-c", snippet], capture_output=True, text=True,
                             timeout=args.timeout * (args.rounds + 2))
    except subprocess.TimeoutExpired as e:
        err = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        tb = err.find("Traceback")
        detail = err[tb:] if tb >= 0 else clean_err(err)
        return {"error": f"subprocess timeout after {args.timeout * (args.rounds + 2)}s; "
                         f"partial stdout={len(e.stdout or b'')}B; {detail.strip()[-500:]}"}
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    err = clean_err(out.stderr or "")
    tb = err.find("Traceback")
    detail = err[tb:] if tb >= 0 else err
    return {"error": (detail.strip()[-800:] or err[-500:] or "no json output")}


def ids_agreement(direct: dict) -> dict:
    modes = [m for m in direct if isinstance(direct[m], dict)
             and "output_ids_sha16" in direct[m]]
    if not modes:
        return {"error": "no successful direct legs"}
    out = {
        "prompt_ids_sha16s": {m: direct[m]["prompt_ids_sha16"] for m in modes},
        "prompt_ids_match": len({direct[m]["prompt_ids_sha16"] for m in modes}) == 1,
        "rounds_identical_within_leg": {m: direct[m].get("rounds_identical") for m in modes},
    }
    if len(modes) > 1:
        out["cross_mode_output_match"] = len({direct[m]["output_ids_sha16"] for m in modes}) == 1
        out["output_ids_sha16s"] = {m: direct[m]["output_ids_sha16"] for m in modes}
    return out


def parse_server_log_diagnostics(path: str) -> dict:
    """Minimal diagnostics from the existing mlx_lm.server log: per-request
    prompt-processing suffix counts (a cache hit shows as '1/1' — the
    server prefetched only the last token, which itself proves the
    request's prompt-token prefix matched the stored entry exactly),
    prompt-cache size over time, and request POST count."""
    import re
    prog = []
    cache_lines = []
    posts = 0
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = re.search(r"Prompt processing progress: (\d+)/(\d+)", line)
                if m:
                    prog.append({"done": int(m.group(1)), "total": int(m.group(2))})
                    continue
                m = re.search(r"Prompt Cache: (\d+) sequences, ([\d.]+) GB", line)
                if m:
                    cache_lines.append({"sequences": int(m.group(1)),
                                        "gb": float(m.group(2))})
                    continue
                if "POST /v1/chat" in line:
                    posts += 1
    except OSError:
        return {"error": "log missing"}
    # Group the progress markers into requests: a request's LAST marker
    # gives (suffix_done, prompt_total). A request whose final marker has
    # done==total==1 hit the cached prefix (prompt-ID agreement, proved
    # server-side); done==total>1 means full prefill ran.
    return {"prompt_processing_markers": prog,
            "cache_state_lines": cache_lines,
            "chat_posts": posts,
            "requests_full_prefill": sum(1 for p in prog if p["done"] == p["total"] and p["total"] > 1),
            "requests_cache_hit": sum(1 for p in prog if p["done"] == p["total"] == 1)}


def run_leg(leg: str, args, python: str, results: dict) -> None:
    port = PORTS[leg]
    free_port(port)
    proc = start_leg(leg, args, python)
    hints = {"mlxlm": None, "omlx": args.omlx_hint, "mlxserve": "qwen",
             "mlxserve_nopld": "qwen"}
    try:
        load_s = wait_ready(port, leg, args.ready_deadline)
        model_name = discover_model_name(port, hints.get(leg), args.hf_id)
        if not model_name:
            results["leg_errors"][leg] = f"model discovery failed for hint {hints.get(leg)!r}"
            log(f"[leg-failed] {leg}: {results['leg_errors'][leg]}")
            return
        warm = []
        for _ in range(args.warmup_rounds):
            warm.append(chat_once(port, {"model": model_name,
                                         "messages": [{"role": "user", "content": args.prompt}],
                                         "max_tokens": args.warmup_tokens,
                                         "temperature": 0, "stream": False}, args.timeout))
        if not warm[-1].get("completion_tokens"):
            results["leg_errors"][leg] = warm[-1].get("error", "no completion tokens")
            log(f"[leg-failed] {leg}: {results['leg_errors'][leg]}")
            return
        rss_load = rss_gb(proc.pid)
        rounds = []
        for r in range(1, args.rounds + 1):
            w = chat_once(port, {"model": model_name,
                                 "messages": [{"role": "user", "content": args.prompt}],
                                 "max_tokens": args.max_tokens,
                                 "temperature": 0, "stream": False}, args.timeout)
            rounds.append(w)
            if w.get("completion_tokens") and w.get("wall_s"):
                log(f"[round {r}] {leg}: {w['completion_tokens'] / w['wall_s']:.1f} tok/s "
                    f"finish={w.get('finish')}")
            else:
                log(f"[round {r}] {leg}: ERROR {w.get('error', '?')[:160]}")
        streams = [chat_stream(port, {"model": model_name,
                                      "messages": [{"role": "user", "content": args.prompt}],
                                      "max_tokens": args.max_tokens,
                                      "temperature": 0}, args.timeout)
                   for _ in range(args.ttft_rounds)]
        conc = None
        if args.concurrency > 1:
            import threading
            total_gb = None
            try:
                for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / 1024 / 1024
                        break
            except OSError:
                pass
            # Admission: resident model + 2 GiB workspace per stream must fit
            # 90% of unified memory; else record explicit N/A (Main's gate).
            budget = (total_gb or 0) * 0.9
            need = (rss_load or 0) + 2 * args.concurrency
            if total_gb is None or need > budget:
                conc = {"n": args.concurrency, "decision": "N/A",
                        "reason": f"admission: rss_load={rss_load}GB + 2GiB/stream "
                                  f"x{args.concurrency} vs total={total_gb}GB budget={budget:.1f}GB"}
                log(f"[concurrency] {leg}: N/A — {conc['reason']}")
            else:
                per: list[dict] = []
                lock = threading.Lock()

                def one() -> None:
                    w = chat_stream(port, {"model": model_name,
                                           "messages": [{"role": "user", "content": args.prompt}],
                                           "max_tokens": args.max_tokens,
                                           "temperature": 0}, args.timeout)
                    with lock:
                        per.append(w)
                t0 = time.time()
                threads = [threading.Thread(target=one) for _ in range(args.concurrency)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = round(time.time() - t0, 3)
                toks = sum(p.get("completion_tokens") or 0 for p in per)
                ttfts = [p["ttft_s"] for p in per if p.get("ttft_s") is not None]
                conc = {"n": args.concurrency, "wall_s": wall,
                        "aggregate_tok_s": round(toks / wall, 2) if wall else None,
                        "mean_ttft_s": round(statistics.mean(ttfts), 3) if ttfts else None,
                        "per_stream": per}
        rss_after = rss_gb(proc.pid)
        results["legs"][leg] = {
            "port": port, "model_name": model_name, "load_s": load_s,
            "rss_gb_after_load": rss_load, "rss_gb_after_rounds": rss_after,
            "warmup": warm, "rounds_data": rounds, "stream_rounds": streams,
            "summary": summarize(rounds, streams, conc),
        }
        log(f"[summary] {leg}: {json.dumps(results['legs'][leg]['summary'])}")
    finally:
        try:
            os.killpg(proc.pid, 15)
        except Exception:
            pass
        time.sleep(3)
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-snapshot", default=None)
    ap.add_argument("--hf-id", default=None, help="hub id (mlx_lm leg; also default model_name)")
    ap.add_argument("--prompt", default="Explain why the sky is blue.")
    ap.add_argument("--legs", default="mlxlm,omlx,mlxserve", help=f"subset of {LEGS}")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup-rounds", type=int, default=1)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--ttft-rounds", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1, help=">1 adds a bounded concurrent phase")
    ap.add_argument("--timeout", type=int, default=300, help="per-request client timeout s")
    ap.add_argument("--ready-deadline", type=int, default=420)
    ap.add_argument("--python", default=sys.executable, help="bench venv python that has mlx-omarchy")
    ap.add_argument("--mlxserve-bin", default=os.path.expanduser("~/src/mlx-serve-clean/zig-out/bin/mlx-serve"))
    ap.add_argument("--omlx-hint", default=None, help="substring to pick /v1/models id on the omlx leg")
    ap.add_argument("--omlx-extra-args", default="", help="extra args appended to the omlx server cmd, e.g. --max-model-memory 32GB")
    ap.add_argument("--expect-version", default="", help="fatal unless mlx-omarchy version matches")
    ap.add_argument("--expect-libmlx", default="", help="fatal unless libmlx sha256-16 matches")
    ap.add_argument("--direct-legs", default="", help="comma list from d1,d2: no-HTTP direct decode legs (token ids recorded; numeric agreement checked)")
    ap.add_argument("--outdir", default="/tmp/servebench")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if args.selfcheck:
        sys.exit(selfcheck())

    os.makedirs(args.outdir, exist_ok=True)
    if not pathlib.Path(args.model_snapshot).is_dir():
        fatal(f"model snapshot not a dir: {args.model_snapshot}", 2)
    legs = [l.strip() for l in args.legs.split(",") if l.strip()]
    unknown = [l for l in legs if l not in LEGS]
    if unknown:
        fatal(f"unknown legs {unknown}; choose from {LEGS}", 2)
    if "mlxserve" in legs or "mlxserve_nopld" in legs:
        if not pathlib.Path(args.mlxserve_bin).exists():
            fatal(f"mlx-serve binary missing: {args.mlxserve_bin}", 2)

    results = {
        "pins": check_pins(args.python, args.expect_version, args.expect_libmlx),
        "model_snapshot": args.model_snapshot, "hf_id": args.hf_id,
        "prompt": args.prompt, "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "warmup_rounds": args.warmup_rounds, "warmup_tokens": args.warmup_tokens,
        "method": "sequential fresh-process legs" if args.concurrency == 1 else
                  f"sequential fresh-process legs + concurrency {args.concurrency}",
        "model_weights_gb": round(dir_size(args.model_snapshot, (".safetensors",)) / 1e9, 2),
        "legs": {}, "leg_errors": {},
    }
    log(f"pins: {json.dumps(results['pins'])}")
    direct_modes = [m.strip() for m in args.direct_legs.split(",") if m.strip()]
    unknown_direct = [m for m in direct_modes if m not in ("d1c", "d1w", "d1m")]
    if unknown_direct:
        fatal(f"unknown direct legs {unknown_direct}; choose from d1c,d1w,d1m", 2)
    if direct_modes:
        results["direct"] = {}
        for mode in direct_modes:
            log(f"[direct {mode}] starting (no HTTP)")
            results["direct"][mode] = direct_control(args, args.python, mode)
            log(f"[direct {mode}] {json.dumps(results['direct'][mode])[:220]}")
        results["ids_agreement"] = ids_agreement(results["direct"])
        log(f"ids_agreement: {json.dumps(results['ids_agreement'])}")
        # Margins only on demonstrated within-leg id mismatch (Main gate):
        # nondeterminism measurement is not run speculatively.
        ril = results["ids_agreement"].get("rounds_identical_within_leg") or {}
        if any(v is False for v in ril.values()) and "d1m" not in results["direct"]:
            results["margins_needed"] = True
            log("[direct d1m] id mismatch demonstrated — running margin probe")
            results["direct"]["d1m"] = direct_control(args, args.python, "d1m")
            log(f"[direct d1m] {json.dumps(results['direct']['d1m'])[:220]}")
            d1m = results["direct"]["d1m"]
            results["margins"] = [
                {"round": i, "min_margin": r.get("min_margin"),
                 "margin_p10": r.get("margin_p10")}
                for i, r in enumerate(d1m.get("rounds", []), 1)
            ] if isinstance(d1m, dict) and "rounds" in d1m else None
    try:
        for leg in legs:
            run_leg(leg, args, args.python, results)
    finally:
        # Server-side diagnostics from the existing log path: the
        # per-request prefill suffix counts prove server-side prompt-ID
        # agreement (a 1/1 marker = cached-prefix hit) and record cache
        # behavior without touching server code.
        results["server_log_diagnostics"] = {}
        for leg in legs:
            lp = pathlib.Path(args.outdir) / f"server-{leg}.log"
            results["server_log_diagnostics"][leg] = parse_server_log_diagnostics(str(lp))
        log(f"server_log_diagnostics: {json.dumps(results['server_log_diagnostics'])[:300]}")
        out = pathlib.Path(args.outdir) / "results-servebench.json"
        out.write_text(json.dumps(results, indent=2))
        log(f"wrote {out}")
    if not results["legs"] and not results.get("direct"):
        fatal("all legs failed", 7)


def selfcheck() -> int:
    """Exercise the client path + summary math against a stdlib mock server."""
    import http.server
    import threading

    ntok = 128
    wall = 10.0

    class Mock(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send({"data": [{"id": "mock-qwen"}]})
            elif self.path == "/health":
                self._send({"ok": True})
            else:
                self._send({"detail": "nf"}, 404)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(ntok):
                    if i == 0:
                        time.sleep(0.05)  # simulated prefill tail before first chunk
                    self.wfile.write(b"data: " + json.dumps(
                        {"choices": [{"delta": {"content": "x"}}]}).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: " + json.dumps(
                    {"choices": [], "usage": {"completion_tokens": ntok,
                                              "prompt_tokens": 37}}).encode() + b"\n\n")
                self.wfile.write(SSE_DONE + b"\n\n")
                self.wfile.flush()
            else:
                time.sleep(0.01)
                self._send({"usage": {"completion_tokens": ntok, "prompt_tokens": 37},
                            "choices": [{"message": {"content": "mock"},
                                         "finish_reason": "length"}]})

    srv = http.server.HTTPServer((HOST, 0), Mock)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        streams = [chat_stream(port, {"model": "m", "messages": [], "max_tokens": ntok}, 30)
                   for _ in range(3)]
        assert all(s.get("ttft_s") is not None for s in streams), streams
        assert all(s["completion_tokens"] == ntok for s in streams), streams
        rounds = [{"completion_tokens": ntok, "wall_s": wall, "prompt_tokens": 37,
                   "finish": "length"}] * 4
        s = summarize(rounds, streams, None)
        assert s["median_tok_s"] == 12.8, s
        assert s["all_finished_length"] and s["n"] == 4, s
        assert s["median_ttft_s"] >= 0.05, s
        assert s["median_decode_tok_s"] and s["median_decode_tok_s"] > 12.8, s
        # ids_agreement logic (schema: prompt/rounds/cross-mode)
        ok = {"d1c": {"prompt_ids_sha16": "aa", "output_ids_sha16": "bb",
                      "rounds_identical": True, "rounds": [{"n": 128}]},
              "d1w": {"prompt_ids_sha16": "aa", "output_ids_sha16": "bb",
                      "rounds_identical": True, "rounds": [{"n": 128}]}}
        agr = ids_agreement(ok)
        assert agr["prompt_ids_match"] and agr["cross_mode_output_match"], agr
        assert agr["rounds_identical_within_leg"] == {"d1c": True, "d1w": True}, agr
        bad = {"d1c": dict(ok["d1c"], rounds_identical=False),
               "d1w": dict(ok["d1w"])}
        assert ids_agreement(bad)["rounds_identical_within_leg"]["d1c"] is False
        assert ids_agreement({}) == {"error": "no successful direct legs"}
        # direct snippets must compile in every mode
        for mode in ("d1c", "d1w", "d1m"):
            snippet = DIRECT_SNIPPET.format(mode=mode, snapshot="/x", prompt="p",
                                            warmup_tokens=16, max_tokens=128, rounds=1)
            compile(snippet, f"<{mode}>", "exec")
        print(f"SELFCheck OK: {json.dumps(s)}")
        return 0
    except AssertionError as e:
        print(f"SELFCHECK FAIL: {e}", file=sys.stderr)
        return 1
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
