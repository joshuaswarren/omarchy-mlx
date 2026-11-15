#!/usr/bin/env python3
"""Qualify a pair's chat model at a backend context cap on the machine it runs on.

Adopts the saved pair lock, starts the pair, grows the chat worker with ensure_context(--tokens)
(the dev flag lifts the catalog cap for this run only), then sends three requests straight to the
chat worker: an omp-shaped one (long system prompt + 52 tool schemas), a needle near the cap, and
one over the cap that must be refused with the budget reason. Writes a JSON receipt.

Run with MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 MLX_OMARCHY_OFFLINE=1 and PYTHONPATH at the serve tree.
Usage: qualify_backend_context.py --home DIR --pair compact --tokens 16384 --out receipt.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SENTENCE = "The ledger for shipment {n} lists {n} crates of brass fittings, checked on day {n}. "


def post(url, body, timeout):
    request = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read()), round(time.monotonic() - started, 2)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace"), round(time.monotonic() - started, 2)


def filler(count):
    return "".join(SENTENCE.format(n=n) for n in range(count))


def per_sentence_tokens(url):
    """Tokens per filler sentence, from the server's own usage count (two probes, so the fixed part cancels)."""
    def used(count):
        _, reply, _ = post(url, {"messages": [{"role": "user", "content": filler(count)}], "max_tokens": 1}, 120)
        return reply["usage"]["prompt_tokens"]
    low, high = used(100), used(400)
    return (high - low) / 300


def tool_schemas(count):
    return [{"type": "function", "function": {
        "name": f"tool_{i}", "description": ("Reads or changes the workspace item named by the caller. " * 4).strip(),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Item path, relative to the workspace root."},
            "mode": {"type": "string", "enum": ["read", "write", "append"]},
            "limit": {"type": "integer", "description": "Maximum number of lines to return."}},
            "required": ["path"]}}} for i in range(count)]


def rss_hwm_kb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1])


def worker_pid(port):
    out = subprocess.run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    match = re.search(r"pid=(\d+)", out)
    return int(match.group(1)) if match else None


def mem_available_kb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])


def proc_kb(pid, *keys):
    found = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            name = line.split(":")[0]
            if name in keys:
                found[name] = int(line.split()[1])
    except OSError:
        pass
    return found


def prefill_step(pid):
    """The --prefill-step-size the worker was really launched with (None if the flag is absent)."""
    try:
        argv = open(f"/proc/{pid}/cmdline", "rb").read().decode().split("\0")
    except OSError:
        return None
    return int(argv[argv.index("--prefill-step-size") + 1]) if "--prefill-step-size" in argv else None


class Sampler(threading.Thread):
    """Every 0.5 s: lowest MemAvailable seen and the highest RssAnon / RssFile / RssShmem of the chat worker."""

    def __init__(self):
        super().__init__(daemon=True)
        self.pid = None
        self.min_available_kb = mem_available_kb()
        self.peak = {}
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.wait(0.5):
            self.min_available_kb = min(self.min_available_kb, mem_available_kb())
            if self.pid:
                for key, value in proc_kb(self.pid, "RssAnon", "RssFile", "RssShmem", "VmHWM").items():
                    self.peak[key] = max(self.peak.get(key, 0), value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--margin", type=int, default=512)
    args = parser.parse_args()
    from mlx_omarchy_assistant.pairs import PairManager

    manager = PairManager(args.home)
    receipt = {"pair": args.pair, "tokens": args.tokens, "steps": {}}
    sampler = Sampler()
    sampler.start()
    receipt["mem_available_kb_baseline"] = mem_available_kb()  # before any worker of this run
    try:
        manager.adopt_saved(args.pair)
        manager.start()
        receipt["context_before"] = manager.status().get("context_tokens")
        receipt["mem_available_kb_at_4096"] = mem_available_kb()
        receipt["ensure_context"] = manager.ensure_context(args.tokens)
        status = manager.status()
        url = status["chat_url"]
        receipt["context_after"] = status.get("context_tokens")
        pid = worker_pid(int(url.rsplit(":", 1)[1].split("/")[0]))
        sampler.pid = pid
        receipt["prefill_step_size"] = prefill_step(pid)
        receipt["mem_available_kb_idle_at_cap"] = mem_available_kb()
        receipt["worker_idle_kb"] = proc_kb(pid, "RssAnon", "RssFile", "RssShmem")
        per = per_sentence_tokens(url)
        receipt["tokens_per_sentence"] = round(per, 2)

        system = ("You are a coding agent. Follow the project rules and use the tools when needed. " * 100)[:8600]
        status_code, reply, seconds = post(url, {"messages": [
            {"role": "system", "content": system}, {"role": "user", "content": "Say ok."}],
            "tools": tool_schemas(52), "max_tokens": 30}, 900)
        receipt["steps"]["omp_shaped"] = {"http": status_code, "seconds": seconds,
                                          "prompt_tokens": reply["usage"]["prompt_tokens"] if status_code == 200 else None,
                                          "reply": (reply["choices"][0]["message"].get("content") or "")[:80] if status_code == 200 else reply}

        needle = "The vault access code is 7316. Remember it.\n"
        count = int((args.tokens - args.margin - 80) / per)
        status_code, reply, seconds = post(url, {"messages": [{"role": "user", "content":
            needle + filler(count) + "\nWhat is the vault access code? Answer with the number only."}], "max_tokens": 64}, 1800)
        receipt["steps"]["needle_near_cap"] = {"http": status_code, "seconds": seconds,
            "prompt_tokens": reply["usage"]["prompt_tokens"] if status_code == 200 else None,
            "found": status_code == 200 and "7316" in (reply["choices"][0]["message"].get("content") or ""),
            "reply": (reply["choices"][0]["message"].get("content") or "")[:80] if status_code == 200 else reply}
        receipt["worker_peak_rss_kb"] = rss_hwm_kb(pid) if pid else None
        receipt["mem_available_kb_after"] = mem_available_kb()
        receipt["pair_peak_footprint_kb"] = receipt["mem_available_kb_baseline"] - sampler.min_available_kb
        receipt["worker_peak_kb"] = dict(sampler.peak)
        from mlx_omarchy_serve import budget
        chat = {m["id"]: m for m in manager.catalog["models"]}[manager.status()["chat_model"]]
        step = budget.effective_prefill_step(budget.DEFAULT_PREFILL_STEP_TOKENS, args.tokens)
        receipt["estimate_bytes"] = {
            "no_prefill": budget.estimate_required(chat["memory"], args.tokens).total,
            "with_step": budget.estimate_required(chat["memory"], args.tokens, step).total,
            "step": step}

        status_code, reply, seconds = post(url, {"messages": [{"role": "user", "content": filler(int(args.tokens / per) + 200)}],
                                                 "max_tokens": 30}, 900)
        receipt["steps"]["over_cap"] = {"http": status_code, "seconds": seconds, "body": str(reply)[:300]}
        status_code, reply, seconds = post(url, {"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8}, 120)
        receipt["steps"]["alive_after_refusal"] = {"http": status_code}
    finally:
        sampler.stop_event.set()
        manager.stop()
    steps = receipt["steps"]
    receipt["pass"] = bool(receipt.get("ensure_context", {}).get("ok")
                           and receipt["context_after"] == args.tokens
                           and steps["omp_shaped"]["http"] == 200
                           and steps["needle_near_cap"]["http"] == 200 and steps["needle_near_cap"]["found"]
                           and steps["over_cap"]["http"] == 404
                           and "admitted context budget" in steps["over_cap"]["body"]
                           and steps["alive_after_refusal"]["http"] == 200)
    with open(args.out, "w") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
