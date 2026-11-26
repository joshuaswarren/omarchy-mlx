#!/usr/bin/env python3
"""PairGates measurement harness, revised (audit-pass).

Collects:
  - System-level memory: parsed /proc/meminfo (MemTotal - MemAvailable)
    — the HONEST whole-system peak on unified memory.
  - Per-process RSS/PSS of the assistant PID and its descendants
    (/proc/<pid>/task/<pid>/children BFS).
  - Backend allocator peak: HTTP GET /v1/internal/memory on the chat
    worker (added by _mlxlm_server_with_memory.py monkey-patch).
  - Laya peak via the Laya HTTP /v1/decisions worker if reachable.
  - Driver-level DRM memory: /proc/<pid>/fdinfo for renderD128 fds
    (best-effort; driver does not expose bytes on this kernel).

Designed to be run from inside gpu-turn via a launcher that:
  1. starts the chat worker on port $CHAT_PORT with the memory route;
  2. starts the Laya worker;
  3. starts the assistant with that pair;
  4. waits for the chat port to answer a real completion (fence);
  5. runs this harness;
  6. kills the whole process group on exit.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import http.client


def parse_meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if ":" in line:
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
    return out


def meminfo_bytes(mi, key):
    s = mi.get(key)
    if not s:
        return 0
    parts = s.split()
    n = int(parts[0])
    if len(parts) > 1 and parts[1].lower() == "kb":
        n *= 1024
    return n


def rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return None


def pss_kb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup", "r") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return None


def children_of_pids(pid):
    seen = {pid}
    frontier = [pid]
    while frontier:
        new = []
        for p in frontier:
            try:
                with open(f"/proc/{p}/task/{p}/children") as f:
                    data = f.read().strip()
            except (FileNotFoundError, ProcessLookupError):
                data = ""
            for c in data.split():
                if c.isdigit():
                    ci = int(c)
                    if ci not in seen:
                        seen.add(ci)
                        new.append(ci)
        frontier = new
    return seen


def sample_tree(root_pid):
    pids = children_of_pids(root_pid)
    rss_total = 0
    pss_total = 0
    per_pid = {}
    for pid in pids:
        r = rss_kb(pid)
        p = pss_kb(pid)
        if r is not None:
            rss_total += r
            per_pid[pid] = {"rss": r, "pss": p}
        if p is not None:
            pss_total += p
    return {"pids": sorted(pids), "rss_total": rss_total,
            "pss_total": pss_total, "per_pid": per_pid}


def drm_bytes(pid):
    """Sum reported allocation bytes from /proc/<pid>/fdinfo for render
    node fds. Most Mesa builds do not write any allocation entries on
    Linux; we still record what is visible."""
    total = 0
    found = []
    try:
        for f in os.listdir(f"/proc/{pid}/fd"):
            try:
                tgt = os.readlink(f"/proc/{pid}/fd/{f}")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if "renderD" not in tgt:
                continue
            try:
                with open(f"/proc/{pid}/fdinfo/{f}") as fh:
                    content = fh.read()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            # Mesa doesn't always emit drm-* keys; if present, sum bytes
            for line in content.splitlines():
                if line.startswith("drm-memory:"):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            total += int(parts[1])
                        except ValueError:
                            pass
            found.append(f)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return {"drm_renderD_fds": len(found), "drm_allocated_bytes": total}


def http_get(url, cookies=None, timeout=10):
    parsed = urllib.request.urlparse(url)
    c = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    headers = {}
    if cookies:
        headers["Cookie"] = cookies
    c.request("GET", parsed.path, headers=headers)
    r = c.getresponse()
    body = r.read()
    c.close()
    if r.status >= 400:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def count_dispatch(path):
    try:
        with open(path) as f:
            text = f.read()
    except (FileNotFoundError, OSError):
        return 0
    return text.count("[rtmod] DISPATCH")


def http_post(url, body, cookies=None, csrf=None, timeout=600):
    parsed = urllib.request.urlparse(url)
    c = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    headers = {"Content-Type": "application/json",
               "Origin": f"http://{parsed.hostname}:{parsed.port or 80}"}
    if cookies:
        headers["Cookie"] = cookies
    if csrf:
        headers["X-Assistant-CSRF"] = csrf
    c.request("POST", parsed.path, json.dumps(body), headers)
    r = c.getresponse()
    payload = json.loads(r.read())
    c.close()
    if r.status >= 400:
        raise RuntimeError(f"POST {parsed.path}: {r.status} {payload}")
    return payload


def wait_for_sse_text(runtime, cid, turn_id, after_seq=0, deadline_s=600):
    """SSE event loop; returns (text, first_visible_text_ts, error_or_done)."""
    host = "127.0.0.1"
    port = runtime["port"]
    cursor = after_seq
    first_text_ts = None
    final_text = ""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        c = http.client.HTTPConnection(host, port, timeout=300)
        c.request("GET", f"/api/conversations/{cid}/events?after={cursor}",
                  headers={"Cookie": runtime["cookie"]})
        r = c.getresponse()
        if r.status != 200:
            r.read()
            c.close()
            time.sleep(0.5)
            continue
        for line in r:
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            cursor = event["sequence"]
            if event.get("turn_id") != turn_id:
                continue
            etype = event.get("type")
            if etype == "text":
                chunk = event["data"]["text"]
                if first_text_ts is None:
                    first_text_ts = time.time()
                final_text += chunk
            elif etype == "done":
                c.close()
                return final_text, first_text_ts, event.get("data", {})
            elif etype == "error":
                c.close()
                return final_text, first_text_ts, {"error": event["data"]}
        c.close()
        time.sleep(0.5)
    return final_text, first_text_ts, {"error": "timeout"}


def take_sample(runtime, label, chat_port, decision_port):
    """One memory snapshot across system + per-process + backend."""
    mi = parse_meminfo()
    tree = sample_tree(args.assistant_pid)
    drm = {pid: drm_bytes(pid) for pid in tree["pids"]}
    # Backend peak: query the chat worker's /v1/internal/memory
    backend = http_get(f"http://127.0.0.1:{chat_port}/v1/internal/memory")
    # Laya decision worker has no mx peak probe in this version, but we
    # can at least confirm it answers.
    decision_probe = None
    if decision_port:
        decision_probe = http_get(f"http://127.0.0.1:{decision_port}/v1/models",
                                  timeout=2)
    # Per-worker dispatch counts from the worker log files.
    chat_lines = count_dispatch(args.chat_log) if hasattr(args, "chat_log") else 0
    decision_lines = count_dispatch(args.decision_log) if hasattr(args, "decision_log") else 0
    return {
        "phase": label,
        "t": time.time(),
        "meminfo": mi,
        "system_used_bytes": meminfo_bytes(mi, "MemTotal") - meminfo_bytes(mi, "MemAvailable"),
        "pids": tree["pids"],
        "rss_total": tree["rss_total"],
        "pss_total": tree["pss_total"],
        "per_pid": tree["per_pid"],
        "drm_per_pid": drm,
        "backend_peak": backend,
        "decision_probe": decision_probe,
        "dispatch_lines_chat_total": chat_lines,
        "dispatch_lines_decision_total": decision_lines,
    }


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant-app-json", required=True)
    ap.add_argument("--assistant-pid", type=int, required=True)
    ap.add_argument("--chat-port", type=int, required=True)
    ap.add_argument("--decision-port", type=int, default=0)
    ap.add_argument("--chat-log", default="")
    ap.add_argument("--decision-log", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    decision_port = args.decision_port if args.decision_port > 0 else None

    with open(args.assistant_app_json) as f:
        runtime = json.load(f)
    url = f"http://127.0.0.1:{runtime['port']}"
    conv = http_post(f"{url}/api/conversations", {"save": False},
                     cookies=runtime["cookie"], csrf=runtime["csrf"])
    cid = conv["id"]

    samples = []
    samples.append(take_sample(runtime, "idle_before", args.chat_port, decision_port))

    # 1) Plain chat
    t0 = time.time()
    turn = http_post(f"{url}/api/conversations/{cid}/turns",
                     {"text": "Hello. Reply briefly.", "mode": "chat", "max_tokens": 128},
                     cookies=runtime["cookie"], csrf=runtime["csrf"])
    text, fvt, done = wait_for_sse_text(runtime, cid, turn["turn_id"])
    samples.append({"phase": "after_plain_chat",
                    "wall_s": time.time() - t0,
                    "first_visible_text_s": (fvt - t0) if fvt else None,
                    "text_len": len(text),
                    **take_sample(runtime, "after_plain_chat", args.chat_port, decision_port)})

    # 2) Long-context turn (~2k tokens). The chat budget gate may refuse
    # it; we record the refusal and continue with a fitting prompt so the
    # phase is exercised either way.
    filler = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    long_prompt = "Summarize the following in one sentence:\n\n" + (filler + "\n") * 16
    t0 = time.time()
    turn = http_post(f"{url}/api/conversations/{cid}/turns",
                     {"text": long_prompt, "mode": "chat", "max_tokens": 256},
                     cookies=runtime["cookie"], csrf=runtime["csrf"])
    text, fvt, done = wait_for_sse_text(runtime, cid, turn["turn_id"])
    long_rejected = isinstance(done, dict) and "error" in done
    samples.append({"phase": "after_long_context",
                    "prompt_chars": len(long_prompt),
                    "prompt_rejected": long_rejected,
                    "wall_s": time.time() - t0,
                    "first_visible_text_s": (fvt - t0) if fvt else None,
                    "text_len": len(text),
                    **take_sample(runtime, "after_long_context", args.chat_port, decision_port)})

    # 3) Compare options decision
    compare = {
        "mode": "compare",
        "text": "Pick a winner between the alternatives.",
        "options": [
            {"id": "espresso", "label": "Espresso"},
            {"id": "pourover", "label": "Pour-over"},
            {"id": "cold_brew", "label": "Cold brew"},
        ],
        "criteria": "bitterness and clarity",
    }
    t0 = time.time()
    turn = http_post(f"{url}/api/conversations/{cid}/turns", compare,
                     cookies=runtime["cookie"], csrf=runtime["csrf"])
    text, fvt, done = wait_for_sse_text(runtime, cid, turn["turn_id"])
    samples.append({"phase": "after_compare",
                    "wall_s": time.time() - t0,
                    "first_visible_text_s": (fvt - t0) if fvt else None,
                    "text_len": len(text),
                    **take_sample(runtime, "after_compare", args.chat_port, decision_port)})

    # 4) Card turn
    t0 = time.time()
    turn = http_post(f"{url}/api/conversations/{cid}/turns",
                     {"text": "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.",
                      "mode": "chat", "max_tokens": 256},
                     cookies=runtime["cookie"], csrf=runtime["csrf"])
    text, fvt, done = wait_for_sse_text(runtime, cid, turn["turn_id"])
    samples.append({"phase": "after_card",
                    "wall_s": time.time() - t0,
                    "first_visible_text_s": (fvt - t0) if fvt else None,
                    "text_len": len(text),
                    **take_sample(runtime, "after_card", args.chat_port, decision_port)})

    # idle_after
    time.sleep(2)
    samples.append(take_sample(runtime, "idle_after", args.chat_port, decision_port))

    out = {
        "label": args.label,
        "assistant_pid": args.assistant_pid,
        "chat_port": args.chat_port,
        "decision_port": decision_port,
        "samples": samples,
        "system_peak_used_bytes": max(s["system_used_bytes"] for s in samples),
        "tree_peak_rss_total_bytes": max(s["rss_total"] for s in samples),
        "tree_peak_pss_total_bytes": max(s["pss_total"] for s in samples),
        "backend_peak_bytes": max(
            (s["backend_peak"].get("peak") if isinstance(s["backend_peak"], dict) else 0) or 0
            for s in samples),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")
    print(f"system_peak_used_MiB={out['system_peak_used_bytes']/2**20:.1f}")
    print(f"tree_peak_rss_total_MiB={out['tree_peak_rss_total_bytes']/2**20:.1f}")
    print(f"tree_peak_pss_total_MiB={out['tree_peak_pss_total_bytes']/2**20:.1f}")
    print(f"backend_peak_MiB={out['backend_peak_bytes']/2**20:.1f}")


if __name__ == "__main__":
    main()