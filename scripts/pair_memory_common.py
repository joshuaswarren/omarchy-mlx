#!/usr/bin/env python3
"""Shared plumbing for the PairGates memory harnesses.

pair_memory_turns.py (assistant-API scenario driver) and pair_memory_v2.py
(per-phase asserted harness) copy-evolved from the same original; the
/proc + HTTP measurement plumbing lives here so it exists once.

Behavior flags keep each harness's exact sample shape:
- drm_allocated: turns parses fdinfo drm-memory totals; v2 reports FD
  counts only.
- include_stub_pid: turns keeps the pid-0 placeholder in per_pid /
  drm_per_pid for pre-assistant baselines; v2 drops it.
"""
import http.client
import json
import os
import subprocess
import time
import urllib.request


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


def children_of(pid):
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


def rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def pss_kb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def drm_bytes(pid, allocated=False):
    total = 0
    found = []
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                tgt = os.readlink(f"/proc/{pid}/fd/{fd}")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if "renderD" not in tgt:
                continue
            found.append(fd)
            if allocated:
                try:
                    with open(f"/proc/{pid}/fdinfo/{fd}") as fh:
                        for line in fh:
                            if line.startswith("drm-memory:"):
                                parts = line.split()
                                if len(parts) >= 3:
                                    try:
                                        total += int(parts[1])
                                    except ValueError:
                                        pass
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return {"drm_renderD_fds": len(found), "drm_allocated_bytes": total}


def backend_peak(chat_port):
    try:
        conn = http.client.HTTPConnection("127.0.0.1", chat_port, timeout=5)
        conn.request("GET", "/v1/internal/memory")
        r = conn.getresponse()
        body = r.read()
        conn.close()
        return json.loads(body)
    except Exception as e:
        return {"error": str(e)}


def sample(label, assistant_pid, chat_port, *, drm_allocated=False,
           include_stub_pid=False):
    mi = parse_meminfo()
    pids = children_of(assistant_pid) if assistant_pid else {0}
    per_pid = {}
    rss_total = 0
    pss_total = 0
    for p in pids:
        if p == 0 and not include_stub_pid:
            continue
        r = rss_kb(p)
        s = pss_kb(p)
        if r is not None:
            rss_total += r
        if s is not None:
            pss_total += s
        per_pid[p] = {"rss": r, "pss": s}
    return {"phase": label, "t": time.time(),
            "uname": subprocess.check_output(["uname", "-r"]).decode().strip(),
            "meminfo": {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "MemFree")},
            "system_used_bytes": meminfo_bytes(mi, "MemTotal") - meminfo_bytes(mi, "MemAvailable"),
            "pids": sorted(pids), "rss_total": rss_total, "pss_total": pss_total,
            "per_pid": per_pid,
            "drm_per_pid": {p: drm_bytes(p, allocated=drm_allocated)
                            for p in pids if include_stub_pid or p > 0},
            "backend_peak": backend_peak(chat_port) if chat_port else None}


def call(method, path, body=None, runtime_path=None):
    rt = json.load(open(runtime_path))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
               "Origin": base, "Content-Type": "application/json"}
    req = urllib.request.Request(base + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode())


def capture_baseline_top_rss(rss_dump_path):
    """Sample the pre-assistant baseline and dump the top-15 >50 MiB
    processes for contamination visibility. Returns the baseline sample."""
    baseline = sample("baseline_before_assistant", 0, None)
    proc_rss = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        r = rss_kb(int(d))
        if r is not None and r > 50 * 1024 * 1024:
            try:
                with open(f"/proc/{d}/cmdline") as f:
                    cmd = f.read().replace("\0", " ").strip()[:160]
            except Exception:
                cmd = "?"
            proc_rss.append({"pid": int(d), "rss_bytes": r, "cmd": cmd})
    proc_rss.sort(key=lambda x: -x["rss_bytes"])
    os.makedirs(os.path.dirname(rss_dump_path) or ".", exist_ok=True)
    with open(rss_dump_path, "w") as f:
        json.dump(proc_rss[:15], f, indent=2)
    return baseline


def wait_for_setup(pair, runtime_path, do_setup, *, fail_on_timeout=True):
    """Poll /api/setup to completion (absent state restarts into setup with
    a fresh deadline). Returns the last setup state. fail_on_timeout picks
    the harness behavior: turns raises, v2 falls through."""
    deadline = time.monotonic() + (1500 if do_setup else 180)
    state = {}
    while time.monotonic() < deadline:
        state = call("GET", "/api/status", runtime_path=runtime_path).get("setup") or {}
        if state.get("state") == "complete":
            return state
        if state.get("state") == "error":
            pair.kill()
            raise RuntimeError(f"setup error: {state}")
        if state.get("state") == "absent":
            pair.kill()
            pair.start(do_setup=True)
            deadline = time.monotonic() + 1500
        time.sleep(1)
    if fail_on_timeout:
        raise RuntimeError("setup did not complete in time")
    return state


def find_chat_port(home, pair_id, timeout_s=120):
    """Read the LAST "Starting httpd at" line from the pair's chat worker
    log (each restart appends; only the last one is the live process).
    Returns the port string, or None on timeout."""
    chat_log = os.path.join(home, "assistant", "logs", f"{pair_id}-chat.log")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(chat_log):
            last = None
            for line in open(chat_log, errors="replace"):
                if "Starting httpd at" in line:
                    last = line
            if last:
                return last.rsplit(" ", 1)[-1].strip().rstrip(".")
        time.sleep(1)
    return None


def read_context_tokens(home, pair_id):
    """The admitted ceiling the pair actually runs, from its pair lock."""
    try:
        lock = json.load(open(os.path.join(
            home, "assistant", "pair-locks", f"{pair_id}.json")))
        return lock.get("context_tokens")
    except Exception:
        return None
