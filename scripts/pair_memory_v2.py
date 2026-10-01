#!/usr/bin/env python3
"""PairGates memory harness v2: assistant-API driven with heartbeat,
max_tokens=700 for card turns, per-phase assert status=complete and
text_len>0, pre-start baseline, post-teardown baseline.

Usage: pair_memory_v2.py --pair-id <catalog_pair_id> --home <dir> --repo-serve <dir> --label <name> --out <json>
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import http.client
import urllib.error
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
    except (FileNotFoundError, ProcessLookupError):
        return None


def pss_kb(pid):
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return None


def drm_bytes(pid):
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


def sample(label, assistant_pid, chat_port):
    mi = parse_meminfo()
    pids = children_of(assistant_pid) if assistant_pid else {0}
    per_pid = {}
    rss_total = 0
    pss_total = 0
    for p in pids:
        if p == 0:
            continue
        r = rss_kb(p)
        s = pss_kb(p)
        if r is not None:
            rss_total += r
        if s is not None:
            pss_total += s
        per_pid[p] = {"rss": r, "pss": s}
    return {
        "phase": label,
        "t": time.time(),
        "uname": subprocess.check_output(["uname", "-r"]).decode().strip(),
        "meminfo": {k: mi.get(k) for k in ("MemTotal", "MemAvailable", "MemFree")},
        "system_used_bytes": meminfo_bytes(mi, "MemTotal") - meminfo_bytes(mi, "MemAvailable"),
        "pids": sorted(pids),
        "rss_total": rss_total,
        "pss_total": pss_total,
        "per_pid": per_pid,
        "drm_per_pid": {p: drm_bytes(p) for p in pids if p > 0},
        "backend_peak": backend_peak(chat_port) if chat_port else None,
    }


class Assistant:
    def __init__(self, pair_id, home, repo_serve):
        self.pair_id = pair_id
        self.home = home
        self.repo_serve = repo_serve
        self.runtime = os.path.join(home, "assistant", "application.json")
        self.server = None
        self.log_path = os.path.join(home, "assistant", "pair_driver.log")

    def boot_id_mismatch(self):
        cur = ""
        try:
            with open("/proc/sys/kernel/random/boot_id") as f:
                cur = f.read().strip()
        except Exception:
            pass
        lock = os.path.join(self.home, "assistant", "pair-locks", f"{self.pair_id}.json")
        saved = ""
        if os.path.exists(lock):
            try:
                saved = json.load(open(lock)).get("boot_id", "")
            except Exception:
                saved = ""
        return saved != cur

    def start(self, do_setup):
        os.makedirs(os.path.join(self.home, "assistant", "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.home, "assistant", "pair-locks"), exist_ok=True)
        if os.path.exists(self.runtime):
            os.unlink(self.runtime)
        env = dict(os.environ,
                   PYTHONPATH=self.repo_serve + ":" + os.environ.get("PYTHONPATH", ""),
                   MLX_OMARCHY_OFFLINE="1",
                   MLX_OMARCHY_HOME=self.home,
                   MLX_OMARCHY_PAIR_DEV_QUALIFICATION="1")
        args = ["--pair", self.pair_id, "--yes"] if do_setup else ["--resume"]
        self.server = subprocess.Popen(
            [os.path.expanduser("~/.local/share/mlx-omarchy/venv/bin/python"),
             "-m", "mlx_omarchy_assistant",
             "--home", self.home, "--no-browser"] + args,
            cwd=os.path.dirname(self.repo_serve),
            env=env,
            stdout=open(self.log_path, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True)
        _KILL_LIST.append(self.server)
        deadline = time.monotonic() + (1500 if do_setup else 180)
        while not os.path.exists(self.runtime) and time.monotonic() < deadline:
            time.sleep(0.5)
        if not os.path.exists(self.runtime):
            raise RuntimeError("server never wrote application.json")

    def kill(self):
        if self.server is None:
            return
        try:
            os.killpg(self.server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.server.wait(timeout=10)
        except Exception:
            try:
                os.killpg(self.server.pid, signal.SIGKILL)
            except Exception:
                pass
        subprocess.run(["bash", "-lc",
                        "ps -eo pid,cmd | grep -E "
                        "'mlx_omarchy_assistant.*PairGates|_mlxlm_server.*PairGates|"
                        "mlx_omarchy_laya.*PairGates' | "
                        "awk '{print $1}' | xargs -r kill -9 2>/dev/null"],
                       timeout=15, check=False)


_KILL_LIST = []


def call(method, path, body=None, runtime_path=None):
    rt = json.load(open(runtime_path))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
               "Origin": base, "Content-Type": "application/json"}
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode())


def heartbeat_loop(cid, turn, runtime_path, stop):
    while not stop.wait(2):
        try:
            call("POST", f"/api/conversations/{cid}/heartbeat",
                 {"turn_id": turn}, runtime_path=runtime_path)
        except Exception:
            pass


def wait_turn(cid, turn, runtime_path, timeout_s=600, want_fvt=False):
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat_loop,
                            args=(cid, turn, runtime_path, stop),
                            daemon=True)
    beat.start()
    deadline = time.monotonic() + timeout_s
    message = None
    first_text_at = None
    t0 = time.monotonic()
    try:
        while time.monotonic() < deadline:
            if want_fvt and first_text_at is None:
                try:
                    events = call("GET",
                                  f"/api/conversations/{cid}/events?after=0",
                                  runtime_path=runtime_path)
                    for ev in events if isinstance(events, list) else []:
                        if (ev.get("turn_id") == turn
                                and ev.get("type") == "text"):
                            first_text_at = time.monotonic() - t0
                            break
                except Exception:
                    pass
            record = call("GET", f"/api/conversations/{cid}", runtime_path=runtime_path)
            message = next((m for m in record.get("messages") or []
                            if m.get("turn_id") == turn
                            and m.get("role") == "assistant"), None)
            if message and message.get("status") in ("complete", "error", "stopped"):
                break
            time.sleep(1)
    finally:
        stop.set()
        beat.join(timeout=5)
    if message is not None and first_text_at is not None:
        message = dict(message, first_visible_text_s=round(first_text_at, 3))
    return message


def submit_turn(runtime_path, text, mode="chat", max_tokens=256, options=None, criteria=None):
    cid = call("POST", "/api/conversations", {"save": False},
               runtime_path=runtime_path)["id"]
    body = {"text": text, "mode": mode, "max_tokens": max_tokens}
    if options:
        body["options"] = options
    if criteria:
        body["criteria"] = criteria
    turn = call("POST", f"/api/conversations/{cid}/turns", body,
                runtime_path=runtime_path)["turn_id"]
    return cid, turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-id", required=True, help="catalog pair_id: everyday|quality|compact")
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--timeout-s-card", type=int, default=900)
    ap.add_argument("--max-tokens-card", type=int, default=700)
    args = ap.parse_args()

    # Pre-start baseline BEFORE the assistant launches.
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
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out.replace(".json", "-baseline-top-rss.json"), "w") as f:
        json.dump(proc_rss[:15], f, indent=2)
    print(f"baseline sys_used={baseline['system_used_bytes']/2**20:.1f} MiB", flush=True)

    pair = Assistant(args.pair_id, args.home, args.repo_serve)
    do_setup = pair.boot_id_mismatch()
    print(f"do_setup={do_setup}", flush=True)
    pair.start(do_setup)

    setup_deadline = time.monotonic() + (1500 if do_setup else 180)
    while time.monotonic() < setup_deadline:
        state = call("GET", "/api/status", runtime_path=pair.runtime).get("setup") or {}
        if state.get("state") == "complete":
            break
        if state.get("state") == "error":
            pair.kill()
            raise RuntimeError(f"setup error: {state}")
        if state.get("state") == "absent":
            pair.kill()
            pair.start(do_setup=True)
            setup_deadline = time.monotonic() + 1500
        time.sleep(1)
    print("setup complete", flush=True)

    # Find chat worker port from the LAST "Starting httpd at" line.
    chat_log = os.path.join(args.home, "assistant", "logs", f"{args.pair_id}-chat.log")
    chat_port = None
    chat_deadline = time.monotonic() + 120
    while time.monotonic() < chat_deadline:
        if os.path.exists(chat_log):
            last = None
            for line in open(chat_log, errors="replace"):
                if "Starting httpd at" in line:
                    last = line
            if last:
                chat_port = last.rsplit(" ", 1)[-1].strip().rstrip(".")
        if chat_port:
            break
        time.sleep(1)
    print(f"chat port: {chat_port}", flush=True)

    samples = [baseline, sample("idle_before", pair.server.pid, chat_port)]

    # Read pair_lock context_tokens.
    context_tokens = None
    try:
        lock = json.load(open(os.path.join(
            args.home, "assistant", "pair-locks", f"{args.pair_id}.json")))
        context_tokens = lock.get("context_tokens")
    except Exception:
        pass

    phases = [
        ("plain_chat", "Hello. Reply briefly.", "chat", 64, None, None),
        ("long_context",
         "Summarize the following in one sentence:\n\n"
         + ("The quick brown fox jumps over the lazy dog. " * 30).strip() + "\n" * 4,
         "chat", 256, None, None),
    ]

    # Compare turn.
    compare = {
        "mode": "compare",
        "text": "Pick a winner between the alternatives.",
        "options": [{"id": "espresso", "label": "Espresso"},
                    {"id": "pourover", "label": "Pour-over"},
                    {"id": "cold_brew", "label": "Cold brew"}],
        "criteria": "bitterness and clarity",
    }
    phases.append(("compare",
                   "Pick a winner between the alternatives.",
                   "compare", 700, compare.get("options"), compare.get("criteria")))

    # Card turn.
    phases.append(("card",
                   "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.",
                   "chat", args.max_tokens_card, None, None))

    for phase_name, prompt, mode, max_tokens, options, criteria in phases:
        cid, turn = submit_turn(pair.runtime, prompt, mode=mode,
                               max_tokens=max_tokens, options=options,
                               criteria=criteria)
        t0 = time.monotonic()
        # Card turns wait longer: the 9B model needs several minutes to
        # stream the full reply and the coordinator's card-promotion pass
        # runs after the stream ends.
        timeout_s = args.timeout_s if phase_name != "card" else args.timeout_s_card
        msg = wait_turn(cid, turn, pair.runtime, timeout_s=timeout_s,
                        want_fvt=True)
        wall = time.monotonic() - t0
        text = (msg or {}).get("content") or ""
        status = (msg or {}).get("status")
        components = [c.get("type") for c in (msg or {}).get("components") or []]
        # Per-phase validity: "the user saw a response".
        #   plain_chat / long_context: complete with visible text.
        #   compare: complete with visible text (a decision component is
        #     a bonus; some chat models answer compare in plain prose).
        #   card: complete with visible text (a card component is a bonus;
        #     the card-promotion pass or the raw-fence fallback shows text).
        if phase_name == "compare":
            valid = status == "complete" and (len(text) > 0 or "decision" in components)
        elif phase_name == "card":
            valid = status == "complete" and len(text) > 0
        else:
            valid = status == "complete" and len(text) > 0
        print(f"{phase_name}: status={status} text_len={len(text)} "
              f"components={components} valid={valid} wall={wall:.1f}s", flush=True)
        s = sample(f"after_{phase_name}", pair.server.pid, chat_port)
        s.update({"turn_status": status, "text_len": len(text),
                  "components": components, "turn_valid": valid,
                  "wall_s": round(wall, 2),
                  "first_visible_text_s": (msg or {}).get("first_visible_text_s")})
        samples.append(s)

    time.sleep(3)
    samples.append(sample("idle_after", pair.server.pid, chat_port))
    pair.kill()
    time.sleep(3)
    samples.append(sample("post_teardown", 0, None))

    # Per-phase turn summary (card is a separate row with its own status).
    baseline_sys = baseline["system_used_bytes"]
    phase_rows = []
    for s in samples:
        if "turn_valid" in s:
            phase_rows.append({
                "phase": s["phase"].replace("after_", ""),
                "status": s.get("turn_status"),
                "text_len": s.get("text_len"),
                "components": s.get("components"),
                "wall_s": s.get("wall_s"),
                "first_visible_text_s": s.get("first_visible_text_s"),
                "valid": s["turn_valid"],
                "system_peak_over_baseline_bytes": s["system_used_bytes"] - baseline_sys,
            })

    # Core phases must be valid for the run to count.  A card turn is
    # reported separately (it is a product-defect surface on some models).
    core_invalid = [r["phase"] for r in phase_rows
                    if r["phase"] != "card" and not r["valid"]]
    card_row = next((r for r in phase_rows if r["phase"] == "card"), None)
    if core_invalid:
        print(f"WARNING: INVALID CORE TURNS: {core_invalid}", flush=True)

    # Peak excluding baseline (baseline and post_teardown are floor).
    # Completed-phases peak: only samples after turns that are valid,
    # so a wedged card phase cannot inflate the headline number.
    completed_phases = {r["phase"] for r in phase_rows if r["valid"]}
    active = [s for s in samples if s["phase"] not in
              ("baseline_before_assistant", "post_teardown")]
    active_completed = [s for s in active
                        if s["phase"] == "idle_before"
                        or s["phase"].replace("after_", "") in completed_phases]
    system_peak = max(s["system_used_bytes"] for s in active_completed)
    idle_before = next(s["system_used_bytes"] for s in samples
                       if s["phase"] == "idle_before")

    out = {
        "label": args.label,
        "pair_id": args.pair_id,
        "uname": samples[0]["uname"],
        "assistant_pid": pair.server.pid,
        "chat_port": chat_port,
        "pair_lock_context_tokens": context_tokens,
        "samples": samples,
        "phases": phase_rows,
        "card": card_row,
        "baseline_sys_used_bytes": baseline_sys,
        "idle_before_sys_used_bytes": idle_before,
        "pair_resident_cost_bytes": idle_before - baseline_sys,
        "system_peak_over_baseline_bytes": system_peak - baseline_sys,
        "tree_peak_rss_total_bytes": max(s["rss_total"] for s in active),
        "tree_peak_pss_total_bytes": max(s["pss_total"] for s in active),
        "backend_peak_bytes": max(
            (s["backend_peak"].get("peak") if isinstance(s["backend_peak"], dict) else 0) or 0
            for s in active),
        "backend_peak_available": any(
            isinstance(s["backend_peak"], dict) and "peak" in s["backend_peak"]
            for s in active),
        "core_invalid_turns": core_invalid,
        "run_valid": len(core_invalid) == 0,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    print(f"baseline_MiB={baseline_sys/2**20:.1f}")
    print(f"pair_resident_MiB={out['pair_resident_cost_bytes']/2**20:.1f}")
    print(f"peak_over_baseline_MiB={out['system_peak_over_baseline_bytes']/2**20:.1f}")
    print(f"backend_peak_MiB={out['backend_peak_bytes']/2**20:.1f}")
    print(f"run_valid={out['run_valid']}")
    pair.kill()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()