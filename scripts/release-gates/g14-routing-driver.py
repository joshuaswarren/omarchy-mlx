#!/usr/bin/env python3
"""Gate 14 leg driver: wait for the assistant to come up on HOME, run one
structured-decision chat turn, and report which model answered.

Prints, one per line:
  STATUS_ROUTING <status routing field or absent>
  SETUP_COMPLETE <s>
  ANSWER <json: status, model, head_ms, text-head>
  MODEL <model-id>
The shell gate greps MODEL and STATUS_ROUTING.
"""
import json
import sys
import time
import urllib.error
import urllib.request

HOME, LOG = sys.argv[1], sys.argv[2]
RUNTIME = HOME + "/assistant/application.json"
PROMPT = ("Decide for me: tea or coffee? Answer with one word.")


def say(line):
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def wait_runtime(deadline_s=1200.0):
    started = time.monotonic()
    while True:
        try:
            return json.load(open(RUNTIME))
        except FileNotFoundError:
            if time.monotonic() - started > deadline_s:
                raise
            time.sleep(1)


def call(rt, method, path, body=None, timeout=300):
    base = f"http://127.0.0.1:{rt['port']}"
    req = urllib.request.Request(
        base + path, data=None if body is None else json.dumps(body).encode(),
        method=method, headers={"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
                                "Origin": base, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    rt = wait_runtime()
    t0 = time.monotonic()
    while True:
        try:
            st = call(rt, "GET", "/api/status")
            if (st.get("setup") or {}).get("state") == "complete":
                break
            if (st.get("setup") or {}).get("state") == "error":
                say("SETUP_ERROR " + str(st["setup"].get("message"))[:300])
                return 1
        except Exception as exc:
            say("STATUS_RETRY " + str(exc)[:120])
        if time.monotonic() - t0 > 1200:
            say("SETUP_TIMEOUT")
            return 1
        time.sleep(1)
    say(f"SETUP_COMPLETE {time.monotonic() - t0:.1f}s")

    st = call(rt, "GET", "/api/status")
    routing = (st.get("routing")
               or (st.get("active_pair") or {}).get("routing")
               or "absent")
    say("STATUS_ROUTING " + json.dumps(routing)[:300])

    cid = call(rt, "POST", "/api/conversations", {"save": False})["id"]
    t0 = time.monotonic()
    tid = call(rt, "POST", f"/api/conversations/{cid}/turns",
               {"text": PROMPT, "mode": "chat", "max_tokens": 400})["turn_id"]
    while time.monotonic() - t0 < 600:
        rec = call(rt, "GET", f"/api/conversations/{cid}")
        m = next((m for m in rec.get("messages") or []
                  if m.get("turn_id") == tid and m.get("role") == "assistant"), None)
        if m and m.get("status") in ("complete", "error", "stopped"):
            d = m.get("decision") or {}
            timing = m.get("timing") or d.get("timing") or {}
            model = (d.get("model") or m.get("model") or "")
            head_ms = timing.get("head_ms", timing.get("head", ""))
            say("ANSWER " + json.dumps({
                "status": m.get("status"), "model": model,
                "head_ms": head_ms, "text_head": str(m.get("content"))[:60]}))
            print(f"MODEL {model or 'unrecorded'}")
            return 0
        time.sleep(1)
    say("TURN_TIMEOUT")
    print("MODEL unrecorded")
    return 1


if __name__ == "__main__":
    sys.exit(main())
