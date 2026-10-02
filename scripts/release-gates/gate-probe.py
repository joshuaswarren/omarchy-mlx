#!/usr/bin/env python3
"""v0.7.7 installed-from-release probe.

Usage: v077-gate-probe.py ASSISTANT_HOME LOG [cards]
Default: setup wait + one chat turn + one compare turn (PASS needs both).
Mode "cards": setup wait + one chat turn + one card-cue turn; the card turn
must return status=complete with NON-EMPTY visible text (the 4B empty-reply
defect class) and the turn must not end stopped.
"""
import json, os, socket, sys, time, urllib.error, urllib.request

HOME, LOG = sys.argv[1], sys.argv[2]
CARDS = len(sys.argv) > 3 and sys.argv[3] == "cards"
RUNTIME = os.path.join(HOME, "assistant", "application.json")

def say(text):
    print(text, flush=True)
    with open(LOG, "a") as fh:
        fh.write(text + "\n")

def denied():
    for addr in (("1.1.1.1", 443), ("140.82.112.3", 443)):
        try:
            socket.create_connection(addr, timeout=3).close()
            return False
        except OSError:
            pass
    return True

def call(method, path, body=None):
    started = time.monotonic()
    while True:
        try:
            rt = json.load(open(RUNTIME))
            base = f"http://127.0.0.1:{rt['port']}"
            req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                         method=method, headers={"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
                                                                 "Origin": base, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, FileNotFoundError, ConnectionError, json.JSONDecodeError):
            if time.monotonic() - started > 1200:
                raise
            time.sleep(0.5)

def turn(cid, payload):
    t0 = time.monotonic()
    tid = call("POST", f"/api/conversations/{cid}/turns", payload)["turn_id"]
    while time.monotonic() - t0 < 600:
        call("POST", f"/api/conversations/{cid}/heartbeat", {"turn_id": tid})
        rec = call("GET", f"/api/conversations/{cid}")
        m = next((m for m in rec.get("messages") or [] if m.get("turn_id") == tid and m.get("role") == "assistant"), None)
        if m and m.get("status") in ("complete", "error", "stopped"):
            return m, time.monotonic() - t0
        time.sleep(1)
    raise SystemExit("turn timed out")

say(f"OUTBOUND_DENIED {denied()}")
t0 = time.monotonic()
while True:
    s = call("GET", "/api/status")
    state = (s.get("setup") or {}).get("state")
    if state == "complete":
        break
    if state == "error":
        say("SETUP_ERROR " + str(s["setup"].get("message"))[:600])
        sys.exit(1)
    if time.monotonic() - t0 > 1200:
        say("SETUP_TIMEOUT")
        sys.exit(1)
    time.sleep(1)
say(f"SETUP_COMPLETE {time.monotonic() - t0:.1f}s pair={(s.get('active_pair') or {}).get('id')}")
cid = call("POST", "/api/conversations", {"save": False})["id"]
m, dt = turn(cid, {"text": "Reply with the single word ready.", "mode": "chat", "max_tokens": 16})
say(f"CHAT status={m.get('status')} {dt:.2f}s text={str(m.get('content'))[:60]!r}")
ok = m.get("status") == "complete" and str(m.get("content") or "").strip()
if CARDS:
    m2, dt2 = turn(cid, {"text": "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.",
                         "mode": "chat", "max_tokens": 700})
    txt = str(m2.get("content") or "")
    comps = m2.get("components") or []
    say(f"CARD status={m2.get('status')} {dt2:.2f}s text_len={len(txt)} components={len(comps)} "
        f"head={txt[:60]!r}")
    ok = ok and m2.get("status") == "complete" and len(txt) > 0
    say("CARD_CHECK " + ("PASS" if (m2.get("status") == "complete" and len(txt) > 0) else "FAIL"))
else:
    m3, dt3 = turn(cid, {"text": "Pick the shorter word.", "mode": "compare", "criteria": "shorter spelling",
                         "options": [{"id": "cat", "label": "cat"}, {"id": "elephant", "label": "elephant"}]})
    d = m3.get("decision") or {}
    say(f"COMPARE status={m3.get('status')} {dt3:.2f}s choice={d.get('choice')} p={d.get('probabilities')} model={d.get('model')}")
    ok = ok and m3.get("status") == "complete" and d.get("choice") == "cat"
say("GATE " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)