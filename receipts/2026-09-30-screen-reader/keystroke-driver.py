#!/usr/bin/env python3
"""Drive each scenario and capture Orca's actual speech utterances.

Each scenario uses Tab-based keyboard navigation. The driver sets the
stub state, reloads the page, waits for ready, then runs a keystroke
script that tab-walks the controls. Orca's debug-file log captures every
SPEECH OUTPUT line emitted during the scenario; we extract those and
save them verbatim per scenario.

This driver does NOT exercise Orca's web-mode keybindings (Insert+;,
Insert+H, etc.) because the Orca bookworm-arm64 package lacks the web
script — those commands are no-ops. We rely on Tab navigation (which
Orca's default keymap handles correctly) and on Chromium's AX tree
captured separately to verify the semantics.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import threading
import asyncio
import websockets

LOG = Path("/<home>/sr/logs/orca-debug.log")
SCENARIOS_DIR = Path("/<home>/sr/logs/scenarios")
MARKER_FILE = Path("/<home>/sr/logs/_scenario_marker.txt")
MARKER = "SCENARIO_START_"


def get_target():
    raw = urllib.request.urlopen("http://127.0.0.1:9222/json/list", timeout=10).read()
    for t in json.loads(raw):
        if t.get("type") == "page" and "MLX Chat" in (t.get("title") or ""):
            return t
    raise SystemExit("no target")


def cdp_call(method, params=None):
    out = [None]
    def runner():
        async def go():
            target = get_target()
            async with websockets.connect(target["webSocketDebuggerUrl"],
                                            max_size=64 * 1024 * 1024) as ws:
                await ws.send(json.dumps({"id": 1, "method": method,
                                             "params": params or {}}))
                r = await ws.recv()
                out[0] = json.loads(r)
        asyncio.run(go())
    t = threading.Thread(target=runner); t.start(); t.join(timeout=15)
    return out[0]


def cdp_eval(expr):
    out = [None]
    def runner():
        async def go():
            target = get_target()
            async with websockets.connect(target["webSocketDebuggerUrl"],
                                            max_size=64 * 1024 * 1024) as ws:
                await ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate",
                                             "params": {"expression": expr,
                                                         "returnByValue": True}}))
                r = await ws.recv()
                j = json.loads(r)
                out[0] = j.get("result", {}).get("result", {}).get("value", "")
        asyncio.run(go())
    t = threading.Thread(target=runner); t.start(); t.join(timeout=15)
    return out[0]


def set_state(name):
    urllib.request.urlopen(
        f"http://127.0.0.1:8765/api/__state?set={name}", timeout=5).read()


def reload_page():
    # Cache-bust by navigating with a unique query string.
    import time as _t
    cdp_eval(f"window.location.href = '/index.html?cb={int(_t.time())}'")
    time.sleep(0.5)


def reset_orca_focus():
    """Click an empty area to reset Orca's locus of focus to the document."""
    cdp_eval("document.body.click(); document.body.focus()")
    time.sleep(0.3)


def wait_for_settled(timeout_s=15):
    """Wait until the page has fully re-rendered the new state.
    We poll for: view-loading hidden AND view has more than 1 child.
    Then add a 2s grace period for JS-driven renders (decision/card)."""
    end = time.time() + timeout_s
    last_html = ""
    while time.time() < end:
        html = cdp_eval("document.getElementById('view')?.innerHTML.length || 0")
        loading = cdp_eval("document.getElementById('view-loading') ? document.getElementById('view-loading').hidden : true")
        if loading and html and html != last_html and html > 100:
            time.sleep(2.0)
            return True
        last_html = html
        time.sleep(0.5)
    return False


def find_chromium_window():
    out = subprocess.run(["xdotool", "search", "--name", "MLX Chat"],
                          env={"DISPLAY": ":99", "PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True)
    ids = [int(x) for x in out.stdout.split() if x.strip()]
    if not ids:
        out = subprocess.run(["xdotool", "search", "--classname", "chromium"],
                              env={"DISPLAY": ":99", "PATH": "/usr/bin:/bin"},
                              capture_output=True, text=True)
        ids = [int(x) for x in out.stdout.split() if x.strip()]
    return ids


def key(*keys):
    ids = find_chromium_window()
    if not ids: return False
    cmd = ["xdotool", "key", "--window", str(ids[0])] + list(keys)
    subprocess.run(cmd, env={"DISPLAY": ":99", "PATH": "/usr/bin:/bin"},
                    capture_output=True)
    return True


def type_text(text):
    ids = find_chromium_window()
    if not ids: return False
    cmd = ["xdotool", "type", "--window", str(ids[0]), "--delay", "20", "--", text]
    subprocess.run(cmd, env={"DISPLAY": ":99", "PATH": "/usr/bin:/bin"},
                    capture_output=True)
    return True


def tab(n=1, shift=False):
    ids = find_chromium_window()
    if not ids: return False
    seq = (["shift+Tab"] if shift else ["Tab"]) * n
    cmd = ["xdotool", "key", "--window", str(ids[0])] + seq
    subprocess.run(cmd, env={"DISPLAY": ":99", "PATH": "/usr/bin:/bin"},
                    capture_output=True)
    return True


def write_marker():
    with MARKER_FILE.open("w") as f:
        f.write(f"{MARKER}{time.time()}\n")


def read_utterances():
    if not LOG.exists() or not MARKER_FILE.exists(): return []
    marker_ts = MARKER_FILE.stat().st_mtime
    out = []
    with LOG.open() as f:
        for line in f:
            m = re.match(r"^(\d+):(\d+):(\d+)\.(\d+)", line)
            if not m: continue
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            today = time.gmtime(marker_ts)
            log_ts = time.mktime((today.tm_year, today.tm_mon, today.tm_mday,
                                   h, mn, s, 0, 0, 0))
            if log_ts > marker_ts + 86400 / 2: log_ts -= 86400
            elif log_ts < marker_ts - 86400 / 2: log_ts += 86400
            if log_ts < marker_ts: continue
            sm = re.search(r"SPEECH OUTPUT:\s+'((?:[^'\\]|\\.)*)'", line)
            if sm: out.append((line[:15], sm.group(1)))
    return out


def run(name, keystrokes, label, state_name=None, duration=8):
    """Run a single scenario end-to-end."""
    print(f"=== {name} ({label}) ===")
    sd = SCENARIOS_DIR / name
    sd.mkdir(parents=True, exist_ok=True)

    if state_name is not None:
        set_state(state_name)
        reload_page()
        time.sleep(3)
        wait_for_settled(15)
        reset_orca_focus()
    else:
        # Default: ensure we're in the 'ready' state
        set_state("ready")
        reload_page()
        time.sleep(3)
        wait_for_settled(15)
        reset_orca_focus()

    log_before = LOG.stat().st_size if LOG.exists() else 0
    write_marker()

    keystrokes()
    time.sleep(duration)

    log_after = LOG.stat().st_size if LOG.exists() else 0

    utterances = read_utterances()
    if LOG.exists():
        with LOG.open() as f:
            f.seek(log_before)
            data = f.read(log_after - log_before)
        (sd / "orca-raw.txt").write_text(data)
    with (sd / "orca-utterances.txt").open("w") as f:
        f.write(f"# scenario: {name} ({label})\n")
        f.write(f"# total utterances: {len(utterances)}\n\n")
        for ts, txt in utterances:
            f.write(f"[{ts}] {txt}\n")
    print(f"  {len(utterances)} utterances captured")
    return utterances


# === Scenarios ===

def scenario_setup():
    """Tab through radios + checkbox + Start button."""
    def ks():
        # Move focus into the page
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        # Walk the focus chain
        for _ in range(15):
            tab(); time.sleep(0.2)
    ks_name = "setup"
    run(ks_name, ks, "setup: Tab through all controls",
        duration=8)


def scenario_chat_stream():
    """Type a message, send, watch stream. Count utterances during stream."""
    def ks():
        cdp_eval("document.getElementById('composer-text').focus()")
        time.sleep(0.3)
        type_text("Reply with one short sentence.")
        time.sleep(0.3)
        key("Return")
    # The chat stream sends 200 tokens; we wait long enough for it.
    run("chat-stream", ks, "chat: send message, observe polite announcements",
        duration=12)


def scenario_chat_escape():
    """Type a message, send, Escape stops."""
    def ks():
        cdp_eval("document.getElementById('composer-text').focus()")
        time.sleep(0.3)
        type_text("Long test message")
        time.sleep(0.3)
        key("Return")
        time.sleep(4)
        key("Escape")
    run("chat-escape", ks, "chat: Escape stops response",
        duration=10)


def scenario_decision():
    """Decision card present. Tab into it, inspect."""
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(10):
            tab(); time.sleep(0.2)
    run("decision", ks, "decision: card reading order",
        duration=8)


def scenario_card():
    """Checklist card present."""
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(10):
            tab(); time.sleep(0.2)
    run("card", ks, "checklist card: Tab through items",
        duration=8)


def scenario_compare():
    """Open Compare panel, submit empty (validation)."""
    def ks():
        cdp_eval("document.getElementById('compare-btn').click()")
        time.sleep(0.5)
        # Tab to find Submit
        for _ in range(8):
            tab(); time.sleep(0.2)
        cdp_eval("document.getElementById('compare-submit').click()")
        time.sleep(0.5)
    run("compare", ks, "compare: open, submit empty, expect validation",
        duration=8)


def scenario_drawer():
    """History drawer focus trap + Escape + Details drawer."""
    def ks():
        # History
        cdp_eval("document.getElementById('history-btn').click()")
        time.sleep(0.5)
        tab(); time.sleep(0.3)
        tab(); time.sleep(0.3)
        tab(); time.sleep(0.3)
        key("Escape"); time.sleep(0.5)
        # Details
        cdp_eval("document.getElementById('details-btn').click()")
        time.sleep(0.5)
        tab(); time.sleep(0.3)
        tab(); time.sleep(0.3)
        key("Escape"); time.sleep(0.5)
    run("drawer", ks, "drawers: focus trap, Escape, focus return",
        duration=10)


def scenario_voice():
    """Voice unavailable, mic disabled."""
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(15):
            tab(); time.sleep(0.2)
    run("voice", ks, "voice: Tab to disabled microphone",
        duration=8)


def scenario_error():
    """Error banner present. Tab to topbar."""
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(5):
            tab(); time.sleep(0.3)
    run("error", ks, "error: Tab to error banner",
        duration=6)


def scenario_ready_offline_chip():
    """Ready offline chip in topbar."""
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(5):
            tab(); time.sleep(0.3)
    run("ready-offline-chip", ks, "ready offline chip: Tab through topbar",
        duration=6)


SCENARIO_SPEC = [
    ("setup", "setup", "setup: Tab through all controls"),
    ("chat-stream", "ready", "chat: send message, observe polite announcements"),
    ("chat-escape", "ready", "chat: Escape stops response"),
    ("decision", "decision", "decision: card reading order"),
    ("card", "card", "checklist card: Tab through items"),
    ("compare", "ready", "compare: open, submit empty, expect validation"),
    ("drawer", "ready", "drawers: focus trap, Escape, focus return"),
    ("voice", "voice-unqualified", "voice: Tab to disabled microphone"),
    ("error", "error", "error: Tab to error banner"),
    ("ready-offline-chip", "ready-offline-true", "ready offline chip: Tab through topbar"),
]


def _scenario_runner(scenario_fn, state_name):
    def go():
        scenario_fn(state_name)
    return go


def scenario_setup(state):
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(15):
            tab(); time.sleep(0.2)
    run("setup", ks, "setup: Tab through all controls",
        state_name=state, duration=8)


def scenario_chat_stream(state):
    def ks():
        cdp_eval("document.getElementById('composer-text').focus()")
        time.sleep(0.3)
        type_text("Reply with one short sentence.")
        time.sleep(0.3)
        key("Return")
    run("chat-stream", ks, "chat: send message, observe polite announcements",
        state_name=state, duration=12)


def scenario_chat_escape(state):
    def ks():
        cdp_eval("document.getElementById('composer-text').focus()")
        time.sleep(0.3)
        type_text("Long test message")
        time.sleep(0.3)
        key("Return")
        time.sleep(4)
        key("Escape")
    run("chat-escape", ks, "chat: Escape stops response",
        state_name=state, duration=10)


def scenario_decision(state):
    def ks():
        # Walk focus across the chat log so Orca reads the assistant
        # bubble, decision card options, the selected state, criteria,
        # and the "How this was decided" disclosure.
        cdp_eval(
            "const log = document.querySelector('[role=log]'); if (log) log.click();")
        time.sleep(0.3)
        # Click directly into the assistant bubble.
        cdp_eval(
            "const b = document.querySelectorAll('[role=log] article')[1];"
            "if (b) { const r = b.getBoundingClientRect();"
            "  const e = document.elementFromPoint(r.left + r.width/2, r.top + 10);"
            "  if (e && e.click) e.click(); }")
        time.sleep(0.5)
        # Tab forward to enter the decision card controls.
        for _ in range(15):
            tab(); time.sleep(0.2)
    run("decision", ks, "decision: card reading order",
        state_name=state, duration=14)


def scenario_card(state):
    def ks():
        cdp_eval(
            "const log = document.querySelector('[role=log]'); if (log) log.click();")
        time.sleep(0.3)
        # Focus the assistant bubble by tabbing past the user bubble.
        cdp_eval(
            "const b = document.querySelectorAll('[role=log] article')[1];"
            "if (b) { const r = b.getBoundingClientRect();"
            "  const e = document.elementFromPoint(r.left + r.width/2, r.top + 10);"
            "  if (e && e.click) e.click(); }")
        time.sleep(0.5)
        for _ in range(15):
            tab(); time.sleep(0.2)
    run("card", ks, "checklist card: Tab through items",
        state_name=state, duration=14)


def scenario_compare(state):
    def ks():
        cdp_eval("document.getElementById('compare-btn').click()")
        time.sleep(0.5)
        for _ in range(8):
            tab(); time.sleep(0.2)
        cdp_eval("document.getElementById('compare-submit').click()")
        time.sleep(0.5)
    run("compare", ks, "compare: open, submit empty, expect validation",
        state_name=state, duration=8)


def scenario_drawer(state):
    def ks():
        cdp_eval("document.getElementById('history-btn').click()")
        time.sleep(0.5)
        tab(); time.sleep(0.3)
        tab(); time.sleep(0.3)
        tab(); time.sleep(0.3)
        key("Escape"); time.sleep(0.5)
        cdp_eval("document.getElementById('details-btn').click()")
        time.sleep(0.5)
        tab(); time.sleep(0.3)
        tab(); time.sleep(0.3)
        key("Escape"); time.sleep(0.5)
    run("drawer", ks, "drawers: focus trap, Escape, focus return",
        state_name=state, duration=10)


def scenario_voice(state):
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(15):
            tab(); time.sleep(0.2)
    run("voice", ks, "voice: Tab to disabled microphone",
        state_name=state, duration=8)


def scenario_error(state):
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(5):
            tab(); time.sleep(0.3)
    run("error", ks, "error: Tab to error banner",
        state_name=state, duration=6)


def scenario_ready_offline_chip(state):
    def ks():
        cdp_eval("document.body.focus()")
        time.sleep(0.3)
        for _ in range(5):
            tab(); time.sleep(0.3)
    run("ready-offline-chip", ks, "ready offline chip: Tab through topbar",
        state_name=state, duration=6)


ALL = [
    ("setup", scenario_setup),
    ("chat-stream", scenario_chat_stream),
    ("chat-escape", scenario_chat_escape),
    ("decision", scenario_decision),
    ("card", scenario_card),
    ("compare", scenario_compare),
    ("drawer", scenario_drawer),
    ("voice", scenario_voice),
    ("error", scenario_error),
    ("ready-offline-chip", scenario_ready_offline_chip),
]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "one":
        name = sys.argv[2]
        for n, fn in ALL:
            if n == name:
                # Find the state name
                for sname, _l, _d in SCENARIO_SPEC:
                    if sname == name:
                        fn(sname)
                        break
                break
    else:
        for sname, fn in ALL:
            try:
                fn(sname)
            except Exception as e:
                print(f"err: {e}")
            time.sleep(2)