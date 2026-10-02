#!/usr/bin/env python3
"""Drive the voice-path scenarios under real Orca and capture utterances.

Adapted from receipts/2026-09-30-screen-reader/keystroke-driver.py (same
marker + SPEECH OUTPUT extraction; adds tab_until targeting, an AX/live
region snapshot per scenario, and Chromium relaunches for fake-mic
variants). Every scenario is keyboard-only: Tab, Space, Enter, Escape,
arrows. Orca 43.1 in this image ships scripts/web, so web-mode live
region announcements are expected in the capture.
"""
import asyncio
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import websockets

LOG = Path("/srv/logs/orca-debug.log")
SCENARIOS_DIR = Path("/srv/out/scenarios")
MARKER_FILE = Path("/srv/logs/_scenario_marker.txt")
MARKER = "SCENARIO_START_"
STUB = "http://127.0.0.1:8765"
ENV = {"DISPLAY": ":99", "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
       "HOME": "/tmp"}


def get_target():
    raw = urllib.request.urlopen("http://127.0.0.1:9222/json/list",
                                 timeout=10).read()
    for t in json.loads(raw):
        if t.get("type") == "page" and "MLX Chat" in (t.get("title") or ""):
            return t
    raise SystemExit("no target")


def cdp_eval(expr):
    out = [None]

    def runner():
        async def go():
            target = get_target()
            async with websockets.connect(target["webSocketDebuggerUrl"],
                                          max_size=64 * 1024 * 1024) as ws:
                await ws.send(json.dumps({
                    "id": 1, "method": "Runtime.evaluate",
                    "params": {"expression": expr, "returnByValue": True}}))
                r = await ws.recv()
                j = json.loads(r)
                out[0] = (j.get("result", {}).get("result", {}) or {}).get("value", "")
        asyncio.run(go())

    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=15)
    return out[0]


def set_state(name):
    urllib.request.urlopen(f"{STUB}/api/__state?set={name}", timeout=5).read()


def reload_page():
    cdp_eval(f"window.location.href = '/index.html?cb={int(time.time())}'")
    time.sleep(0.5)


def reset_orca_focus():
    cdp_eval("document.body.click(); document.body.focus()")
    time.sleep(0.3)


def wait_for_settled(timeout_s=15):
    end = time.time() + timeout_s
    last_html = ""
    while time.time() < end:
        html = cdp_eval("document.getElementById('view')?.innerHTML.length || 0")
        loading = cdp_eval(
            "document.getElementById('view-loading') ? "
            "document.getElementById('view-loading').hidden : true")
        if loading and html and html != last_html and html > 100:
            time.sleep(2.0)
            return True
        last_html = html
        time.sleep(0.5)
    return False


def find_chromium_window():
    out = subprocess.run(["xdotool", "search", "--name", "MLX Chat"],
                         env=ENV, capture_output=True, text=True)
    return [int(x) for x in out.stdout.split() if x.strip()]


def key(*keys):
    ids = find_chromium_window()
    if not ids:
        return False
    subprocess.run(["xdotool", "key", "--window", str(ids[0])] + list(keys),
                   env=ENV, capture_output=True)
    return True


def active_desc():
    return cdp_eval(
        "(() => { const a = document.activeElement;"
        " return a ? (a.id || a.tagName + ':' + (a.textContent || '').slice(0, 30))"
        " : 'none'; })()")


def tab_until(pred_js, max_tabs=25, label=""):
    """Tab until pred_js (evaluated on document.activeElement) is truthy."""
    for i in range(max_tabs):
        key("Tab")
        time.sleep(0.15)
        if cdp_eval(pred_js):
            print(f"    focused {active_desc()} after {i + 1} tabs ({label})")
            return True
    print(f"    tab_until FAILED ({label}); active={active_desc()}")
    return False


def ui_snapshot():
    return cdp_eval(
        "(() => { const d = (el) => el ? {id: el.id, tag: el.tagName,"
        " text: (el.textContent || '').slice(0, 60), label:"
        " el.getAttribute('aria-label'), title: el.title, disabled:"
        " !!el.disabled, hidden: !!el.hidden, pressed:"
        " el.getAttribute('aria-pressed')} : null;"
        " const vs = document.querySelector('.composer__voice-status');"
        " return JSON.stringify({active: d(document.activeElement),"
        " mic: d(document.getElementById('mic-btn')),"
        " stopSpeak: d(document.getElementById('stop-speak-btn')),"
        " continueChip: d(document.querySelector('.jump-chip--continue')),"
        " voiceStatus: vs ? vs.textContent : null,"
        " live: (document.getElementById('live-region') || {}).textContent,"
        " notice: (document.querySelector('.voice-notice') || {}).textContent,"
        " composerText: (document.getElementById('composer-text') || {}).value,"
        " composerFocused: document.activeElement ==="
        " document.getElementById('composer-text')}); })()")


def write_marker():
    with MARKER_FILE.open("w") as f:
        f.write(f"{MARKER}{time.time()}\n")


def read_utterances():
    if not LOG.exists() or not MARKER_FILE.exists():
        return []
    marker_ts = MARKER_FILE.stat().st_mtime
    out = []
    with LOG.open() as f:
        for line in f:
            m = re.match(r"^(\d+):(\d+):(\d+)\.(\d+)", line)
            if not m:
                continue
            h, mn, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            today = time.gmtime(marker_ts)
            log_ts = time.mktime((today.tm_year, today.tm_mon, today.tm_mday,
                                  h, mn, s, 0, 0, 0))
            if log_ts > marker_ts + 86400 / 2:
                log_ts -= 86400
            elif log_ts < marker_ts - 86400 / 2:
                log_ts += 86400
            if log_ts < marker_ts:
                continue
            sm = re.search(r"SPEECH OUTPUT:\s+'((?:[^'\\]|\\.)*)'", line)
            if sm:
                out.append((line[:15], sm.group(1)))
    return out


def run(name, keystrokes, label, state_name=None, duration=8):
    print(f"=== {name} ({label}) ===")
    sd = SCENARIOS_DIR / name
    sd.mkdir(parents=True, exist_ok=True)
    if state_name is not None:
        set_state(state_name)
        reload_page()
        time.sleep(2)
        wait_for_settled(15)
        reset_orca_focus()
    ax_before = ui_snapshot()
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
    (sd / "ax-before.json").write_text(str(ax_before))
    (sd / "ax-after.json").write_text(str(ui_snapshot()))
    print(f"  {len(utterances)} utterances captured")
    print(f"  after-state: {ui_snapshot()}")
    return utterances


CHROMIUM_BASE = [
    "chromium", "--no-sandbox", "--enable-accessibility",
    "--force-renderer-accessibility", "--disable-gpu",
    "--remote-debugging-port=9222", "--user-data-dir=/srv/logs/chromium-profile",
    "--window-size=1440,900", "--window-position=0,0",
    "--no-first-run", "--no-default-browser-check",
    "--enable-features=AccessibilityAriaVirtualContent",
    "--force-device-scale-factor=1",
    "--disable-extensions", "--disable-default-apps",
    "--disable-component-update", "--noerrdialogs", "--disable-infobars",
    "--kiosk", "--autoplay-policy=no-user-gesture-required",
]


def launch_chromium(capture_wav):
    subprocess.run(["pkill", "-f", "chromium.*remote-debugging-port"],
                   env=ENV, capture_output=True)
    time.sleep(2)
    cmd = list(CHROMIUM_BASE) + [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={capture_wav}",
        f"{STUB}/index.html",
    ]
    with open("/srv/logs/chromium.log", "a") as lf:
        subprocess.Popen(cmd, env={**ENV, "ACCESSIBILITY_ENABLED": "1",
                                   "GTK_MODULES": "gail:atk-bridge"},
                         stdout=lf, stderr=lf)
    time.sleep(4)
    wait_for_settled(15)
    reset_orca_focus()


def to_mic():
    return tab_until(
        "document.activeElement && document.activeElement.id === 'mic-btn'",
        label="mic-btn")


def to_read_aloud():
    return tab_until(
        "document.activeElement && document.activeElement.textContent === 'Read aloud'",
        label="read-aloud")


def to_stop_speaking():
    return tab_until(
        "document.activeElement && document.activeElement.id === 'stop-speak-btn'",
        label="stop-speak-btn")


def to_continue_chip():
    return tab_until(
        "document.activeElement && document.activeElement.className.includes('jump-chip--continue')",
        label="continue-chip")


# === Scenarios ===

def scenario_tab_walk(name, state, steps=16):
    def ks():
        time.sleep(0.3)
        for _ in range(steps):
            key("Tab")
            time.sleep(0.2)
    run(name, ks, f"Tab through controls ({state})", state_name=state)


def scenario_record_transcript():
    def ks():
        if not to_mic():
            return
        key("space")
        time.sleep(1.5)
        key("space")
        time.sleep(0.5)
    run("voice-record-transcript", ks,
        "keyboard start, stop, transcript insertion", state_name="voice-ready",
        duration=4)


def scenario_escape_cancel():
    def ks():
        if not to_mic():
            return
        key("space")
        time.sleep(1.0)
        key("Escape")
        time.sleep(0.5)
        # No keyboard trap: Tab must still move focus afterwards.
        key("Tab")
        time.sleep(0.3)
    run("voice-escape-cancel", ks,
        "Escape cancels from the focused mic button; Tab still moves",
        state_name="voice-ready", duration=2)


def scenario_transcribe_error():
    def ks():
        if not to_mic():
            return
        key("space")
        time.sleep(1.2)
        key("space")
        time.sleep(0.5)
    run("voice-transcribe-error", ks,
        "transcribe endpoint 500 -> error announcement",
        state_name="voice-transcribe-error", duration=4)


def scenario_no_speech():
    def ks():
        if not to_mic():
            return
        key("space")
        time.sleep(1.5)
        key("space")
        time.sleep(0.5)
    run("voice-no-speech", ks,
        "silent fake mic -> client silence gate announcement",
        state_name="voice-ready", duration=4)


def scenario_tts_read_stop():
    def ks():
        if not to_read_aloud():
            return
        key("Return")
        time.sleep(1.2)
        if not to_stop_speaking():
            return
        key("Return")
        time.sleep(0.5)
    run("tts-read-stop", ks,
        "read aloud, then Stop speaking via keyboard",
        state_name="voice-ready", duration=2)


def scenario_tts_playout():
    def ks():
        if not to_read_aloud():
            return
        key("Return")
        time.sleep(0.5)
    run("tts-playout", ks,
        "read aloud, playback runs to natural end",
        state_name="voice-ready", duration=11)


def scenario_tts_truncate():
    def ks():
        if not to_read_aloud():
            return
        key("Return")
        time.sleep(1.0)
    run("tts-truncate", ks,
        "12 s stream truncated at the 10 s cap; Continue reading",
        state_name="tts-truncate", duration=13)


def scenario_tts_truncate_resume():
    def ks():
        if not to_read_aloud():
            return
        key("Return")
        time.sleep(11.0)
        if not to_continue_chip():
            return
        key("Return")
        time.sleep(0.5)
    run("tts-truncate-resume", ks,
        "Continue reading resumes the truncated queue",
        state_name="tts-truncate", duration=3)


def scenario_voice_preview_picker():
    def ks():
        # Topbar: skip-link, new-chat, history, transfer, details.
        tab_until("document.activeElement && document.activeElement.id === 'details-btn'",
                  label="details-btn")
        key("Return")
        time.sleep(1.5)
        if not tab_until(
                "document.activeElement && document.activeElement.id === 'details-voice-select'",
                label="voice-select"):
            return
        key("Down")
        time.sleep(0.8)
        if not tab_until(
                "document.activeElement && document.activeElement.id === 'details-voice-preview'",
                label="voice-preview"):
            return
        key("Return")
        time.sleep(3.0)
        key("Escape")
        time.sleep(1.0)
        key("Tab")
        time.sleep(0.3)
    run("voice-preview-picker", ks,
        "Details drawer: voice picker change + Preview + Escape closes",
        state_name="voice-ready", duration=2)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    launch_chromium("/srv/logs/fake-mic-tone.wav")
    if which in ("all", "states"):
        scenario_tab_walk("voice-unqualified", "voice-unqualified")
        scenario_tab_walk("voice-missing", "voice-missing")
        scenario_tab_walk("voice-usable", "voice-usable")
        scenario_tab_walk("voice-ready-idle", "voice-ready")
    if which in ("all", "record"):
        scenario_record_transcript()
        scenario_escape_cancel()
        scenario_transcribe_error()
    if which in ("all", "nospeech"):
        launch_chromium("/srv/logs/fake-mic-silence.wav")
        scenario_no_speech()
        launch_chromium("/srv/logs/fake-mic-tone.wav")
    if which in ("all", "tts"):
        scenario_tts_read_stop()
        scenario_tts_playout()
        scenario_tts_truncate_resume()
    if which in ("all", "preview"):
        scenario_voice_preview_picker()
    print("done")


if __name__ == "__main__":
    import sys
    sys.exit(main())
