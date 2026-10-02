#!/usr/bin/env python3
"""Quality-pair TTFT phase probe.

Starts/resumes the Quality pair (qwen3.8-27b-4bit + laya-mlx), then runs
warm-up turns and N measured chat turns with an exact-token prompt,
capturing for each turn:

  - first-visible-text from the live SSE stream (opened BEFORE submit)
  - the coordinator's per-phase timestamps (ttft-phases.jsonl tail)
  - the chat worker's [ttft] engine lines (tokenized / first_chunk)
  - loadavg + PSI cpu before/after each measured turn (timing gate:
    load 1min < 0.5 and PSI cpu avg10 == 0.00, else wait)

Writes one JSON object per session to --out. Read-only on the repo; the
only side effects are the assistant's own home files and this JSON.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import http.client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality27b_perf_api import Assistant, call, submit_turn  # noqa: E402

FILLER = ("The quick brown fox jumps over the lazy dog. " * 30).strip()


def loadavg():
    with open("/proc/loadavg") as f:
        parts = f.read().split()
    return [float(p) for p in parts[:3]]


def psi_cpu():
    with open("/proc/pressure/cpu") as f:
        for line in f:
            if line.startswith("some"):
                m = re.search(r"avg10=([\d.]+)", line)
                return float(m.group(1))
    return 0.0


def quiet_gate(max_wait_s):
    """Wait until load1 < 0.5 and PSI cpu avg10 == 0; record the wait."""
    start = time.monotonic()
    waited = 0.0
    while True:
        la, psi = loadavg(), psi_cpu()
        if la[0] < 0.5 and psi == 0.0:
            return {"load1": la[0], "psi_cpu_avg10": psi,
                    "waited_s": round(waited, 1)}
        if time.monotonic() - start > max_wait_s:
            return {"load1": la[0], "psi_cpu_avg10": psi,
                    "waited_s": round(waited, 1), "gate": "NOT_QUIET"}
        time.sleep(2.0)
        waited = time.monotonic() - start


def kill_pair_tree(home):
    """Kill assistant/workers by MLX_OMARCHY_HOME env sweep (workers carry
    no home path in argv)."""
    target = os.path.realpath(home)
    out = subprocess.run(["bash", "-c", "ps -eo pid= -o cmd="],
                         capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid_s, _, cmd = line.strip().partition(" ")
        if not pid_s.isdigit():
            continue
        if not re.search(r"mlx_omarchy_assistant|_mlxlm_server|mlx_omarchy_laya", cmd):
            continue
        try:
            with open(f"/proc/{pid_s}/environ", "rb") as f:
                env = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        if f"MLX_OMARCHY_HOME={target}" in env:
            pids.append(int(pid_s))
    for pid in pids:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    deadline = time.monotonic() + 10
    while pids and time.monotonic() < deadline:
        pids = [p for p in pids if os.path.exists(f"/proc/{p}")]
        if pids:
            time.sleep(0.3)
    for pid in pids:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


class SseDrain:
    """Open the events SSE stream before submit; collect timestamped events."""

    def __init__(self, cid, runtime, after_seq=0):
        rt = json.load(open(runtime))
        self.conn = http.client.HTTPConnection("127.0.0.1", rt["port"], timeout=600)
        self.conn.request("GET", f"/api/conversations/{cid}/events?after={after_seq}",
                          headers={"Cookie": rt["cookie"]})
        self.response = self.conn.getresponse()
        self.buf = b""
        self.events = []
        self.error = None

    def read_until_done(self, timeout_s=300, stop=None):
        started = time.monotonic()
        while time.monotonic() - started < timeout_s:
            if stop is not None and stop.is_set():
                self.error = "stopped"
                return
            chunk = self.response.read1(65536) if hasattr(self.response, "read1") \
                else self.response.read(65536)
            if not chunk:
                self.error = "stream closed"
                return
            self.buf += chunk
            while b"\n\n" in self.buf:
                block, self.buf = self.buf.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    if not line.startswith(b"data:"):
                        continue
                    event = json.loads(line[5:])
                    self.events.append((event.get("type"), event.get("data") or {},
                                        time.monotonic()))
                    if event.get("type") in ("done", "error"):
                        return

    def close(self):
        try:
            self.conn.close()
        except OSError:
            pass


def read_phases(home, turn):
    """Return the phase record for `turn` from ttft-phases.jsonl, or None."""
    path = os.path.join(home, "assistant", "logs", "ttft-phases.jsonl")
    found = None
    try:
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("turn") == turn:
                    found = record
    except OSError:
        pass
    return found


def read_engine_lines(chat_log, since_byte):
    """Return [ttft] lines appended to the chat worker log since a byte offset."""
    try:
        with open(chat_log, "rb") as f:
            f.seek(since_byte)
            fresh = f.read().decode("utf-8", "replace")
            end = since_byte + len(fresh.encode())
        return [ln for ln in fresh.splitlines() if ln.startswith("[ttft] ")], end
    except OSError:
        return [], since_byte


def load_tokenizer(runtime):
    """The chat model's tokenizer for exact token counting (local files only)."""
    try:
        status = call("GET", "/api/status", runtime=runtime)
        chat_path = (status.get("setup") or {}).get("model_paths", {}).get("chat")
        if not chat_path:
            return None
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(chat_path, local_files_only=True,
                                             trust_remote_code=False)
    except Exception as error:
        print(f"tokenizer unavailable ({error}); char/4 fallback", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--target-tokens", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--quiet-wait-s", type=int, default=900)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    home = os.path.realpath(args.home)
    session = {"label": args.label, "repo_serve": args.repo_serve,
               "home": home, "runs": []}

    quiet_gate(args.quiet_wait_s)
    with open("/proc/sys/kernel/random/boot_id") as f:
        session["boot_id"] = f.read().strip()
    session["uname"] = os.uname().release
    session["load_before"] = loadavg()
    session["psi_before"] = psi_cpu()

    prov = subprocess.run(
        [os.path.expanduser("~/.local/share/mlx-omarchy/venv/bin/python"),
         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "mlx_provenance.py")],
        capture_output=True, text=True, timeout=120)
    session["provenance"] = (prov.stdout or prov.stderr).strip()[-2000:]

    pair = Assistant("quality", home, args.repo_serve)
    do_setup = pair.boot_id_mismatch()
    session["did_setup"] = do_setup
    pair.start(do_setup)

    setup_deadline = time.monotonic() + (1500 if do_setup else 180)
    state = {}
    while time.monotonic() < setup_deadline:
        state = call("GET", "/api/status", runtime=pair.runtime).get("setup") or {}
        if state.get("state") == "complete":
            break
        if state.get("state") == "error":
            pair.kill()
            raise RuntimeError(f"setup error: {state}")
        time.sleep(1)
    session["setup_state"] = state.get("state")

    tok = load_tokenizer(pair.runtime)

    def tok_count(text):
        if tok is not None:
            return len(tok(text)["input_ids"])
        return len(text) // 4

    def build_exact(target):
        text = FILLER
        while tok_count(text) < target:
            text = text + " " + FILLER
        return text

    ttft_log = os.path.join(home, "assistant", "logs", "ttft-phases.jsonl")
    phases_offset = 0
    if os.path.exists(ttft_log):
        phases_offset = os.path.getsize(ttft_log)
    chat_log = os.path.join(home, "assistant", "logs", "quality-chat.log")
    engine_offset = 0
    if os.path.exists(chat_log):
        engine_offset = os.path.getsize(chat_log)

    prompt = build_exact(args.target_tokens)
    session["user_prompt_tokens"] = tok_count(prompt)
    session["user_prompt_chars"] = len(prompt)

    def heartbeat_loop(cid, turn, stop):
        while not stop.wait(5):
            try:
                call("POST", f"/api/conversations/{cid}/heartbeat",
                     {"turn_id": turn}, runtime=pair.runtime)
            except Exception:
                pass

    def one_turn(label, text, max_tokens, measure):
        nonlocal engine_offset
        cid = call("POST", "/api/conversations", {"save": False},
                   runtime=pair.runtime)["id"]
        gate = quiet_gate(args.quiet_wait_s) if measure else {"skipped": True}
        # Open the SSE stream BEFORE submit so no event is missed.
        drain = SseDrain(cid, pair.runtime)
        t0 = time.monotonic()
        body = {"text": text, "mode": "chat", "max_tokens": max_tokens}
        turn = call("POST", f"/api/conversations/{cid}/turns", body,
                    runtime=pair.runtime)["turn_id"]
        stop = threading.Event()
        beat = threading.Thread(target=heartbeat_loop,
                                args=(cid, turn, stop), daemon=True)
        beat.start()
        try:
            drain.read_until_done()
        finally:
            stop.set()
            beat.join(timeout=5)
        wall = time.monotonic() - t0
        drain.close()
        fvt = None
        text_out = ""
        for etype, data, ts in drain.events:
            if etype == "text":
                if fvt is None and data.get("text"):
                    fvt = ts - t0
                text_out += data.get("text") or ""
        phases = None
        if measure:
            time.sleep(0.5)  # emit flush
            record = read_phases(home, turn)
            if record:
                phases = record.get("phases")
        engine_lines, engine_offset_new = read_engine_lines(chat_log, engine_offset) \
            if measure else ([], engine_offset)
        if measure:
            engine_offset = engine_offset_new
        run = {"label": label, "turn": turn, "wall_s": round(wall, 3),
               "fvt_s": round(fvt, 3) if fvt is not None else None,
               "text_len": len(text_out), "text_head": text_out[:400],
               "gate": gate, "phases": phases,
               "engine_lines": engine_lines if measure else None}
        if measure:
            run["load_after"] = loadavg()
            run["psi_after"] = psi_cpu()
            session["runs"].append(run)
            f = run["fvt_s"]
            print(f"  {label}: fvt={f:.3f}s wall={run['wall_s']}s "
                  f"text_len={len(text_out)} load1={gate.get('load1')}", flush=True)
        return run

    for i in range(args.warmup):
        one_turn(f"warmup_{i}", "Say hello in five words or fewer.", 16, False)

    for i in range(args.runs):
        one_turn(f"run_{i}", prompt, args.max_tokens, True)

    session["load_after"] = loadavg()
    session["psi_after"] = psi_cpu()
    try:
        pair.kill()
    finally:
        kill_pair_tree(home)
    with open(args.out, "w") as f:
        json.dump(session, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
