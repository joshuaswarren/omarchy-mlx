"""Drive the running app over its authenticated loopback API.

  app_client.py setup   <home>          POST /api/setup (everyday, voice) and wait
  app_client.py latency <home> <wav> N  two warmups, then N POST /api/transcribe, upload included
  app_client.py launch  <home>          print a one-use browser URL
  app_client.py status_latency <home> N  N timed GET /api/status
"""
import http.client, json, sys, time
from pathlib import Path


def runtime(home):
    return json.loads((Path(home) / "assistant" / "application.json").read_text())


def call(rt, method, path, body=None, content_type="application/json"):
    conn = http.client.HTTPConnection("127.0.0.1", rt["port"], timeout=300)
    data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
    conn.request(method, path, data, {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
                                      "Origin": f"http://127.0.0.1:{rt['port']}",
                                      "Content-Type": content_type})
    resp = conn.getresponse()
    payload = json.loads(resp.read() or b"{}")
    conn.close()
    if resp.status >= 400:
        raise RuntimeError(f"{method} {path}: {resp.status} {payload}")
    return payload


cmd, home = sys.argv[1], sys.argv[2]
rt = runtime(home)
if cmd == "setup":
    call(rt, "POST", "/api/setup", {"pair_id": "everyday", "approve_download": True, "voice": True})
    started = time.monotonic()
    while True:
        status = call(rt, "GET", "/api/status")
        state = (status.get("setup") or {}).get("state")
        if state in ("complete", "error"):
            break
        time.sleep(1)
    voice = status.get("voice", {})
    print(json.dumps({"setup": status.get("setup"), "seconds": round(time.monotonic() - started, 1),
                      "recognition_state": voice.get("recognition", {}).get("state"),
                      "recognition_reasons": voice.get("recognition", {}).get("reasons"),
                      "synthesis_state": voice.get("synthesis", {}).get("state")}, indent=1))
elif cmd == "latency":
    wav, n = Path(sys.argv[3]).read_bytes(), int(sys.argv[4])
    status = call(rt, "GET", "/api/status")
    pair = {"active_pair": (status.get("active_pair") or {}).get("id"), "state": status.get("state"),
            "setup": status.get("setup"),
            "recognition_state": status.get("voice", {}).get("recognition", {}).get("state")}
    thermal = lambda: int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    thermal_before = thermal()
    for _ in range(2):
        call(rt, "POST", "/api/transcribe", wav, "audio/wav")
    times, texts = [], set()
    for _ in range(n):
        t0 = time.monotonic()
        texts.add(call(rt, "POST", "/api/transcribe", wav, "audio/wav")["text"])
        times.append((time.monotonic() - t0) * 1000)
    ordered = sorted(times)
    print(json.dumps({"pair": pair, "warmups": 2, "thermal_zone0_c": [thermal_before, thermal()],
                      "n": n, "wav_bytes": len(wav), "ms": [round(t, 1) for t in times],
                      "p50_ms": round(ordered[(n + 1) // 2 - 1], 1),
                      "p95_ms": round(ordered[-(-n * 95 // 100) - 1], 1),
                      "max_ms": round(ordered[-1], 1), "distinct_transcripts": sorted(texts)}, indent=1))
elif cmd == "status_latency":
    n, times = int(sys.argv[3]), []
    for _ in range(n):
        t0 = time.monotonic()
        call(rt, "GET", "/api/status")
        times.append((time.monotonic() - t0) * 1000)
    ordered = sorted(times)
    print(json.dumps({"n": n, "ms": [round(t, 1) for t in times], "p50_ms": round(ordered[(n + 1) // 2 - 1], 1),
                      "p95_ms": round(ordered[-(-n * 95 // 100) - 1], 1)}, indent=1))
elif cmd == "launch":
    token = call(rt, "POST", "/api/launch", {})["token"]
    print(f"http://127.0.0.1:{rt['port']}/#token={token}")
