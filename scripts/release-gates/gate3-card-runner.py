#!/usr/bin/env python3
import http.client, json, os, socket, sys, threading, time, urllib.error, urllib.request
HOME, LOG = sys.argv[1], sys.argv[2]
RUNTIME = os.path.join(HOME, "assistant", "application.json")
def say(s):
    print(s, flush=True)
    with open(LOG, "a") as f: f.write(s + "\n")
def runtime(): return json.load(open(RUNTIME))
def call(method, path, body=None):
    started = time.monotonic()
    while True:
        try:
            rt = runtime(); base = f"http://127.0.0.1:{rt['port']}"
            req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(), method=method,
                headers={"Cookie":rt["cookie"], "X-Assistant-CSRF":rt["csrf"], "Origin":base, "Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read())
        except (urllib.error.URLError, FileNotFoundError, ConnectionError, json.JSONDecodeError):
            if time.monotonic() - started > 1200: raise
            time.sleep(.5)
def denied():
    for addr in (("1.1.1.1",443),("140.82.112.3",443)):
        try: socket.create_connection(addr, timeout=3).close(); return False
        except OSError: pass
    return True
def sse_reader(cid, turn_ref, t0, events, stop):
    while not stop.is_set():
        try:
            rt=runtime(); c=http.client.HTTPConnection("127.0.0.1",rt["port"],timeout=300)
            c.request("GET",f"/api/conversations/{cid}/events?after=0",headers={"Cookie":rt["cookie"]})
            r=c.getresponse()
            if r.status != 200: c.close(); time.sleep(.2); continue
            for line in r:
                if not line.startswith(b"data:"): continue
                e=json.loads(line[5:]); now=time.monotonic()
                if e.get("turn_id")==turn_ref[0]:
                    events.append({"type":e.get("type"),"data":e.get("data"),"at_s":round(now-t0,3)})
                    if e.get("type") in ("done","error"): c.close(); return
            c.close()
        except Exception as e:
            if not stop.is_set(): events.append({"sse_error":str(e)[:200],"at_s":round(time.monotonic()-t0,3)})
            time.sleep(.2)
def turn(cid,payload,stream=False):
    t0=time.monotonic(); tref=[None]; events=[]; stop=threading.Event(); thread=None
    if stream:
        thread=threading.Thread(target=sse_reader,args=(cid,tref,t0,events,stop),daemon=True); thread.start()
    tid=call("POST",f"/api/conversations/{cid}/turns",payload)["turn_id"]; tref[0]=tid
    while time.monotonic()-t0<600:
        call("POST",f"/api/conversations/{cid}/heartbeat",{"turn_id":tid})
        rec=call("GET",f"/api/conversations/{cid}")
        m=next((x for x in rec.get("messages",[]) if x.get("turn_id")==tid and x.get("role")=="assistant"),None)
        if m and m.get("status") in ("complete","error","stopped"): break
        time.sleep(1)
    else: raise RuntimeError("turn timed out")
    wall=time.monotonic()-t0
    if stream:
        time.sleep(.3); stop.set(); thread.join(timeout=3)
    return m,wall,events
def main():
    say(f"OUTBOUND_DENIED {denied()}")
    t=time.monotonic()
    while True:
        st=call("GET","/api/status"); state=(st.get("setup") or {}).get("state")
        if state=="complete": break
        if state=="error": raise RuntimeError(str(st.get("setup")))
        if time.monotonic()-t>1200: raise RuntimeError("setup timeout")
        time.sleep(1)
    say(f"SETUP_COMPLETE {time.monotonic()-t:.1f}s pair={(st.get('active_pair') or {}).get('id')}")
    cid=call("POST","/api/conversations",{"save":False})["id"]
    m,dt,_=turn(cid,{"text":"Reply with the single word ready.","mode":"chat","max_tokens":16})
    say(f"CHAT status={m.get('status')} {dt:.2f}s text={str(m.get('content'))[:60]!r}")
    m2,dt2,events=turn(cid,{"text":"Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.","mode":"chat","max_tokens":700},True)
    txt=str(m2.get("content") or ""); comps=m2.get("components") or []
    component_types=[c.get("type") for c in comps if isinstance(c,dict)]
    component_events=[e for e in events if e.get("type")=="component"]
    event_types=[(e.get("data") or {}).get("type") for e in component_events]
    first_text=next((e["at_s"] for e in events if e.get("type")=="text"),None)
    first_component=next((e["at_s"] for e in component_events),None)
    notices=[(e.get("data") or {}).get("state") for e in events if e.get("type")=="status"]
    valid_visible=bool(comps) and len(component_events)>0 and bool(component_types) and set(component_types).issubset(set(event_types))
    ok=m2.get("status")=="complete" and (bool(txt.strip()) or valid_visible)
    say(f"CARD status={m2.get('status')} wall_s={dt2:.2f} text_len={len(txt)} components={len(comps)} component_types={component_types} sse_component_types={event_types} first_visible_text_s={first_text} first_visible_component_s={first_component} notices={notices}")
    say("CARD_CHECK " + ("PASS" if ok else "FAIL"))
    say("GATE " + ("PASS" if m.get("status")=="complete" and bool(str(m.get("content") or "").strip()) and ok else "FAIL"))
    say("SSE_EVENT_COUNT " + str(len(events)))
    for e in events: say("SSE " + json.dumps(e,ensure_ascii=False))
    return 0 if m.get("status")=="complete" and bool(str(m.get("content") or "").strip()) and ok else 1
if __name__=="__main__": sys.exit(main())
