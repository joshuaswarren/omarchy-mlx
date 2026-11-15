"""parse2.py: per-submission GPU time from drm_sched_job_{queue,run,done} traces.
busy = done - max(run, previous done of the same process): the time the job held the GPU.
Phases split at the first streamed token (FIRSTTOK in the .drv file)."""
import re,glob,os,collections,lzma
R=os.path.dirname(os.path.abspath(__file__))
def pct(v,p): v=sorted(v); return v[min(len(v)-1,int(round(p/100*(len(v)-1))))] if v else float('nan')
def s(v): return f"{pct(v,50):6.2f} {pct(v,99):6.2f} {max(v) if v else float('nan'):6.2f}"
rx=re.compile(r"^\s*(\S+)-(\d+)\s+\[\d+\].*?\s(\d+\.\d+): drm_sched_job_(queue|run|done): (.*)$")
print("label | gen_tps | MLX phase n | busy(start..done excl. queue-ahead) p50 p99 max | queue->done p50 p99 max | vkcube frame p50 p99 max >16.7 >33 | Hyprland queue->done p50 p99 max")
for tf in sorted(glob.glob(R+"/*.trace.xz")):
    lab=os.path.basename(tf)[:-9]; w=open(tf[:-9]+".drv").read()
    m=re.search(r"WINDOW (\S+) (\S+)",w); ft=re.search(r"FIRSTTOK (\S+) gen_tps=(\S+)",w)
    if not m: continue
    t0,t1=map(float,m.groups()); tfirst=float(ft.group(1)) if ft else None; gtps=ft.group(2) if ft else "-"
    J={}
    for l in lzma.open(tf,'rt'):
        r=rx.match(l)
        if not r: continue
        ts=float(r.group(3)); ev=r.group(4); f=re.search(r"fence=(\d+:\d+)",r.group(5)).group(1)
        j=J.setdefault(f,{}); j[ev]=ts
        if ev=="queue": j["c"]=r.group(1); j["ring"]=re.search(r"ring=([^,]+)",r.group(5)).group(1)
    js=[j for j in J.values() if {"queue","run","done","c"}<=j.keys() and t0<=j["queue"]<=t1]
    py=sorted([j for j in js if j["c"].startswith("python")],key=lambda j:j["done"])
    prev=0
    for j in py: j["busy"]=(j["done"]-max(j["run"],prev))*1e3; prev=j["done"]
    vk=sorted(j["done"] for j in js if j["c"]=="vkcube"); iv=[(b-a)*1e3 for a,b in zip(vk,vk[1:])]
    hy=[(j["done"]-j["queue"])*1e3 for j in js if j["c"]=="Hyprland"]
    phases=[("all",py)]
    if tfirst: phases+= [("prefill",[j for j in py if j["queue"]<tfirst]),("decode",[j for j in py if j["queue"]>=tfirst])]
    for ph,sel in phases:
        print(f"{lab:38} {gtps:>7} {ph:7} n={len(sel):5} | {s([j['busy'] for j in sel])} | {s([(j['done']-j['queue'])*1e3 for j in sel])} | {s(iv)} {sum(x>16.7 for x in iv):3} {sum(x>33.3 for x in iv):3} | {s(hy)}")
