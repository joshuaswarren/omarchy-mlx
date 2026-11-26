import sys,json,statistics as st,collections,re
d=collections.defaultdict(list); load=collections.defaultdict(list); cur=None
for l in open(sys.argv[1]):
    m=re.search(r"load=([0-9.]+)",l)
    if l.startswith("#") and m: cur=float(m.group(1))
    if l.startswith("JSON"):
        j=json.loads(l[5:]); k=(j["repo"].split("/")[1],j["label"]); d[k].append(j["decode_tps"]); load[k].append(cur)
base=sys.argv[2] if len(sys.argv)>2 else "g16"
for k,v in sorted(d.items()):
    b=d.get((k[0],base)); bm=st.median(b) if b else None
    print(f"{k[0][:20]:20} {k[1]:10} med={st.median(v):7.2f} {'%+.1f%%'%((st.median(v)/bm-1)*100) if bm else '':>7} runs={v} maxload={max(x or 0 for x in load[k]):.1f}")
