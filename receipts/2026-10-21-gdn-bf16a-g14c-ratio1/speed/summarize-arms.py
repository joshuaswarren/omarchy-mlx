import re,sys,statistics as S,collections
d=collections.defaultdict(dict)
for l in open(sys.argv[1]):
    m=re.match(r'ARMS round (\d+) (\w+) rc=(\d+) decode ([\d.]+).* sha (\w+)',l)
    if m: d[m[2]][int(m[1])]=(float(m[4]),m[5],m[3])
b=d['base']
for a,v in d.items():
    r=[v[k][0]/b[k][0]-1 for k in v if k in b]
    print(a,'median tok/s',S.median(x[0] for x in v.values()),'pct vs base per round',[round(100*x,2) for x in r],'median',round(100*S.median(r),2),'digests',set(x[1] for x in v.values()),'rc',set(x[2] for x in v.values()))
