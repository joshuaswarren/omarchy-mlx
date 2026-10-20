import re,statistics,sys
for path in sys.argv[1:]:
    rows={}
    for l in open(path):
        m=re.search(r'sample \d+ (\w+) rc=0 decode ([\d.]+) prefill ([\d.]+) ttft ([\d.]+)',l)
        if m: rows.setdefault(m.group(1),[]).append(tuple(map(float,m.groups()[1:])))
    names=list(rows)
    med={k:tuple(statistics.median(x[i] for x in v) for i in range(3)) for k,v in rows.items()}
    print(path)
    for k in names: print('  %-10s n=%d decode %.1f prefill %.1f ttft %.1f' % (k,len(rows[k]),*med[k]))
    a,b=names[0],names[1]
    print('  %s vs %s: decode %+.1f%% prefill %+.1f%% ttft %+.1f%%' % (b,a,*[100*(med[b][i]-med[a][i])/med[a][i] for i in range(3)]))
