import json
import os

for b in (40000, 50000, 60000, 80000, 120000):
    q = []
    path = os.path.expanduser("~/scratch/w9c-r2/subcap-r/subcap.jsonl.{}".format(b))
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("k") == "s" and "dur" in r:
                q.append(r["dur"] / 1e6)
    q.sort()
    n = len(q)

    def pct(p, q=q, n=n):
        return q[min(n - 1, int(p * n))]

    print("{} n={} p50={:.2f} p95={:.2f} p99={:.2f} max={:.2f}".format(b, n, pct(0.5), pct(0.95), pct(0.99), q[-1]))
