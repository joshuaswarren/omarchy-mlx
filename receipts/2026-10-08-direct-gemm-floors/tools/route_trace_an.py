"""Analyse route traces (H27 amendment 4). usage: route_trace_an.py <dir> <chip>; reads rt-<chip>-C2.err and rt-<chip>-C7.err."""
import re
import sys

d, chip = sys.argv[1], sys.argv[2]
EXPECT = {
    "g13c": set(),
    "g13g": {("bf16", "tn", m) for m in (512, 1024, 2048)} | {("f16", "tn", m) for m in (1024, 2048)}
            | {("f32", "nt", m) for m in (512, 1024, 2048)} | {("f32", "tn", m) for m in (512, 1024, 2048)}
            | {("f32", "nn", 2048)},
    "g14c": {("f16", "nt", m) for m in (1024, 2048, 4096)} | {("bf16", "nt", m) for m in (512, 1024, 2048, 4096)}
            | {("bf16", "tn", 4096)},
}[chip]


def seqs(path):
    out, cur = {}, None
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"@@CELL (\S+) (\S+) (\d+)", line)
            if m:
                cur = (m[1], m[2], int(m[3]))
                out[cur] = []
            elif line.startswith("@@END"):
                cur = None
            elif cur is not None:
                k = re.search(r"DISPATCH kernel=(\d+)", line)
                if k:
                    out[cur].append(int(k[1]))
    return out


a, b = seqs(f"{d}/rt-{chip}-C2.err"), seqs(f"{d}/rt-{chip}-C7.err")
print(f"cells: C2 {len(a)} C7 {len(b)}; empty traces C2 {sum(1 for v in a.values() if not v)} C7 {sum(1 for v in b.values() if not v)}")
diff = {c for c in a if c in b and a[c] != b[c]}
missing = EXPECT - diff
extra = diff - EXPECT
for c in sorted(diff):
    print("differs", c, a[c], "->", b[c])
print("expected-different cells that match:", sorted(missing) or "none")
print("unexpected differences:", sorted(extra) or "none")
print("VERDICT", "PASS" if not missing and not extra and len(a) == len(b) and len(a) > 0 else "FAIL")
