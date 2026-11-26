"""Route trace verdict by kernel NAME. usage: route_trace_names.py <dir> <chip> <enumA.h> <enumB.h>
The two builds have different ComputeKernel numbering (the PR base inserted MaskedScatterBool), so ids are mapped to names
with the enum of each build before the registered rule (exactly the EXPECT cells differ) is applied."""
import re
import sys

d, chip, ha, hb = sys.argv[1:5]
sys.path.insert(0, __file__.rsplit("/", 1)[0])


def names(path):
    out, on = [], False
    for line in open(path):
        if "enum class ComputeKernel" in line:
            on = True
            continue
        if on:
            if line.strip().startswith("};"):
                break
            s = re.sub(r"//.*", "", line).strip()
            m = re.fullmatch(r"([A-Za-z0-9_]+),?", s)
            if m:
                out.append(m[1])
    return out


na, nb = names(ha), names(hb)
EXPECT = {
    "g13c": set(),
    "g13g": {("bf16", "tn", m) for m in (512, 1024, 2048)} | {("f16", "tn", m) for m in (1024, 2048)}
            | {("f32", "nt", m) for m in (512, 1024, 2048)} | {("f32", "tn", m) for m in (512, 1024, 2048)}
            | {("f32", "nn", 2048)},
    "g14c": {("f16", "nt", m) for m in (1024, 2048, 4096)} | {("bf16", "nt", m) for m in (512, 1024, 2048, 4096)}
            | {("bf16", "tn", 4096)},
}[chip]


def seqs(path, nm):
    out, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.match(r"@@CELL (\S+) (\S+) (\d+)", line)
        if m:
            cur = (m[1], m[2], int(m[3]))
            out[cur] = []
        elif line.startswith("@@END"):
            cur = None
        elif cur is not None:
            k = re.search(r"DISPATCH kernel=(\d+)", line)
            if k:
                out[cur].append(nm[int(k[1])])
    return out


a, b = seqs(f"{d}/rt-{chip}-C2.err", na), seqs(f"{d}/rt-{chip}-C7.err", nb)
print(f"enum sizes A {len(na)} B {len(nb)}; cells: C2 {len(a)} PR {len(b)}; empty C2 {sum(1 for v in a.values() if not v)} PR {sum(1 for v in b.values() if not v)}")
diff = {c for c in a if c in b and a[c] != b[c]}
for c in sorted(diff):
    print("differs", c, a[c], "->", b[c])
missing, extra = EXPECT - diff, diff - EXPECT
print("expected-different cells that match:", sorted(missing) or "none")
print("unexpected differences:", sorted(extra) or "none")
print("VERDICT", "PASS" if not missing and not extra and len(a) == len(b) and len(a) > 0 else "FAIL")
