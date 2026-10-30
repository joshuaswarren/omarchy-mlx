"""Per-step split of a batch decode profile: wall, GPU busy, the batch qmv kernel, everything else, and per-kernel-name tables.

Pairing is the one in tools/analyze_decode_profile.py: a submission belongs to the window of its `s` host time; window i is
[t1[i-1], t1[i]]. Steady steps skip the first 4. GPU time per dispatch = (t1 - t0) ticks * period_ns (the profiler adds a barrier per
dispatch, so absolute times inflate; differences between arms are the quantity of interest).

  b2prof_summary.py one --profile P --windows W --compute-h H [--out J]
  b2prof_summary.py cmp A.json [A2.json ...] --vs B.json [B2.json ...]     (arm A minus arm B, medians over the files of each arm)
"""
import argparse
import json
import re
import statistics as st

BATCH_KERNEL = "QmmVecQ4WordSubgroupBatch4BF16"


def kernel_names(path):
    """Index -> name for `enum class ComputeKernel`: comments stripped, explicit values honoured (a line-by-line regex missed entries that carry a trailing comment)."""
    with open(path) as f:
        src = f.read()
    i = src.index("enum class ComputeKernel")
    j = src.index("{", i)
    depth, k = 0, j
    while True:
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    body = re.sub(r"/\*.*?\*/", "", src[j + 1:k], flags=re.S)
    body = re.sub(r"//[^\n]*", "", body)
    names, val = {}, -1
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"(\w+)\s*(?:=\s*(\d+))?$", item)
        if not m:
            raise SystemExit(f"unparsed enum item: {item[:60]}")
        val = int(m.group(2)) if m.group(2) else val + 1
        names[val] = m.group(1)
    if names.get(547) != BATCH_KERNEL and BATCH_KERNEL not in names.values():
        raise SystemExit("batch kernel not in the enum")
    return [names.get(n, f"e{n}") for n in range(max(names) + 1)]


def one(a):
    names = kernel_names(a.compute_h)
    meta, wins = None, []
    with open(a.windows) as f:
        for line in f:
            r = json.loads(line)
            if "prompt" in r:
                meta = r
            else:
                wins.append(r)
    edges = [meta["start_monotonic"]] + [w["t1"] for w in wins]
    period_ns, subs, joins = 1.0, {}, 0
    with open(a.profile) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get("k")
            if k == "meta":
                period_ns = r["period_ns"]
            elif k == "s":
                subs.setdefault(r["s"], {"t": r["t"], "d": []})["t"] = r["t"]
            elif k == "d":
                subs.setdefault(r["s"], {"t": None, "d": []})["d"].append(r)

    def win_of(t):
        if t is None:
            return -1
        for i in range(len(edges) - 1):
            if edges[i] <= t < edges[i + 1]:
                return i
        return -2

    rows = [{"wall": (edges[i + 1] - edges[i]) / 1e6, "gpu": 0.0, "batch": 0.0, "disp": 0, "by": {}, "cnt": {}, "shape": {}}
            for i in range(len(wins))]
    for s in subs.values():
        w = win_of(s["t"])
        if w < 0:
            continue
        for d in s["d"]:
            t0, t1 = d.get("t0"), d.get("t1")
            g = (t1 - t0) * period_ns / 1e6 if t0 is not None and t1 is not None and t1 >= t0 else 0.0
            nm = names[d["e"]] if d["e"] < len(names) else f"e{d['e']}"
            r = rows[w]
            r["gpu"] += g
            r["disp"] += 1
            r["by"][nm] = r["by"].get(nm, 0.0) + g
            r["cnt"][nm] = r["cnt"].get(nm, 0) + 1
            if nm == BATCH_KERNEL:
                r["batch"] += g
                key = f"n={d['n']}"
                r["shape"].setdefault(key, [0.0, 0])
                r["shape"][key][0] += g
                r["shape"][key][1] += 1
    steady = rows[4:] if len(rows) > 8 else rows[1:]
    med = lambda f: st.median([f(r) for r in steady])
    kinds = sorted({k for r in steady for k in r["by"]})
    out = {
        "label": a.profile, "steady_steps": len(steady), "batch": meta.get("batch"),
        "wall_ms": med(lambda r: r["wall"]), "gpu_ms": med(lambda r: r["gpu"]),
        "batch_kernel_ms": med(lambda r: r["batch"]), "other_ms": med(lambda r: r["gpu"] - r["batch"]),
        "gap_ms": med(lambda r: r["wall"] - r["gpu"]), "disp": med(lambda r: r["disp"]),
        "batch_kernel_dispatches": med(lambda r: r["cnt"].get(BATCH_KERNEL, 0)),
        "by_name": {k: {"ms": st.median([r["by"].get(k, 0.0) for r in steady]),
                        "n": st.median([r["cnt"].get(k, 0) for r in steady])} for k in kinds},
        "batch_shapes": {k: {"ms": st.median([r["shape"].get(k, [0.0, 0])[0] for r in steady]),
                             "n": st.median([r["shape"].get(k, [0.0, 0])[1] for r in steady])}
                         for k in sorted({k for r in steady for k in r["shape"]})},
    }
    txt = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    print(f"steps {out['steady_steps']} wall {out['wall_ms']:.3f} gpu {out['gpu_ms']:.3f} batch_kernel {out['batch_kernel_ms']:.3f} "
          f"other {out['other_ms']:.3f} gap {out['gap_ms']:.3f} disp {out['disp']} batch_disp {out['batch_kernel_dispatches']}")
    top = sorted(out["by_name"].items(), key=lambda kv: -kv[1]["ms"])[:12]
    for k, v in top:
        print(f"  {k:46s} {v['ms']:8.3f} ms  {v['n']:7.1f} disp")


def load(paths):
    arms = [json.load(open(p)) for p in paths]
    keys = ("wall_ms", "gpu_ms", "batch_kernel_ms", "other_ms", "gap_ms", "disp")
    return {k: st.median([a[k] for a in arms]) for k in keys}, arms


def cmp_(a):
    x, xa = load(a.files)
    y, ya = load(a.vs)
    dw = x["wall_ms"] - y["wall_ms"]
    print("arm A medians:", {k: round(v, 3) for k, v in x.items()})
    print("arm B medians:", {k: round(v, 3) for k, v in y.items()})
    d = {k: x[k] - y[k] for k in x}
    print("A - B:", {k: round(v, 3) for k, v in d.items()})
    if abs(dw) > 1e-9:
        print(f"dWall {dw:.3f} ms; dBatch/dWall {d['batch_kernel_ms'] / dw:.2f}; dOther/dWall {d['other_ms'] / dw:.2f}; dGap/dWall {d['gap_ms'] / dw:.2f}")
    names = sorted({k for arm in xa + ya for k in arm["by_name"]})
    mean = lambda arms, k: st.mean([arm["by_name"].get(k, {"ms": 0.0})["ms"] for arm in arms])
    rows = sorted(((mean(xa, k) - mean(ya, k), k) for k in names), key=lambda r: -abs(r[0]))[:10]
    for v, k in rows:
        print(f"  {k:46s} A-B {v:+8.3f} ms per step")


def wall(a):
    wins = []
    with open(a.windows) as f:
        for line in f:
            r = json.loads(line)
            if "prompt" in r:
                meta = r
            else:
                wins.append(r)
    edges = [meta["start_monotonic"]] + [w["t1"] for w in wins]
    steps = [(edges[i + 1] - edges[i]) / 1e6 for i in range(len(wins))]
    steady = steps[4:] if len(steps) > 8 else steps[1:]
    out = {"wall_ms": st.median(steady), "steady_steps": len(steady)}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"unprofiled wall median {out['wall_ms']:.3f} ms over {len(steady)} steps")


def wallcmp(a):
    h = [json.load(open(p))["wall_ms"] for p in a.files]
    s = [json.load(open(p))["wall_ms"] for p in a.vs]
    print(f"hoisted {[round(x, 3) for x in h]} median {st.median(h):.3f}; shipped {[round(x, 3) for x in s]} median {st.median(s):.3f}; "
          f"shipped/hoisted speed ratio {st.median(s) / st.median(h):.3f}")


def tokhash(a):
    import hashlib
    h = hashlib.sha256()
    n = 0
    with open(a.windows) as f:
        for line in f:
            r = json.loads(line)
            if "tok" in r:
                h.update(json.dumps(r["tok"]).encode())
                n += 1
    print(f"tokens sha256 {h.hexdigest()[:16]} over {n} steps")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
o = sub.add_parser("one")
o.add_argument("--profile", required=True)
o.add_argument("--windows", required=True)
o.add_argument("--compute-h", required=True)
o.add_argument("--out", default="")
w = sub.add_parser("wall")
w.add_argument("--windows", required=True)
w.add_argument("--out", required=True)
th = sub.add_parser("tokhash")
th.add_argument("--windows", required=True)
wc = sub.add_parser("wallcmp")
wc.add_argument("files", nargs="+")
wc.add_argument("--vs", nargs="+", required=True)
c = sub.add_parser("cmp")
c.add_argument("files", nargs="+")
c.add_argument("--vs", nargs="+", required=True)
args = p.parse_args()
{"one": one, "cmp": cmp_, "wall": wall, "wallcmp": wallcmp, "tokhash": tokhash}[args.cmd](args)
