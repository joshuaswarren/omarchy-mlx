"""table.py <linux_dir> <macos_dir> [out.md]: Linux as % of macOS per row from score.sh outputs (score.log SCORE| lines + out/*.json + llama-*.json)."""
import json
import os
import sys


def load(d):
    rows = {}
    for key_file in sorted(os.listdir(os.path.join(d, "out"))) if os.path.isdir(os.path.join(d, "out")) else []:
        if key_file.endswith(".json"):
            with open(os.path.join(d, "out", key_file)) as f:
                rows[("mlx", key_file[:-5])] = json.load(f)
    for fn in sorted(os.listdir(d)):
        if fn.startswith("llama-") and fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn)) as f:
                    data = json.load(f)
            except ValueError:
                continue
            ts = {}
            for e in data:
                ts["pp512" if e.get("n_prompt") and not e.get("n_gen") else "tg128"] = e["avg_ts"]
            rows[("llama", fn[6:-5])] = ts
    notes = {}
    log = os.path.join(d, "score.log")
    if os.path.exists(log):
        with open(log) as f:
            for line in f:
                if line.startswith("SCORE|"):
                    p = line.rstrip("\n").split("|")
                    notes.pop(p[1], None)  # the last line for a key wins (earlier smoke or failed attempts are superseded)
                    if p[3] in ("FAILED", "SKIPPED", "NOT-RUN") or p[3].startswith("FAILED"):
                        notes[p[1]] = p[3] + " " + (p[4] if len(p) > 4 else "")
    return rows, notes


def cell(lin, mac):
    if lin is None and mac is None:
        return "not run", None
    if mac is None:
        return "macOS n/a", None
    if lin is None:
        return "Linux FAILED (0%)", 0.0
    return "%.1f%%" % (100.0 * lin / mac), lin / mac


lrows, lnotes = load(sys.argv[1])
mrows, mnotes = load(sys.argv[2])
keys = sorted({k[1] for k in list(lrows) + list(mrows)} | set(lnotes) | set(mnotes))
out = ["| Model | Metric | Linux | macOS | Linux as % of macOS |", "|---|---|---|---|---|"]
ratios = []
for key in keys:
    for kind in ("mlx", "llama"):
        lk, mk = lrows.get((kind, key)), mrows.get((kind, key))
        if lk is None and mk is None:
            continue
        if kind == "mlx":
            for name, path in (("decode tok/s", "decode_tok_s"), ("prefill 512 tok/s", "prefill_tok_s")):
                lv = lk[path]["median"] if lk else None
                mv = mk[path]["median"] if mk else None
                txt, r = cell(lv, mv)
                out.append("| %s | %s | %s | %s | %s |" % (key, name, "-" if lv is None else "%.1f" % lv, "-" if mv is None else "%.1f" % mv, txt))
                if r is not None:
                    ratios.append((r, key, name))
        else:
            for name in ("pp512", "tg128"):
                lv = lk.get(name) if lk else None
                mv = mk.get(name) if mk else None
                txt, r = cell(lv, mv)
                out.append("| %s (llama.cpp Vulkan vs Metal) | %s tok/s | %s | %s | %s |" % (key, name, "-" if lv is None else "%.1f" % lv, "-" if mv is None else "%.1f" % mv, txt))
                if r is not None:
                    ratios.append((r, key, name))
    for tag, notes in (("Linux", lnotes), ("macOS", mnotes)):
        if key in notes:
            out.append("| %s | %s note | %s | | |" % (key, tag, notes[key][:100]))
out.append("")
ratios.sort()
out.append("Lowest rows: " + "; ".join("%s %s %.0f%%" % (k, n, 100 * r) for r, k, n in ratios[:5]))
text = "\n".join(out)
print(text)
if len(sys.argv) > 3:
    with open(sys.argv[3], "w") as f:
        f.write(text + "\n")
