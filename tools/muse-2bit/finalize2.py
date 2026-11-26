#!/usr/bin/env python3
"""Finalize for the streaming converter's output (layer-NN.safetensors + shared.safetensors, possibly spread over several dirs).
Checks every tensor against the published mlx-community 3-bit inventory (names, dtypes, logical shapes; widths depend on bits/group), checks the recipe bits per group,
writes model.safetensors.index.json + config.json (published 3-bit config + this recipe's per-module quantization) into META_DIR, and with --rename renames the parts
in place to model-000NN-of-00063.safetensors (shared = 00001, layer L = L+2) so mlx-lm's glob 'model*.safetensors' finds them. Reads only headers.
usage: finalize2.py INVENTORY_3BIT_JSON META_DIR OUTDIR [OUTDIR...] [--partial] [--rename]"""
import json, os, re, struct, sys, collections, urllib.request
args = [a for a in sys.argv[1:] if not a.startswith("--")]; PARTIAL = "--partial" in sys.argv; RENAME = "--rename" in sys.argv
INV3, META, DIRS = args[0], args[1], args[2:]
RECIPE = {"expert": (2, 128), "attn": (4, 64), "latent": (4, 64), "embed": (4, 64), "head": (6, 64)}
N = 63
def grp(b): return "expert" if "switch_mlp" in b else "latent" if "expert_proj" in b else "attn" if "self_attn" in b else "embed" if "embed_tokens" in b else "head"
def final_name(p):
    return "model-00001-of-%05d.safetensors" % N if p == "shared.safetensors" else "model-%05d-of-%05d.safetensors" % (int(re.match(r"layer-(\d+)", p).group(1)) + 2, N)
parts = {}  # original name -> dir
for d in DIRS:
    for p in os.listdir(d):
        if re.fullmatch(r"(layer-\d\d|shared)\.safetensors", p) and os.path.exists(f"{d}/{p}.ok"):
            assert os.path.getsize(f"{d}/{p}") == int(open(f"{d}/{p}.ok").read().strip()), f"size != .ok marker: {d}/{p}"
            assert p not in parts, f"{p} present in two dirs: {parts[p]} and {d}"; parts[p] = d
wm, shapes, dtypes, sizes = {}, {}, {}, {}
for p, d in parts.items():
    f = open(f"{d}/{p}", "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); h.pop("__metadata__", None); sizes[p] = os.path.getsize(f"{d}/{p}")
    for k, v in h.items():
        assert k not in wm, f"duplicate tensor {k}"; wm[k] = p; shapes[k] = v["shape"]; dtypes[k] = v["dtype"]
_inv = json.load(open(INV3)); inv = {t["name"]: t for t in _inv["tensors"]}  # the inventory records repo + revision; the reference config is fetched from that same revision
missing = sorted(set(inv) - set(wm)); extra = sorted(set(wm) - set(inv))
print("parts", len(parts), "of", N, "| tensors", len(wm), "expected", len(inv), "missing", len(missing), "extra", len(extra), extra[:3])
bad = []
for k in set(wm) & set(inv):
    a, b = shapes[k], inv[k]["shape"]
    if dtypes[k] != inv[k]["dtype"]: bad.append((k, "dtype", dtypes[k], inv[k]["dtype"]))
    elif k.endswith((".scales", ".biases")):
        g = RECIPE[grp(k)][1]; exp = [*b[:-1], b[-1] * 64 // g]
        if a != exp: bad.append((k, a, exp))
    elif dtypes[k] != "U32":
        if a != b: bad.append((k, a, b))
    elif a[:-1] != b[:-1]: bad.append((k, a, b))
cnt = collections.Counter()
for k in wm:
    if k.endswith(".weight") and dtypes[k] == "U32":
        g = grp(k); sc = shapes[k[:-7] + ".scales"]; cnt[(g, round(shapes[k][-1] * 32 / (sc[-1] * RECIPE[g][1]), 3))] += 1
for g, (b, gs) in RECIPE.items():
    got = {bb for (gg, bb) in cnt if gg == g}
    if got and got != {float(b)}: bad.append((g, "bits", got, b))
print("shape/dtype/bits mismatches:", len(bad), bad[:5], "| bits groups:", dict(cnt))
total = sum(sizes.values()); print("bytes in parts", total)
if (missing or extra or bad) and not PARTIAL: sys.exit("INCOMPLETE/MISMATCH (missing, extra or mismatched tensors)")
if PARTIAL: sys.exit(0)
os.makedirs(META, exist_ok=True)
json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted((k, final_name(p)) for k, p in wm.items()))}, open(f"{META}/model.safetensors.index.json", "w"), indent=1)
base = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/{_inv['repo']}/resolve/{_inv['sha']}/config.json", headers={"User-Agent": "w7y-finalize"}), timeout=120).read())
q = {"group_size": 64, "bits": 4, "mode": "affine"}
for k in wm:
    if k.endswith(".scales"):
        b = k[:-7]; bits, gs = RECIPE[grp(b)]
        if (bits, gs) != (4, 64): ov = {"group_size": gs, "bits": bits, "mode": "affine"}; q[b] = ov; q["language_model." + b] = ov
base["quantization"] = q; base["quantization_config"] = dict(q)
json.dump(base, open(f"{META}/config.json", "w"), indent=2)
if RENAME:
    for p, d in parts.items():
        dst = f"{d}/{final_name(p)}"; assert not os.path.exists(dst); os.rename(f"{d}/{p}", dst); os.rename(f"{d}/{p}.ok", dst + ".ok")
    print("renamed", len(parts), "parts")
import collections as _c
by = _c.defaultdict(list)
for p, d in parts.items(): by[d].append(int(re.match(r"layer-(\d+)", p).group(1)) if p != "shared.safetensors" else -1)
for d, l in by.items(): print("DIR", d, "shared" if -1 in l else "", "layers", sorted(x for x in l if x >= 0))
print("FINALIZED meta in", META)
