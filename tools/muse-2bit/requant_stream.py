#!/usr/bin/env python3
"""Streaming FP8 -> MLX mixed-bit converter for Muse Spark Open (no temp shards: every tensor is fetched by HTTP Range straight from the AMD FP8 repo).
Recipe: routed experts 2-bit g128; attention q/k/v/o/gate, pre/post_expert_proj, embed 4-bit g64; lm_head 6-bit g64; router f32; norms/residual gates unchanged. Text only, published-3bit tensor names.
One output part per layer (+ 'shared' for embed/head/final norm), written to the first --dest with room. CPU stream only. Resumable (.ok markers).
usage: requant_stream.py --layers 0-61|shared|1,5,9 --dest DIR:MIN_FREE_GB [--dest ...] [--conns 8] [--tag w0]"""
import argparse, json, os, re, shutil, struct, sys, threading, time, urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlx.core as mx
mx.set_default_device(mx.cpu); mx.set_cache_limit(1 << 30)  # CPU arrays use MLX's unified-memory allocator: bound its free-buffer cache (swap growth seen with 4 workers)
ap = argparse.ArgumentParser(); ap.add_argument("--layers", required=True); ap.add_argument("--dest", action="append", required=True)
ap.add_argument("--conns", type=int, default=8); ap.add_argument("--tag", default="w"); ap.add_argument("--state", default=".")
A = ap.parse_args()
REPO = "amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8"; SHA = "52cace63f43ed9fc6acef6dcd391adb02f4bfbe1"
RECIPE = {"expert": (2, 128), "attn": (4, 64), "latent": (4, 64), "embed": (4, 64), "head": (6, 64)}
UA = {"User-Agent": "w7y-stream"}
dests = []
for d in A.dest:
    p, _, mf = d.partition(":"); os.makedirs(p, exist_ok=True); dests.append((p, float(mf or 10) * 1e9))
LOGF = open(os.path.join(A.state, f"stream-{A.tag}.log"), "a", buffering=1); LOCK = threading.Lock()
def log(*a):
    s = time.strftime("%FT%TZ", time.gmtime()) + " [" + A.tag + "] " + " ".join(str(x) for x in a)
    with LOCK: print(s, flush=True); LOGF.write(s + "\n")
def jget(u): return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=120).read())
IDX = jget(f"https://huggingface.co/{REPO}/resolve/{SHA}/model.safetensors.index.json")["weight_map"]
URL = lambda f: f"https://huggingface.co/{REPO}/resolve/{SHA}/{f}"
NET = {"bytes": 0}; HDR = {}; HLOCK = threading.Lock()
def rng(f, a, b):
    for attempt in range(12):
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL(f), headers={**UA, "Range": f"bytes={a}-{b}"}), timeout=180); data = r.read()
            if len(data) != b - a + 1: raise IOError("short read")
            with LOCK: NET["bytes"] += len(data)
            return data
        except Exception as e: time.sleep(min(90, 3 * 2 ** attempt)); last = e
    raise last
def header(f):
    with HLOCK:
        if f in HDR: return HDR[f]
    n = struct.unpack("<Q", rng(f, 0, 7))[0]; h = json.loads(rng(f, 8, 8 + n - 1)); h.pop("__metadata__", None)
    with HLOCK: HDR[f] = (8 + n, h)
    return HDR[f]
def fetch(name):
    f = IDX[name]; base, h = header(f); t = h[name]; a, b = t["data_offsets"]; buf = rng(f, base + a, base + b - 1)
    dt = t["dtype"]; shape = tuple(t["shape"])
    if dt == "F8_E4M3": return np.frombuffer(buf, np.uint8).reshape(shape), dt
    if dt == "F32": return np.frombuffer(buf, np.float32).reshape(shape), dt
    if dt == "BF16": return mx.array(np.frombuffer(buf, np.uint16).reshape(shape)).view(mx.bfloat16), dt
    raise ValueError(dt)
def q(w, kind):
    bits, gs = RECIPE[kind]; return mx.quantize(w.astype(mx.bfloat16), group_size=gs, bits=bits)
def dq(raw, scale): return mx.from_fp8(mx.array(raw)).astype(mx.float32) * mx.array(scale.reshape(-1, 1).astype(np.float32))
def rel_rms(a, b):
    a = a.astype(mx.float32); b = b.astype(mx.float32); return float(mx.sqrt(mx.mean((a - b) ** 2)) / mx.sqrt(mx.mean(b ** 2)))
def put(out, base, r): out[base + ".weight"], out[base + ".scales"], out[base + ".biases"] = r
def pick_dest(need):
    while True:
        for p, mf in dests:
            if shutil.disk_usage(p).free - need >= mf: return p
        log("waiting for space", need); time.sleep(60)
def write_part(name, out):
    d = pick_dest(int(sum(int(np.prod(v.shape)) * v.dtype.size for v in out.values()) * 1.02))
    path = os.path.join(d, name); tmp = os.path.join(d, "inprogress-" + name)
    mx.save_safetensors(tmp, out, metadata={"format": "mlx"}); os.replace(tmp, path)
    n = struct.unpack("<Q", open(path, "rb").read(8))[0]; h = json.loads(open(path, "rb").read(8 + n)[8:]); h.pop("__metadata__", None)
    assert set(h) == set(out), "readback mismatch"
    open(path + ".ok", "w").write(str(os.path.getsize(path))); return path
def done(name): return any(os.path.exists(os.path.join(p, name + ".ok")) for p, _ in dests)
pool = ThreadPoolExecutor(max_workers=A.conns)
def do_layer(L, selfcheck):
    name = f"layer-{L:02d}.safetensors"
    if done(name): log("skip", name); return
    t0 = time.time(); pre = f"model.language_model.layers.{L}."; out = {}; stats = {"chk": None}
    names = [n for n in IDX if n.startswith(pre) and not n.endswith("_scale")]
    for n in names:
        rest = n[len(pre):]
        if re.match(r"^mlp\.experts\.\d+\.", rest) and not rest.startswith("mlp.experts.post_expert_norm"): continue
        if re.match(r"^self_attn\.(q|k|v|o|gate)_proj\.weight$", rest):
            raw, _ = fetch(n); sc, _ = fetch(n + "_scale"); put(out, f"model.layers.{L}.{rest[:-7]}", q(dq(raw, sc), "attn")); continue
        if re.match(r"^mlp\.(pre|post)_expert_proj\.weight$", rest):
            w, _ = fetch(n); put(out, f"model.layers.{L}.{rest[:-7]}", q(w, "latent")); continue
        if rest in ("mlp.gate.weight", "mlp.gate.e_score_correction_bias"):
            a, dt = fetch(n); out[f"model.layers.{L}.{rest}"] = (mx.array(a) if dt == "F32" else a.astype(mx.float32)); continue
        a, _ = fetch(n); out[f"model.layers.{L}." + rest.replace("mlp.experts.post_expert_norm.weight", "mlp.post_expert_norm.weight")] = (mx.array(a) if not hasattr(a, "dtype") or str(a.dtype).startswith("float") and not isinstance(a, mx.array) else a)
    mx.eval(*out.values())
    P3 = ("gate_proj", "up_proj", "down_proj"); En = lambda e, p: f"{pre}mlp.experts.{e}.{p}.weight"
    f0 = IDX[En(0, "gate_proj")]; base0, H0 = header(f0)
    snames = [En(e, p) + "_scale" for e in range(256) for p in P3]
    # scales: one ranged request for the whole layer's scale region (contiguous in the shard)
    slo = min(H0[n]["data_offsets"][0] for n in snames); shi = max(H0[n]["data_offsets"][1] for n in snames)
    assert all(IDX[n] == f0 for n in snames) and shi - slo <= 768 * 16384 * 2, "unexpected scale layout"
    SCB = rng(f0, base0 + slo, base0 + shi - 1)
    def scale_of(n): a, b = H0[n]["data_offsets"]; return np.frombuffer(SCB[a - slo:b - slo], np.float32)
    def getw(e):  # one ranged request for the 3 weights of one expert (contiguous in the shard)
        ns = [En(e, p) for p in P3]; offs = [H0[n]["data_offsets"] for n in ns]
        lo = min(o[0] for o in offs); hi = max(o[1] for o in offs)
        assert all(IDX[n] == f0 for n in ns) and hi - lo == sum(o[1] - o[0] for o in offs), "unexpected weight layout"
        buf = rng(f0, base0 + lo, base0 + hi - 1)
        return [np.frombuffer(buf[o[0] - lo:o[1] - lo], np.uint8).reshape(H0[n]["shape"]) for n, o in zip(ns, offs)]
    R = {p: ([None] * 256, [None] * 256, [None] * 256) for p in P3}
    win = deque(); nxt = 0
    while nxt < 12: win.append((nxt, pool.submit(getw, nxt))); nxt += 1
    while win:
        e, fw = win.popleft()
        if nxt < 256: win.append((nxt, pool.submit(getw, nxt))); nxt += 1
        raws = fw.result()
        if e % 16 == 0: mx.clear_cache()
        for p, raw in zip(P3, raws):
            w = dq(raw, scale_of(En(e, p) + "_scale")); r = q(w, "expert"); mx.eval(*r)
            for lst, t in zip(R[p], r): lst[e] = t
            if selfcheck and e == 0 and p == "gate_proj" and stats["chk"] is None: stats["chk"] = rel_rms(mx.dequantize(*r, group_size=128, bits=2), w)
    for p in P3:
        base = f"model.layers.{L}.mlp.switch_mlp.{p}"; ws, ss, bs = R.pop(p)
        out[base + ".weight"], out[base + ".scales"], out[base + ".biases"] = mx.stack(ws), mx.stack(ss), mx.stack(bs); mx.eval(out[base + ".weight"], out[base + ".scales"], out[base + ".biases"])
        del ws, ss, bs; mx.clear_cache()  # free the per-expert pieces before stacking the next projection (halves the peak)
    del SCB
    t1 = time.time(); path = write_part(name, out)
    import resource; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS reports bytes
    log(f"layer {L} done: {len(out)} tensors, {os.path.getsize(path) / 1e9:.2f} GB, fetch+convert {t1 - t0:.0f} s, write {time.time() - t1:.0f} s, net total {NET['bytes'] / 1e9:.1f} GB, peak RSS {rss:.1f} GB, selfcheck rel_rms expert0 gate {stats['chk']}")
def do_shared():
    name = "shared.safetensors"
    if done(name): log("skip", name); return
    out = {}
    for n, kind, base in (("model.language_model.embed_tokens.weight", "embed", "model.embed_tokens"), ("lm_head.weight", "head", "lm_head")):
        w, _ = fetch(n); put(out, base, q(w, kind))
    a, _ = fetch("model.language_model.norm.weight"); out["model.norm.weight"] = a
    mx.eval(*out.values()); path = write_part(name, out); log("shared done", os.path.getsize(path), "B")
sel = A.layers
if sel == "shared": do_shared()
else:
    if "-" in sel: a, b = sel.split("-"); Ls = list(range(int(a), int(b) + 1))
    else: Ls = [int(x) for x in sel.split(",")]
    for i, L in enumerate(Ls): do_layer(L, selfcheck=(i == 0))
log("DONE", sel, f"net {NET['bytes'] / 1e9:.1f} GB")
