#!/usr/bin/env python3
"""One read pass over every part (F_NOCACHE: no page-cache pressure): whole-file sha256 + size, and per-tensor sha256 (name dtype shape sha256).
Writes SHA256SUMS (file sha256, name, size) and TENSOR_SHA256.txt into OUT_DIR (first argument; the part directories are all arguments), plus the sha256 of the small meta files. Read-only on the parts."""
import fcntl, hashlib, json, os, re, struct, sys, time
DIRS = sys.argv[1:]; assert DIRS, "usage: mac-manifest.py OUT_DIR [PART_DIR ...]"; OUT = DIRS[0]
files = sorted((f, d) for d in DIRS for f in os.listdir(d) if re.fullmatch(r"model-\d{5}-of-00063\.safetensors", f))
assert len(files) == 63, len(files)
sums = open(f"{OUT}/SHA256SUMS.inprogress", "w"); tens = open(f"{OUT}/TENSOR_SHA256.txt.inprogress", "w"); t0 = time.time(); tot = 0
for f, d in files:
    fd = os.open(f"{d}/{f}", os.O_RDONLY); fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
    fh = os.fdopen(fd, "rb", buffering=0); whole = hashlib.sha256()
    n = struct.unpack("<Q", fh.read(8))[0]; raw = fh.read(n); whole.update(struct.pack("<Q", n)); whole.update(raw)
    hdr = json.loads(raw); hdr.pop("__metadata__", None); pos = 8 + n
    for name, t in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0]):  # file order: one sequential pass
        a, b = t["data_offsets"]; assert 8 + n + a == pos, (f, name)
        h = hashlib.sha256(); left = b - a
        while left:
            c = fh.read(min(left, 1 << 24)); assert c; h.update(c); whole.update(c); left -= len(c)
        pos += b - a; tens.write(f"{f} {name} {t['dtype']} {'x'.join(map(str, t['shape']))} {h.hexdigest()}\n")
    assert pos == os.path.getsize(f"{d}/{f}"), f
    sums.write(f"{whole.hexdigest()}  {f}  {pos}\n"); tot += pos; sums.flush(); tens.flush()
    print(f, pos, f"{tot / 1e9:.1f} GB {time.time() - t0:.0f}s", flush=True)
for fn in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
    b = open(f"{OUT}/{fn}", "rb").read(); sums.write(f"{hashlib.sha256(b).hexdigest()}  {fn}  {len(b)}\n")
sums.close(); tens.close(); os.rename(f"{OUT}/SHA256SUMS.inprogress", f"{OUT}/SHA256SUMS"); os.rename(f"{OUT}/TENSOR_SHA256.txt.inprogress", f"{OUT}/TENSOR_SHA256.txt")
print("DONE", tot, "bytes", f"{time.time() - t0:.0f}s")
