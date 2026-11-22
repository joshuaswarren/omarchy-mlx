#!/usr/bin/env python3
"""Host-stamped mlx-lm prefill + decode for a DRM scheduler ftrace window.
Prints CLOCK_MONOTONIC timestamps for the whole run (WINDOW t0 t1) and
first token (FIRSTTOK tf, plus DECODE tf_ns tl_ns n), matching both
receipts/2026-11-15-m4-pro-g16s-batch-work/real/reald.py and
receipts/2026-11-21-g13g-batch-work-20000/tools/decode_jobs.py.

Usage: decode_jobs.py MODEL_DIR GEN EXPECT_WHEEL REPO_ROOT
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

mdl, gen_s, expect_wheel, repo_root = sys.argv[1:5]
GEN = int(gen_s)

sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
if prov["verified"] != "match" or prov["version_match"] is not True:
    print("PROVENANCE_REFUSAL", json.dumps(prov), file=sys.stderr)
    sys.exit(3)
print(mp.provenance_line(prov), flush=True)

import mlx.core as mx
from mlx_lm import load, stream_generate

model, tok = load(mdl)
prompt = "The history of France begins in antiquity. " * 60
for _ in stream_generate(model, tok, prompt, max_tokens=8):
    pass
time.sleep(0.2)

ids = []
t0_s = time.clock_gettime(time.CLOCK_MONOTONIC)
tf_s = None
tf_ns = tl_ns = None
for r in stream_generate(model, tok, prompt, max_tokens=GEN):
    tns = time.monotonic_ns()
    ts = tns / 1e9
    if tf_ns is None:
        tf_ns = tns
        tf_s = ts
    tl_ns = tns
    ids.append(int(r.token))
t1_s = time.clock_gettime(time.CLOCK_MONOTONIC)
steps = len(ids) - 1
gen_tps = steps / ((tl_ns - tf_ns) / 1e9) if steps > 0 and tl_ns > tf_ns else 0.0
sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
print(f"PID {os.getpid()}", flush=True)
print(f"WINDOW {t0_s:.6f} {t1_s:.6f}", flush=True)
print(f"FIRSTTOK {tf_s:.6f} gen_tps={gen_tps:.2f}", flush=True)
print(f"DECODE {tf_ns} {tl_ns} {len(ids)} ids_sha256={sha}", flush=True)
