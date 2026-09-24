#!/usr/bin/env python3
"""Offline numeric check: coopmat GDN prefill vs the two-pass scan kernel.

Both kernels ship in the same wheel; the check runs the identical
deterministic inputs through each by flipping MLX_OMARCHY_NO_COOPMAT_GDN
(process-lifetime static, so each arm is a fresh interpreter) and compares

  1. determinism: each arm run twice must produce identical bytes;
  2. finiteness: no NaN/Inf in y or hf;
  3. y (bf16) distance scan-kernel vs coopmat, in bf16 quanta;
  4. hf (f32) max/rel deltas against the scan kernel;
  5. both arms against a numpy f32 per-token oracle (ops-path rounding:
     decay, sequential ascending dk sums, separate rounding per product).

Recorded inputs: none exist in the receipts (the GDN receipts generate
synthetic shapes), so inputs are fixed-seed synthetic at the exact Qwen3.8
model shapes (Dk=Dv=128, Hk=Hv=16, B=1), g in f32 (the serve path, from
compute_g) and bf16, chunk-boundary aligned and tail T values.

Usage: gdn-coopmat-check.py <python-with-wheel> <out.json>
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

T_CASES = [512, 520, 68]  # 68 = 8 full C=8 chunks + a 4-token tail (coopmat
                          # dispatch covers T >= 64; 7 now routes to scan)
SEED = 7

CHILD = r'''
import json, os, sys
import numpy as np
import mlx.core as mx

outdir, t_cases, seed = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])

def make(T):
    mx.random.seed(seed)
    # Model-scale magnitudes (Qwen3.8 GDN): k/q ~ N(0, 1/Dk) so beta.|k|^2
    # stays a contraction; beta, g in (0, 1) like sigmoid gates. Raw
    # normals blow the recurrence up past f32 within a few hundred tokens.
    q = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16) * 0.088
    k = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16) * 0.088
    v = mx.random.normal((1, T, 16, 128)).astype(mx.bfloat16)
    beta = (mx.random.normal((1, T, 16)) * 0.5).astype(mx.bfloat16)
    h0 = mx.random.normal((1, 16, 128, 128)).astype(mx.float32) * 0.1
    if os.environ.get("GDN_CHECK_G_BF16") == "1":
        g = (mx.random.normal((1, T, 16)) * 0.2).astype(mx.bfloat16) * 0.5 + 0.5
    else:
        g = mx.random.normal((1, T, 16)) * 0.2 * 0.5 + 0.5
    return q, k, v, g, beta, h0

for T in t_cases:
    q, k, v, g, beta, h0 = make(T)
    y, hf = mx.fast.gated_delta_update(q, k, v, g, beta, h0, None)
    mx.eval(y, hf)
    np.save(os.path.join(outdir, f"y{T}.npy"),
            np.asarray(y.astype(mx.float32)))
    np.save(os.path.join(outdir, f"hf{T}.npy"), np.asarray(hf))
    if os.environ.get("GDN_CHECK_SAVE_INPUTS") == "1":
        for name, x in (("q", q), ("k", k), ("v", v), ("beta", beta), ("h0", h0)):
            np.save(os.path.join(outdir, f"{name}{T}.npy"),
                    np.asarray(x.astype(mx.float32)))
        np.save(os.path.join(outdir, f"g{T}.npy"),
                np.asarray(g.astype(mx.float32)))
print("child ok")
'''

def run_arm(python, workdir, tag, g_bf16, save_inputs=False):
    env = dict(os.environ)
    env["GDN_CHECK_G_BF16"] = "1" if g_bf16 else "0"
    env["GDN_CHECK_SAVE_INPUTS"] = "1" if save_inputs else "0"
    if tag == "scan":
        env["MLX_OMARCHY_NO_COOPMAT_GDN"] = "1"
    else:
        env.pop("MLX_OMARCHY_NO_COOPMAT_GDN", None)
    outdir = os.path.join(workdir, f"{tag}_{int(g_bf16)}")
    os.makedirs(outdir, exist_ok=True)
    r = subprocess.run(
        [python, "-c", CHILD, outdir, json.dumps(T_CASES), str(SEED)],
        capture_output=True, text=True, env=env, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"[{tag}] child failed: {r.stderr[-2000:]}")
    digest = hashlib.sha256()
    for name in sorted(os.listdir(outdir)):
        if not (name.startswith("y") or name.startswith("hf")):
            continue  # outputs only: input file sets differ between arms
        digest.update(name.encode())
        with open(os.path.join(outdir, name), "rb") as f:
            digest.update(f.read())
    return outdir, digest.hexdigest()

def oracle(scan_dir):
    """Numpy f32 per-token reference with ops-path rounding."""
    y_out, hf_out = {}, {}
    for T in T_CASES:
        q = np.load(os.path.join(scan_dir, f"q{T}.npy"))[0]
        k = np.load(os.path.join(scan_dir, f"k{T}.npy"))[0]
        v = np.load(os.path.join(scan_dir, f"v{T}.npy"))[0]
        g = np.load(os.path.join(scan_dir, f"g{T}.npy"))[0]
        beta = np.load(os.path.join(scan_dir, f"beta{T}.npy"))[0]
        S = np.load(os.path.join(scan_dir, f"h0{T}.npy"))[0].copy()
        ys = np.empty((T, 16, 128), np.float32)
        for t in range(T):
            S = S * g[t][:, None, None]
            P = S * k[t][:, None, :]              # Hk==Hv 1:1 head map
            kv = np.zeros((16, 128), np.float32)
            for i in range(128):                  # sequential ascending sum
                kv += P[:, :, i]
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, None, :] * delta[:, :, None]
            o = np.zeros((16, 128), np.float32)
            for i in range(128):
                o += S[:, :, i] * q[t][:, i][:, None]
            ys[t] = o
        y_out[T] = ys
        hf_out[T] = S
    return y_out, hf_out

def bf16_quantum(x):
    a = np.abs(x)
    return np.where(a > 0, np.exp2(np.floor(np.log2(np.maximum(a, 1e-30))) - 7), 1e-30)

def dist(a, b):
    d = np.abs(a - b)
    rel = d / np.maximum(np.abs(a), 1e-30)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float(rel.max()),
        "p999_abs": float(np.quantile(d, 0.999)),
    }

def main():
    py = sys.argv[1]
    out_path = sys.argv[2]
    report = {"T_cases": T_CASES, "seed": SEED}
    workdir = tempfile.mkdtemp(prefix="gdn-coopmat-check-")
    try:
        for g_bf16 in (False, True):
            tag = "g_bf16" if g_bf16 else "g_f32"
            scan, scan_d = run_arm(py, workdir, "scan", g_bf16, save_inputs=True)
            scan2, scan2_d = run_arm(py, workdir, "scan", g_bf16)
            coop, coop_d = run_arm(py, workdir, "coop", g_bf16)
            coop2, coop2_d = run_arm(py, workdir, "coop", g_bf16)
            yr, hfr = oracle(scan)
            rep = {"determinism": scan_d == scan2_d and coop_d == coop2_d,
                   "scan_sha": scan_d, "coop_sha": coop_d,
                   "cases": {}}
            for T in T_CASES:
                y_s = np.load(os.path.join(scan, f"y{T}.npy"))
                y_c = np.load(os.path.join(coop, f"y{T}.npy"))
                h_s = np.load(os.path.join(scan, f"hf{T}.npy"))
                h_c = np.load(os.path.join(coop, f"hf{T}.npy"))
                y_o = yr[T].reshape(y_s.shape)
                h_o = hfr[T].reshape(h_s.shape)
                quanta = np.abs(y_s - y_c) / bf16_quantum(
                    np.maximum(np.abs(y_s), np.abs(y_c)))
                rep["cases"][str(T)] = {
                    "finite": bool(np.isfinite(y_c).all() and np.isfinite(h_c).all()),
                    "y_scan_vs_coop": dist(y_s, y_c),
                    "y_quanta_scan_vs_coop_max": float(quanta.max()),
                    "y_quanta_p999": float(np.quantile(quanta, 0.999)),
                    "y_exact_frac": float((y_s == y_c).mean()),
                    "hf_scan_vs_coop": dist(h_s, h_c),
                    "y_scan_vs_oracle": dist(y_s, y_o),
                    "y_coop_vs_oracle": dist(y_c, y_o),
                    "hf_coop_vs_oracle": dist(h_c, h_o),
                }
            report[tag] = rep
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    ok = all(report[t]["determinism"] and
             all(c["finite"] for c in report[t]["cases"].values())
             for t in ("g_f32", "g_bf16"))
    print("DETERMINISM+FINITE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
