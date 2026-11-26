#!/usr/bin/env python3
# Isolated causal flash SDPA bit-identity dumps (#455).
# --dump writes the bf16 outputs (widened exactly to f32); --check compares
# against a dump and prints max_abs and relative L2.
import os, sys, time
os.environ.setdefault("MLX_OMARCHY_NO_MLXLM_WARNING", "1")
import numpy as np
import mlx.core as mx

def run(tag, B=1, Hq=32, Hkv=8, L=256, D=128, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, L, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, Hkv, L, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, Hkv, L, D)).astype(mx.bfloat16)
    scale = D ** -0.5
    mx.eval(q, k, v)
    t0 = time.monotonic_ns()
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(y)
    dt = (time.monotonic_ns() - t0) / 1e6
    arr = np.array(y.astype(mx.float32))
    print(f"SHAPE {tag} {arr.shape} dtype={y.dtype} ms={dt:.3f}", flush=True)
    return arr

def rel_l2(a, b):
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den != 0 else float(num)

def max_abs(a, b):
    return float(np.max(np.abs(a - b)))

def main():
    tag = os.environ.get("FLASH_GQA_TAG", "cur")
    Ls = [int(x) for x in os.environ.get("FLASH_GQA_LS", "128,256,512,1024").split(",") if x]
    dump = None
    check = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dump":
            dump = args[i+1]; i += 2
        elif args[i] == "--check":
            check = args[i+1]; i += 2
        elif args[i] == "--ls":
            Ls = [int(x) for x in args[i+1].split(",")]; i += 2
        else:
            i += 1
    out = {}
    for L in Ls:
        out[str(L)] = run(f"{tag}_L{L}", L=L)
        # second run for identity
        b = run(f"{tag}_L{L}b", L=L)
        d = max_abs(out[str(L)], b)
        print(f"IDENT {tag} L={L} max_abs={d:.6g} {'MATCH' if d == 0 else 'DRIFT'}", flush=True)
    if dump:
        np.savez_compressed(dump, **{f"L{k}": v for k, v in out.items()})
        print(f"DUMPED {dump}", flush=True)
    if check:
        ref = np.load(check)
        for L in Ls:
            a = out[str(L)]
            b = ref[f"L{L}"]
            print(
                f"CMP {tag} vs {check} L={L} max_abs={max_abs(a,b):.6g} rel_l2={rel_l2(a,b):.6g}",
                flush=True,
            )

if __name__ == "__main__":
    main()
