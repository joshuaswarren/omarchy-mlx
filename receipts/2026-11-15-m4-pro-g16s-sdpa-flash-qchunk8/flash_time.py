#!/usr/bin/env python3
# Isolated causal flash SDPA timing (#455 isolated-flash rows).
# Warmup then N timed reps per L; prints median ms, range and causal TFLOPS.
import os, sys, time, statistics
os.environ.setdefault("MLX_OMARCHY_NO_MLXLM_WARNING", "1")
import mlx.core as mx

def flops(Hq, D, L):
    # causal QK + PV: 2 * Hq * D * L * (L+1)
    return 2 * Hq * D * L * (L + 1)

def one(B, Hq, Hkv, L, D, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, L, D)).astype(mx.bfloat16)
    k = mx.random.normal((B, Hkv, L, D)).astype(mx.bfloat16)
    v = mx.random.normal((B, Hkv, L, D)).astype(mx.bfloat16)
    scale = D ** -0.5
    mx.eval(q, k, v)
    def run():
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        mx.eval(y)
        return y
    return run, flops(Hq, D, L)

def main():
    tag = os.environ.get("FLASH_KILL_TAG", "base")
    Ls = [int(x) for x in os.environ.get("FLASH_KILL_LS", "2048,8192,16384").split(",") if x]
    reps = int(os.environ.get("FLASH_KILL_REPS", "7"))
    Hq = int(os.environ.get("FLASH_KILL_HQ", "32"))
    Hkv = int(os.environ.get("FLASH_KILL_HKV", "8"))
    D = 128
    for L in Ls:
        run, fl = one(1, Hq, Hkv, L, D)
        run()  # warmup (compile + first launch)
        times = []
        for i in range(reps):
            t0 = time.monotonic_ns()
            run()
            times.append((time.monotonic_ns() - t0) / 1e6)
        med = statistics.median(times)
        tflops = (fl / (med * 1e-3)) / 1e12
        lo, hi = min(times), max(times)
        print(
            f"ATTR {tag} L={L} n={reps} med_ms={med:.3f} range={lo:.3f}-{hi:.3f} "
            f"TFLOPS={tflops:.3f} times=" + ",".join(f"{t:.3f}" for t in times),
            flush=True,
        )

if __name__ == "__main__":
    main()
