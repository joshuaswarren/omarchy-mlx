"""Failing-first benchmark: small-row quantized matmul on G13G, narrowed gate.

Scope (narrowed 2026-11-15 after wheel measurement): the Rows8 route applies
to m in {2,3,4} only. m >= 5 keeps the tile/coopmat path (the scalar-FMA row
chain loses to the coopmat tile there), so this file measures m = 2, 3, 4.

Targets: beat the tile path by >= 1.2x (tile baselines from the 11-15 spike
on jwm1, descending run: m=2 -> 0.299, m=4 -> 0.150 ms/row; m=3 interpolates
at ~0.22). If the kernel meets them, the file PASSES and the route is a
pinned win where it applies. m = 8/16 are covered by the doctest gate check
(route not taken), not by wall time here.
"""
import time
import mlx.core as mx

K, N = 2048, 5120
BITS, GROUP = 4, 64
TARGET_MS_PER_ROW = {2: 0.249, 3: 0.183, 4: 0.125}


def bench(m, iters=20, warm=5):
    x = mx.random.normal((m, K), dtype=mx.bfloat16)
    w = (mx.random.uniform(0, 16, (N, K * BITS // 32)).astype(mx.uint32))
    scales = mx.random.normal((N, K // GROUP), dtype=mx.bfloat16)
    biases = mx.random.normal((N, K // GROUP), dtype=mx.bfloat16)
    mx.eval(x, w, scales, biases)
    for _ in range(warm):
        mx.eval(mx.quantized_matmul(x, w, scales, biases, transpose=True,
                                    group_size=GROUP, bits=BITS))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(mx.quantized_matmul(x, w, scales, biases, transpose=True,
                                    group_size=GROUP, bits=BITS))
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2] / m


failures = 0
for m, target in TARGET_MS_PER_ROW.items():
    got = bench(m)
    ok = got < target
    failures += not ok
    print(f"m={m:3d} per-row {got:.4f} ms  target < {target:.3f}  {'PASS' if ok else 'FAIL'}")
print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} shapes over target)")
raise SystemExit(1 if failures else 0)
