#!/bin/bash
# lane UpstreamMlx - 1008b (b): quantized_matmul K-split probe (mlx#4641 class)
set -uo pipefail
mkdir -p "$HOME/u1008b-probe"
export VK_DRIVER_FILES=/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json
export MLX_OMARCHY_TRACE_DISPATCH=1
PY=/usr/lib/omarchy-mlx/venv/bin/python
"$PY" - > "$HOME/u1008b-probe/4641-splitk.log" 2>&1 << 'PROBE'
import numpy as np
import mlx.core as mx

K, N = 9728, 2560
GROUP = 64
print("probe: quantized_matmul bf16 at K=%d N=%d (mlx#4641 split-K class)" % (K, N))
rng = np.random.default_rng(7)
w = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
wq, scales, biases = mx.quantize(mx.array(w), group_size=GROUP, bits=4)
wd = mx.dequantize(wq, scales, biases, group_size=GROUP, bits=4)
w64 = np.asarray(wd, dtype=np.float64)
print("weights quantized: wq%s scales%s" % (wq.shape, scales.shape))

for M in (1, 8, 256):
    for seed in range(3):
        r = np.random.default_rng(1000 + seed)
        x = (r.standard_normal((M, K)) * 0.5).astype(np.float32)
        out = mx.quantized_matmul(
            mx.array(x).astype(mx.bfloat16), wq, scales, biases,
            transpose=True, group_size=GROUP, bits=4)
        mx.eval(out)
        got = np.asarray(out.astype(mx.float32), dtype=np.float64)
        ref = x.astype(np.float64) @ w64.T
        maxabs = float(np.abs(got - ref).max())
        denom = float(np.abs(ref).max())
        print("RESULT M=%d seed=%d maxabs=%.6e rel=%.6e"
              % (M, seed, maxabs, maxabs / denom))

# Cross-path consistency: the M=256 result row j must match a fresh M=1
# call on the same row within the same tolerance class.
r = np.random.default_rng(2000)
x256 = (r.standard_normal((256, K)) * 0.5).astype(np.float32)
out256 = mx.quantized_matmul(
    mx.array(x256).astype(mx.bfloat16), wq, scales, biases,
    transpose=True, group_size=GROUP, bits=4)
mx.eval(out256)
worst = 0.0
for j in (0, 1, 127, 255):
    out1 = mx.quantized_matmul(
        mx.array(x256[j:j + 1]).astype(mx.bfloat16), wq, scales, biases,
        transpose=True, group_size=GROUP, bits=4)
    mx.eval(out1)
    d = float(np.abs(
        np.asarray(out1.astype(mx.float32), dtype=np.float64)
        - np.asarray(out256.astype(mx.float32), dtype=np.float64)[j]
    ).max())
    worst = max(worst, d)
    print("CROSSPATH row=%d maxdiff=%.6e" % (j, d))
print("CROSSPATH worst=%.6e" % worst)
print("probe done")
PROBE
echo "exit=$?" >> "$HOME/u1008b-probe/4641-splitk.log"
grep -E "RESULT|CROSSPATH|probe done|exit=" "$HOME/u1008b-probe/4641-splitk.log"
echo "--- distinct dispatch kernel ids ---"
grep -o "DISPATCH kernel=[0-9]*" "$HOME/u1008b-probe/4641-splitk.log" | sort | uniq -c | sort -rn | head -8
