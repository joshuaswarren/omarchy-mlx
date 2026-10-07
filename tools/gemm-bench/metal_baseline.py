"""Dense GEMM and Q4 prefill baselines through MLX (gemm-bench cells).

Each cell times 3 rounds of 6 evaluated products and reports the median and
best rep in TFLOP/s, so the numbers line up with gemm-bench's "tflops" and
"tflops_best". Operands are uniform in [-1, 1) from a fixed seed. The script
runs unchanged on macOS Metal and on the Omarchy wheel, which gives the
same MLX-level cell on both systems.

macOS Metal (no other GPU load):
  python3 -m venv /tmp/mlx-metal
  /tmp/mlx-metal/bin/pip install -q mlx==0.32.3
  /tmp/mlx-metal/bin/python metal_baseline.py > metal-baseline.jsonl
"""
import hashlib
import json
import platform
import time

import mlx.core as mx

ROUNDS = 3
REPS = 6
# name, dtype, m, n, k, lhs column-major, rhs n-major
CELLS = [
    ("nn4096", mx.float16, 4096, 4096, 4096, False, False),
    ("nt4096", mx.float16, 4096, 4096, 4096, False, True),
    ("nt512", mx.float16, 512, 4096, 4096, False, True),
    ("tn4096", mx.float16, 4096, 4096, 4096, True, False),
    ("nt512k4104", mx.float16, 512, 4096, 4104, False, True),
    ("bf16_nn4096", mx.bfloat16, 4096, 4096, 4096, False, False),
    ("bf16_nt4096", mx.bfloat16, 4096, 4096, 4096, False, True),
    ("f32_nn4096", mx.float32, 4096, 4096, 4096, False, False),
    ("f32_nt4096", mx.float32, 4096, 4096, 4096, False, True),
]
# Qwen3.8-2B prefill linears: bf16 x, 4-bit weights (group 64), m = 512.
# name, m, n, k
QCELLS = [
    ("q4_gate512", 512, 6144, 2048),
    ("q4_down512", 512, 2048, 6144),
]


def chip():
    if mx.metal.is_available():
        return mx.metal.device_info().get("device_name", "")
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            return f.read().split(b"\0")[0].decode()
    except OSError:
        return platform.machine()


def operands(dtype, m, n, k, a_t, b_t):
    key_a, key_b = mx.random.split(mx.random.key(0x5EED))
    a = mx.random.uniform(-1, 1, (k, m) if a_t else (m, k), key=key_a).astype(dtype)
    b = mx.random.uniform(-1, 1, (n, k) if b_t else (k, n), key=key_b).astype(dtype)
    mx.eval(a, b)
    return (a.T if a_t else a), (b.T if b_t else b)


def main():
    print(json.dumps({
        "k": "host",
        "chip": chip(),
        "system": platform.platform(),
        "mlx": mx.__version__,
        "device": str(mx.default_device()),
    }))
    for name, dtype, m, n, k, a_t, b_t in CELLS:
        a, b = operands(dtype, m, n, k, a_t, b_t)
        for _ in range(3):
            mx.eval(a @ b)
        times = []
        for _ in range(ROUNDS):
            for _ in range(REPS):
                start = time.perf_counter()
                mx.eval(a @ b)
                times.append(time.perf_counter() - start)
        times.sort()
        flops = 2.0 * m * n * k
        print(json.dumps({
            "k": "time", "cell": name, "dtype": str(dtype), "m": m, "n": n,
            "kdim": k, "a_t": a_t, "b_t": b_t,
            "median_us": round(times[len(times) // 2] * 1e6, 1),
            "tflops": round(flops / times[len(times) // 2] / 1e12, 3),
            "tflops_best": round(flops / times[0] / 1e12, 3),
        }), flush=True)
    for name, m, n, k in QCELLS:
        x, w = operands(mx.bfloat16, m, n, k, False, True)
        wq, scales, biases = mx.quantize(w.T, group_size=64, bits=4)
        mx.eval(wq, scales, biases)

        def run(x=x, wq=wq, scales=scales, biases=biases):
            return mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                       group_size=64, bits=4)

        times = timed(run)
        out = run().astype(mx.float32)
        mx.eval(out)
        print(json.dumps({
            "k": "time", "cell": name, "m": m, "n": n, "kdim": k,
            "median_us": round(times[len(times) // 2] * 1e6, 1),
            "tflops": round(2.0 * m * n * k / times[len(times) // 2] / 1e12, 3),
            "out_sha": hashlib.sha256(bytes(memoryview(out))).hexdigest()[:16],
        }), flush=True)


def timed(fn):
    for _ in range(3):
        mx.eval(fn())
    times = []
    for _ in range(ROUNDS):
        for _ in range(REPS):
            start = time.perf_counter()
            mx.eval(fn())
            times.append(time.perf_counter() - start)
    return sorted(times)


if __name__ == "__main__":
    main()
