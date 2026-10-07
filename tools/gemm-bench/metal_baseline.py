"""Dense GEMM baseline through MLX, same cells as tools/gemm-bench.

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


if __name__ == "__main__":
    main()
