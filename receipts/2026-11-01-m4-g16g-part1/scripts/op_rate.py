"""Per-op issue rate on the M4 GPU: 16 independent chains of one op per loop
iteration, reported as giga lane-ops per second (FMA counted as one op)."""
import time

import prov

prov.check()
import mlx.core as mx

ITER = 2048
CHAINS = 16
OPS = {
    "fma":  ("float", "a{c} = fma(a{c}, 0.9999f, 0.0001f);"),
    "fadd": ("float", "a{c} = a{c} + 1.0001f;"),
    "fmul": ("float", "a{c} = a{c} * 0.9999f;"),
    "iadd": ("uint", "a{c} = a{c} + 0x9e3779b9u;"),
    "iand": ("uint", "a{c} = (a{c} & 0xfffffff7u) ^ k;"),
    "ior":  ("uint", "a{c} = a{c} | (k + {c}u);"),
    "ushr": ("uint", "a{c} = (a{c} >> 1u) + k;"),
    "ishl": ("uint", "a{c} = (a{c} << 1u) + k;"),
    "imul": ("uint", "a{c} = a{c} * 0x9e3779b9u;"),
    "u2f":  ("float", "a{c} = float(as_type<uint>(a{c}) & 0xffu);"),
    "nib":  ("float", "a{c} = as_type<float>((as_type<uint>(a{c}) >> 4u & 0xfu) | 0x4B000000u) - 8388608.0f;"),
}
THREADS = 1 << 20


def rate(name, ty, stmt):
    body = "uint i = thread_position_in_grid.x;\n uint k = as_type<uint>(inp[i & 1023u]);\n"
    for c in range(CHAINS):
        body += f" {ty} a{c} = {ty}(inp[i & 1023u] + {c}.0f);\n"
    body += f" for (int it = 0; it < {ITER}; ++it) {{\n"
    body += "".join("  " + stmt.format(c=c) + "\n" for c in range(CHAINS))
    body += " }\n"
    body += " out[i] = float(" + " + ".join(f"a{c}" for c in range(CHAINS)) + ");\n"
    k = mx.fast.metal_kernel(name=f"rate_{name}", input_names=["inp"], output_names=["out"], source=body)
    inp = mx.random.normal((1024,))
    run = lambda: mx.eval(k(inputs=[inp], grid=(THREADS, 1, 1), threadgroup=(256, 1, 1),
                            output_shapes=[(THREADS,)], output_dtypes=[mx.float32])[0])
    run()
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t)
    t = sorted(ts)[2]
    ops = THREADS * ITER * CHAINS
    return ops / t / 1e9


for name, (ty, stmt) in OPS.items():
    try:
        print(f"{name:5s} {rate(name, ty, stmt):8.1f} G lane-op/s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name:5s} failed: {str(e)[:100]}", flush=True)
