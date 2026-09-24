# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""TDT device-chain decision microbenchmarks (measurement-only, no installs).

Answers, on the installed stack, what a device-chained TDT decode pays:

  A/B. queued per-dispatch cost and one-eval scaling (96/384/960 dispatches
       in ONE mx.eval) — per-dispatch curve and any command-buffer cliff;
  C.   round-trip cost (one dispatch + one mx.eval, repeated);
  D.   single-workgroup streaming bandwidth on the joint matrix (10.5 MB,
       640x8198 fp16): today's scalar fp16 loads vs uint-packed
       (unpackHalf2x16) loads, 1 workgroup, 512/1024 threads;
  E.   grid streaming bandwidth: 8..33 workgroups, both load styles;
  F.   real kernel-shape costs queued back-to-back: chains (100x256),
       fold (1x640), joint scalar (33x256), joint packed (17x256);
  G.   argmax+control shape (1x1024, first-max scan over 7x8198) plus a
       tie/NaN spot check against np.argmax semantics.

Run inside a jwm1 window under flock /tmp/m1-gpu.lock with the installed
venv's python. No model cache needed.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np


def timed(fn, reps=5, warmup=2):
    best = float("inf")
    for _ in range(warmup):
        fn()
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        best = min(best, time.perf_counter_ns() - t0)
    return best / 1e6  # ms


K_TINY = mx.fast.metal_kernel(
    name="bench_tiny",
    input_names=["a", "b"],
    output_names=["o"],
    header="",
    source="""
    uint i = thread_index_in_threadgroup.x;
    if (i < 64u) { o[i] = a[i] + b[i]; }
    """,
    compile_options={"math_mode": "safe"},
)

# First-max argmax with np.argmax semantics: strict > on an ascending scan,
# smaller index wins ties, first NaN wins. One row per control row; results
# land in ctl[r+1] (ctl[0] reserved).
_REDUCE = """
        threadgroup float s_val[1024];
        threadgroup uint s_idx[1024];
        s_val[t] = bv; s_idx[t] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0u) {
            float v0 = s_val[0]; uint i0 = s_idx[0];
            for (uint i = 1u; i < 1024u; ++i) {
                float v = s_val[i]; uint ix = s_idx[i];
                bool take;
                if (v != v) { take = (v0 == v0) || (ix < i0); }
                else if (v0 != v0) { take = false; }
                else { take = (v > v0) || (v == v0 && ix < i0); }
                if (take) { v0 = v; i0 = ix; }
            }
            ctl[r + 1u] = int(i0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
"""

K_SCAN = mx.fast.metal_kernel(
    name="bench_scan",
    input_names=["logits", "n_rows"],
    output_names=["ctl"],
    header="",
    source=f"""
    uint t = thread_index_in_threadgroup.x;
    uint rows = uint(n_rows[0]);
    if (t == 0u) {{ ctl[0] = 0; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0u; r < rows; ++r) {{
        float bv = 0.0f; uint bi = 0u;
        for (uint j = t; j < 8198u; j += 1024u) {{
            float v = logits[r * 8198u + j];
            if ((j == t) || ((v != v) && (bv == bv)) || (v > bv)) {{
                bv = v; bi = j;
            }}
        }}
{_REDUCE}
    }}
    """,
    compile_options={"math_mode": "safe"},
)

# Window-shaped joint evaluation against N_ROWS consecutive encoder frames,
# decoder relu staged per row in threadgroup memory, weights streamed once.
# SCALAR variant: fp16 weights, one output lane per thread (today's shape).
def k_window_scalar(rows: int, threads: int):
    src = f"""
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint j = g * {threads}u + t;
    threadgroup float16_t sh_relu[{rows * 640}];
    for (uint slot = 0u; slot < {rows}u; ++slot) {{
        for (uint i = t; i < 640u; i += {threads}u) {{
            float16_t rlv = float16_t(enc[slot * 640u + i]) + pj[i];
            sh_relu[slot * 640u + i] =
                (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j < 8198u) {{
{chr(10).join(f"        precise float acc{slot} = 0.0f;" for slot in range(rows))}
        for (uint k = 0u; k < 640u; ++k) {{
            precise float w = float(joint[k * 8198u + j]);
{chr(10).join(f"            acc{slot} = acc{slot} + float(sh_relu[{slot}u * 640u + k]) * w;" for slot in range(rows))}
        }}
{chr(10).join(f"        out[{slot}u * 8198u + j] = float(float16_t(acc{slot}));" for slot in range(rows))}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"bench_wscalar_{rows}_{threads}",
        input_names=["joint", "pj", "enc"],
        output_names=["out"],
        header="",
        source=src,
        compile_options={"math_mode": "safe"},
    )


# PACKED variant: weights as uint32 (two fp16 per uint, unpackHalf2x16 is an
# exact fp16->fp32 widening), each thread owns two output lanes per uint,
# 8200 columns padded to a whole number of uints.
_WCOLS = 8200  # 8198 padded; rows past valid never queried by the walk
_WUINTS = _WCOLS // 2


def k_window_packed(rows: int, threads: int):
    src = f"""
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint ju = g * {threads}u + t;
    threadgroup float16_t sh_relu[{rows * 640}];
    for (uint slot = 0u; slot < {rows}u; ++slot) {{
        for (uint i = t; i < 640u; i += {threads}u) {{
            float16_t rlv = float16_t(enc[slot * 640u + i]) + pj[i];
            sh_relu[slot * 640u + i] =
                (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ju < {_WUINTS}u) {{
        uint j0 = ju * 2u;
{chr(10).join(f"        precise float acc{slot}a = 0.0f;\n        precise float acc{slot}b = 0.0f;" for slot in range(rows))}
        for (uint k = 0u; k < 640u; ++k) {{
            uint packed_w = joint[k * {_WUINTS}u + ju];
            vec2 p = unpackHalf2x16(packed_w);
{chr(10).join(f"            acc{slot}a = acc{slot}a + float(sh_relu[{slot}u * 640u + k]) * p.x;\n            acc{slot}b = acc{slot}b + float(sh_relu[{slot}u * 640u + k]) * p.y;" for slot in range(rows))}
        }}
{chr(10).join(f"        out[{slot}u * {_WCOLS}u + j0] = float(float16_t(acc{slot}a));\n        out[{slot}u * {_WCOLS}u + j0 + 1u] = float(float16_t(acc{slot}b));" for slot in range(rows))}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"bench_wpacked_{rows}_{threads}",
        input_names=["joint", "pj", "enc"],
        output_names=["out"],
        header="",
        source=src,
        compile_options={"math_mode": "safe"},
    )


def main() -> None:
    info = mx.device_info()
    print("device:", {k: info.get(k) for k in ("device_name", "driver", "simulated")})

    # ---- A/B: queued per-dispatch cost, tiny kernel, one eval per batch
    a = mx.array(np.zeros((64,), np.float32))
    b = mx.array(np.zeros((64,), np.float32))
    for batch in (96, 384, 960):
        def run_batch(batch=batch):
            outs = []
            for _ in range(batch):
                (o,) = K_TINY(inputs=[a, b], output_shapes=[(64,)],
                              output_dtypes=[mx.float32],
                              grid=(64, 1, 1), threadgroup=(64, 1, 1),
                              stream=mx.gpu)
                outs.append(o)
            mx.eval(*outs)
        ms = timed(run_batch, reps=7)
        print(f"A/B tiny-dispatch batch={batch:4d}: {ms:8.2f} ms total,"
              f" {ms / batch * 1000:6.1f} us/dispatch")

    # ---- C: round trip (one dispatch + eval)
    def round_trip():
        (o,) = K_TINY(inputs=[a, b], output_shapes=[(64,)],
                      output_dtypes=[mx.float32],
                      grid=(64, 1, 1), threadgroup=(64, 1, 1), stream=mx.gpu)
        mx.eval(o)
    ms = timed(round_trip, reps=30)
    print(f"C round trip (dispatch+eval): {ms * 1000:7.1f} us")

    # ---- shared data
    rng = np.random.default_rng(7)
    joint_h = rng.standard_normal((640, 8198)).astype(np.float16)
    joint = mx.array(joint_h)
    pad = np.zeros((640, _WCOLS - 8198), np.float16)
    u16 = np.concatenate([joint_h, pad], axis=1).view(np.uint16)
    u16 = u16.reshape(640, _WUINTS, 2)
    packed_u32 = (u16[:, :, 0].astype(np.uint32)
                  | (u16[:, :, 1].astype(np.uint32) << np.uint32(16)))
    packed = mx.array(np.ascontiguousarray(packed_u32.reshape(-1)))
    pj = mx.array(rng.standard_normal((640,)).astype(np.float16))
    enc = mx.array(rng.standard_normal((7 * 640,)).astype(np.float16))

    # ---- D/E: streaming bandwidth, scalar vs packed
    for threads in (512, 1024):
        k = k_window_scalar(1, threads)

        def run(k=k, threads=threads):
            (o,) = k(inputs=[joint, pj, enc], output_shapes=[(8198,)],
                     output_dtypes=[mx.float32],
                     grid=(threads, 1, 1), threadgroup=(threads, 1, 1),
                     stream=mx.gpu)
            mx.eval(o)
        ms = timed(run, reps=7)
        print(f"D scalar 1-wg threads={threads}: {ms:7.3f} ms"
              f"  {10.496 / ms:6.1f} GB/s")

    for threads in (512, 1024):
        k = k_window_packed(1, threads)

        def run(k=k, threads=threads):
            (o,) = k(inputs=[packed, pj, enc],
                     output_shapes=[(_WCOLS,)], output_dtypes=[mx.float32],
                     grid=(((_WUINTS + threads - 1) // threads) * threads, 1, 1),
                     threadgroup=(threads, 1, 1), stream=mx.gpu)
            mx.eval(o)
        ms = timed(run, reps=7)
        print(f"D packed 1-wg threads={threads}: {ms:7.3f} ms"
              f"  {10.496 / ms:6.1f} GB/s")

    for groups in (8, 16, 33):
        k = k_window_packed(1, 256)

        def run(groups=groups, k=k):
            (o,) = k(inputs=[packed, pj, enc],
                     output_shapes=[(_WCOLS,)], output_dtypes=[mx.float32],
                     grid=(groups * 256, 1, 1), threadgroup=(256, 1, 1),
                     stream=mx.gpu)
            mx.eval(o)
        ms = timed(run, reps=7)
        print(f"E packed grid groups={groups:3d}: {ms:7.3f} ms"
              f"  {10.496 / ms:6.1f} GB/s")

    # ---- F: real kernel-shape costs queued back-to-back (one eval, x50)
    w_lay = mx.array(rng.standard_normal((640, 2560)).astype(np.float16))
    emb = mx.array(np.zeros((8193, 640), np.float16))
    hid = mx.array(np.zeros((1280,), np.float32))
    xin = mx.array(np.zeros((640,), np.float32))
    bsum = mx.array(np.zeros((25600,), np.float16))
    k_chains = mx.fast.metal_kernel(
        name="bench_chains",
        input_names=["w", "emb", "hid", "xin"],
        output_names=["bsum"],
        header="",
        source="""
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint idx = g * 256u + t;
    threadgroup float16_t sh_a[1280];
    for (uint i = t; i < 1280u; i += 256u) {
        sh_a[i] = (i < 640u) ? float16_t(xin[i]) : float16_t(hid[i - 640u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n = idx % 2560u;
    uint gb = idx / 2560u;
    uint block = gb % 10u;
    float16_t bc = float16_t(0.0f);
    uint k = block * 128u;
    for (uint j = 0u; j < 128u; ++j) {
        bc = float16_t(bc + float16_t(sh_a[k] * w[k * 2560u + n]));
        ++k;
    }
    bsum[idx] = bc;
    """,
        compile_options={"math_mode": "safe"},
    )

    def run_chains():
        outs = []
        for _ in range(50):
            (o,) = k_chains(inputs=[w_lay, emb, hid, xin],
                            output_shapes=[(25600,)], output_dtypes=[mx.float16],
                            grid=(100 * 256, 1, 1), threadgroup=(256, 1, 1),
                            stream=mx.gpu)
            outs.append(o)
        mx.eval(*outs)
    ms = timed(run_chains, reps=5)
    print(f"F chains-shape (100x256, 3.3 MB): {ms / 50 * 1000:7.1f} us/dispatch")

    k_fold = mx.fast.metal_kernel(
        name="bench_fold",
        input_names=["bsum", "bi", "lut", "cell"],
        output_names=["h1o", "c1o"],
        header="",
        source="""
    uint t = thread_index_in_threadgroup.x;
    float16_t pr[4];
    for (uint gate = 0u; gate < 4u; ++gate) {
        uint n = t + gate * 640u;
        float16_t bacc = bsum[n];
        for (uint block = 1u; block < 10u; ++block) {
            bacc = float16_t(bacc + bsum[block * 2560u + n]);
        }
        pr[gate] = float16_t(bacc + bi[n]);
    }
    uint bi_i = packHalf2x16(vec2(float(pr[0]), 0.0f)) & 0xFFFFu;
    uint bf_i = packHalf2x16(vec2(float(pr[1]), 0.0f)) & 0xFFFFu;
    uint bo_i = packHalf2x16(vec2(float(pr[2]), 0.0f)) & 0xFFFFu;
    uint bg_i = packHalf2x16(vec2(float(pr[3]), 0.0f)) & 0xFFFFu;
    float16_t si = lut[bi_i];
    float16_t sf = lut[bf_i];
    float16_t so = lut[bo_i];
    float16_t tg = lut[65536u + bg_i];
    float16_t c0 = float16_t(cell[t]);
    float16_t c1 = sf * c0;
    c1 = float16_t(c1 * si + tg);
    h1o[t] = float(so * c1);
    c1o[t] = float(c1);
    """,
        compile_options={"math_mode": "safe"},
    )
    bi = mx.array(np.zeros((2560,), np.float16))
    lut = mx.array(np.zeros((131072,), np.float16))
    cell = mx.array(np.zeros((1280,), np.float32))

    def run_fold():
        outs = []
        for _ in range(50):
            h1, c1 = k_fold(inputs=[bsum, bi, lut, cell],
                            output_shapes=[(640,), (640,)],
                            output_dtypes=[mx.float32, mx.float32],
                            grid=(640, 1, 1), threadgroup=(640, 1, 1),
                            stream=mx.gpu)
            outs.append(h1)
        mx.eval(*outs)
    ms = timed(run_fold, reps=5)
    print(f"F fold-shape (1x640):             {ms / 50 * 1000:7.1f} us/dispatch")

    for label, k, src, cols, groups in (
        ("W scalar 7r 33x256", k_window_scalar(7, 256), joint, 7 * 8198, 33),
        ("W scalar 1r 33x256", k_window_scalar(1, 256), joint, 1 * 8198, 33),
    ):
        def run(k=k, src=src, cols=cols, groups=groups):
            outs = []
            for _ in range(50):
                (o,) = k(inputs=[src, pj, enc], output_shapes=[(cols,)],
                         output_dtypes=[mx.float32],
                         grid=(groups * 256, 1, 1), threadgroup=(256, 1, 1),
                         stream=mx.gpu)
                outs.append(o)
            mx.eval(*outs)
        try:
            ms = timed(run, reps=5)
            print(f"F {label}: {ms / 50 * 1000:7.1f} us/dispatch"
                  f"  ({10.496 / (ms / 50):6.2f} GB/s-equiv)")
        except Exception as exc:
            print(f"F {label}: FAILED {exc}")

    # ---- H: order-preserving load hoisting (the bit-exact unroll):
    # loads are independent so they can run ahead; the fp32 adds stay in
    # ascending k order, so the result is bit-identical to the plain chain.
    K_JOINT_HOIST = mx.fast.metal_kernel(
        name="bench_joint_hoist4",
        input_names=["joint", "pj", "enc"],
        output_names=["out"],
        header="",
        source="""
    uint g = threadgroup_position_in_grid.x;
    uint t = thread_index_in_threadgroup.x;
    uint j = g * 256u + t;
    threadgroup float16_t sh_relu[640];
    for (uint i = t; i < 640u; i += 256u) {
        float16_t rlv = float16_t(enc[i]) + pj[i];
        sh_relu[i] = (rlv > float16_t(0.0f)) ? rlv : float16_t(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j < 8198u) {
        precise float acc = 0.0f;
        uint k = 0u;
        for (uint kb = 0u; kb < 160u; ++kb) {
            precise float w0 = float(joint[k * 8198u + j]);
            precise float w1 = float(joint[(k+1u) * 8198u + j]);
            precise float w2 = float(joint[(k+2u) * 8198u + j]);
            precise float w3 = float(joint[(k+3u) * 8198u + j]);
            precise float r0 = float(sh_relu[k]);
            precise float r1 = float(sh_relu[k+1u]);
            precise float r2 = float(sh_relu[k+2u]);
            precise float r3 = float(sh_relu[k+3u]);
            acc = acc + r0 * w0;
            acc = acc + r1 * w1;
            acc = acc + r2 * w2;
            acc = acc + r3 * w3;
            k += 4u;
        }
        out[j] = float(float16_t(acc));
    }
    """,
        compile_options={"math_mode": "safe"},
    )

    def run_hoist():
        outs = []
        for _ in range(50):
            (o,) = K_JOINT_HOIST(inputs=[joint, pj, enc],
                                 output_shapes=[(8198,)],
                                 output_dtypes=[mx.float32],
                                 grid=(33 * 256, 1, 1),
                                 threadgroup=(256, 1, 1), stream=mx.gpu)
            outs.append(o)
        mx.eval(*outs)
    try:
        ms = timed(run_hoist, reps=5)
        print(f"H joint 1r hoist4 33x256: {ms / 50 * 1000:7.1f} us/dispatch"
              f"  ({10.496 / (ms / 50):6.2f} GB/s-equiv)")
    except Exception as exc:
        print(f"H joint hoist: FAILED {exc}")

    # ---- G: argmax+control shape (1x1024 scanning 7x8198)
    logits = mx.array(rng.standard_normal((7, 8198)).astype(np.float32))
    nrows = mx.array(np.array([7], np.int32))

    def run_scan():
        outs = []
        for _ in range(50):
            (o,) = K_SCAN(inputs=[logits, nrows], output_shapes=[(8,)],
                          output_dtypes=[mx.int32],
                          grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                          stream=mx.gpu)
            outs.append(o)
        mx.eval(*outs)
    ms = timed(run_scan, reps=5)
    print(f"G argmax+control shape (1x1024):  {ms / 50 * 1000:7.1f} us/dispatch")

    lg = np.zeros((7, 8198), np.float32)
    lg[0, 42] = 3.0
    lg[0, 7] = 3.0  # tie: smaller index must win
    lg[1, 8193] = 9.0  # duration column reachable too
    (o,) = K_SCAN(inputs=[mx.array(lg), nrows], output_shapes=[(8,)],
                  output_dtypes=[mx.int32], grid=(1024, 1, 1),
                  threadgroup=(1024, 1, 1), stream=mx.gpu)
    mx.eval(o)
    got = list(map(int, np.asarray(o)))
    want = [0, 7, 8193] + [0] * 5
    print(f"G scan spot check: {'PASS' if got == want else f'FAIL got {got} want {want}'}")


if __name__ == "__main__":
    main()
