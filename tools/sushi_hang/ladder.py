"""Sushi first-compiled-forward hang: minimal repro ladder (MLX-level, no model pack).

Run one rung per process:  python ladder.py <rung>
Rungs:
  0  eager control (bf16 matmul + eval)         - GPU healthy baseline
  1  mx.compile fused elementwise chain (bf16)
  2  mx.compile GDN-gate-like function          - softplus/sigmoid/exp mirror
  3  mx.compile MoE routing (softmax/topk/take) - qwen4-ish dims
  4  mx.fast.gated_delta_update PREFILL, eager  - T=4304 qwen4-ish GDN shape
  5  mx.fast.metal_kernel custom kernel, eager  - loop-bearing translated path
  6  mx.compile mixing custom kernel + quantized matmul + routing
Each stage prints START/DONE markers flushed immediately; a hang leaves the last
marker as the diagnosis anchor.
"""
import sys
import time

import mlx.core as mx

T0 = time.time()


def mark(msg):
    print("[%7.2f] %s" % (time.time() - T0, msg), flush=True)


def rung0():
    a = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    b = mx.random.normal((4096, 4096)).astype(mx.bfloat16)
    mark("r0 eval eager matmul")
    c = mx.eval(a @ b)
    mark("r0 done %s" % (c.shape,))


def rung1():
    @mx.compile
    def chain(x):
        y = mx.exp(x) * mx.sigmoid(x) + mx.log(mx.abs(x) + 1.0)
        z = mx.erf(y) * y - mx.sigmoid(y)
        return z * z

    x = mx.random.normal((4304, 4096)).astype(mx.bfloat16)
    mark("r1 eval compiled chain")
    mx.eval(chain(x))
    mark("r1 done")


def rung2():
    # GDN gate: mirrors the qwen4 GatedDeltaNet gate prologue the backend's
    # gated_delta_decode/prefill kernels replicate (softplus + exp(A_log) + sigmoid).
    @mx.compile
    def gdn_gate(x, a_log, dt_bias):
        xx = (x + dt_bias).astype(mx.bfloat16)
        sp = mx.logaddexp(0.0, xx)                      # softplus
        g = -mx.exp(a_log.astype(mx.float32)) * sp
        beta = mx.sigmoid((x * 0.5).astype(mx.bfloat16))
        return g.astype(mx.bfloat16), beta

    T = 4304
    x = mx.random.normal((1, T, 16)).astype(mx.bfloat16)
    a_log = mx.random.normal((16,)).astype(mx.bfloat16)
    dt_bias = mx.random.normal((16,)).astype(mx.bfloat16)
    mark("r2 eval compiled gdn gate")
    g, beta = gdn_gate(x, a_log, dt_bias)
    mx.eval(g, beta)
    mark("r2 done")


def rung3():
    E, TOPK, H = 128, 10, 4096

    @mx.compile
    def route(x, wg):
        logits = (x @ wg.T).astype(mx.float32)
        p = mx.softmax(logits + 0.1 * mx.random.normal(logits.shape), axis=-1) \
            if False else mx.softmax(logits, axis=-1)
        idx = mx.argpartition(p, kth=-TOPK, axis=-1)[..., -TOPK:]
        top = mx.take_along_axis(p, idx, axis=-1)
        top = top / mx.sum(top, axis=-1, keepdims=True)
        return top.astype(mx.bfloat16), idx

    x = mx.random.normal((4304, H)).astype(mx.bfloat16)
    wg = mx.random.normal((E, H)).astype(mx.bfloat16)
    mark("r3 eval compiled moe routing")
    mx.eval(*route(x, wg))
    mark("r3 done")


def rung4():
    # mx.fast.gated_delta_update prefill at qwen4-ish GDN shape:
    # square bf16 Dk=Dv=128, Hk=16 heads, scalar g, T=4304 (prefill width).
    import inspect
    f = mx.fast.gated_delta_update
    doc = getattr(f, "__doc__", "") or ""
    print("sig doc:", doc.strip().splitlines()[:6], flush=True)
    Hk, Dk, Dv, T = 16, 128, 128, 4304
    q = mx.random.normal((1, Hk, T, Dk)).astype(mx.bfloat16)
    k = mx.random.normal((1, Hk, T, Dk)).astype(mx.bfloat16)
    v = mx.random.normal((1, Hk, T, Dv)).astype(mx.bfloat16)
    state = mx.zeros((1, Hk, Dk, Dv), mx.float32)
    # gate/beta as produced by rung2's compiled function shape [1,T,H] bf16
    g = mx.random.normal((1, T, Hk)).astype(mx.bfloat16) - 2.0
    beta = mx.sigmoid(mx.random.normal((1, T, Hk))).astype(mx.bfloat16)
    try:
        mark("r4 eval gated_delta_update prefill")
        out, new_state = mx.fast.gated_delta_update(q, k, v, state, g, beta)
        mx.eval(out, new_state)
        mark("r4 done %s %s" % (out.shape, new_state.shape))
    except TypeError:
        mark("r4 signature mismatch, trying raw")
        out = mx.fast.gated_delta_update_raw(q, k, v, state, g, beta)
        mx.eval(out)
        mark("r4 raw done %s" % (out.shape,))


def rung5():
    from mlx.core.fast import metal_kernel
    # Loop-bearing kernel in the style the EXL3/GDN custom sites translate:
    # per-thread serial accumulation loop over K, bf16 in, bf16 out.
    src = """
        uint row = (thread_position_in_grid.x % 4304) * 128u;
        float acc = 0.0;
        for (uint i = 0u; i < 128u; ++i) {
            acc += float(x[row + i]) * float(w[row + i]);
        }
        out[thread_position_in_grid.x] = bfloat16(acc);
    """
    loop_accum = metal_kernel(
        name="loop_accum",
        input_names=["x", "w"],
        output_names=["out"],
        source=src,
        ensure_row_contiguous=True,
    )

    x = mx.random.normal((4304, 128)).astype(mx.bfloat16)
    w = mx.random.normal((4304, 128)).astype(mx.bfloat16)
    mark("r5 eval custom metal kernel")
    res = loop_accum(inputs=[x, w], grid=(4304, 1, 1), threadgroup=(64, 1, 1),
                     output_shapes=[(4304,)], output_dtypes=[mx.bfloat16],
                     verbose=True)
    mx.eval(res[0])
    mark("r5 done %s" % (res[0].shape,))


def rung6():
    from mlx.core.fast import metal_kernel

    @mx.compile
    def route_top(x, wg, topk=10):
        logits = (x @ wg.T).astype(mx.float32)
        p = mx.softmax(logits, axis=-1)
        idx = mx.argpartition(p, kth=-topk, axis=-1)[..., -topk:]
        return mx.take_along_axis(p, idx, axis=-1).astype(mx.bfloat16), idx

    src = """
        float acc = 1.0;
        for (uint i = 0u; i < 4u; ++i) {
            acc *= float(g[thread_position_in_grid.x * 4u + i]);
        }
        out[thread_position_in_grid.x] = bfloat16(acc);
    """
    gate_mix = metal_kernel(
        name="gate_mix",
        input_names=["g"],
        output_names=["out"],
        source=src,
        ensure_row_contiguous=True,
    )

    H, E, TOPK = 4096, 128, 10
    x = mx.random.normal((4304, H)).astype(mx.bfloat16)
    wg = mx.random.normal((E, H)).astype(mx.bfloat16)
    mark("r6 eval compiled mix (routing + custom kernel + quantized matmul)")
    top, idx = route_top(x, wg)

    wq = mx.quantize(mx.random.normal((E, H)).astype(mx.bfloat16), group_size=64, bits=4)
    sel = x[:4304]  # gather experts happens host-side in sushi's plain path
    qmm = mx.quantized_matmul(sel, wq[0], wq[1], wq[2], group_size=64, bits=4)
    mixed = top[:, :4] + qmm[:, :E].mean(axis=-1, keepdims=True).astype(mx.bfloat16)
    gm = gate_mix(inputs=[mx.reshape(mixed, (4304, 4))], grid=(4304, 1, 1),
                  threadgroup=(64, 1, 1), output_shapes=[(4304,)],
                  output_dtypes=[mx.bfloat16])
    mx.eval(gm[0], top, idx)
    mark("r6 done")


def rung7():
    # Prefill-forward-like: one big mixed eval (rope + kv-cache slice write +
    # compiled routing + qmm + norm), the way the first prefill step spans
    # graph-eval boundaries under the batch-first machinery.
    H, KVH, T, DK = 4096, 8, 4304, 128
    x = mx.random.normal((1, T, H)).astype(mx.bfloat16)
    wq = mx.random.normal((H, 4096)).astype(mx.bfloat16)  # qkv fused-ish width
    wk = mx.random.normal((KVH, DK, H)).astype(mx.bfloat16)
    wq_s = mx.random.normal((H, 4096)).astype(mx.bfloat16)

    @mx.compile
    def route(x, wg):
        p = mx.softmax((x @ wg.T).astype(mx.float32), axis=-1)
        idx = mx.argpartition(p, kth=-10, axis=-1)[..., -10:]
        return mx.take_along_axis(p, idx, axis=-1).astype(mx.bfloat16), idx

    wg = mx.random.normal((128, H)).astype(mx.bfloat16)
    cache_k = mx.zeros((1, KVH, 8192, DK), mx.float32).astype(mx.bfloat16)
    # rope
    mark("r7 build")
    q = (x @ wq).reshape(1, T, 32, 128)
    qr = mx.fast.rope(q.astype(mx.bfloat16), 128, base=10000.0, traditional=False,
                     offset=0, scale=1.0)
    kr = qr
    ks = (x.reshape(1, T, KVH, 1, DK) * wk.transpose(0, 2, 1).reshape(
        1, 1, KVH, DK, H % H if False else H)).sum(-1).astype(mx.bfloat16)
    cache_k[:, :, :T] = ks  # slice-update (kv cache write path)
    top, idx = route(x[:, -1], wg)
    wqq = mx.quantize(mx.random.normal((4096, H)).astype(mx.bfloat16),
                      group_size=64, bits=4)
    qmm = mx.quantized_matmul(mx.reshape(x[:, -1], (1, H)), wqq[0], wqq[1],
                              wqq[2], group_size=64, bits=4)
    n = mx.rms_norm(qr.astype(mx.float32)[:, :, :, :1] + qmm.reshape(1, 1, 1, 1),
                    mx.ones((1,)))
    mark("r7 eval mixed prefill")
    mx.eval(n, cache_k, top, idx, kr)
    mark("r7 done")


def rung8():
    # Faithful GDN layer (qwen4, Linux path = plain ops): hc tails compiled +
    # qkv proj + GDN plain chunked recurrence + out proj, at qwen4-ish dims,
    # chunk evals of 64 with a cadence partial eval.
    HK, HV, DK, DV, H, T, CHUNK = 16, 32, 128, 128, 4096, 4304, 64

    @mx.compile
    def gate(A_log, a, dt_bias):
        x = (a + dt_bias).astype(mx.bfloat16)
        g = -mx.exp(A_log.astype(mx.float32)) * mx.logaddexp(0.0, x)
        return g.astype(mx.bfloat16)

    qkv_w = mx.random.normal((2 * HK * DK + HV * DV, H)).astype(mx.bfloat16) * 0.02
    out_w = mx.random.normal((H, HV * DV)).astype(mx.bfloat16) * 0.02
    A_log = mx.random.normal((HV,)).astype(mx.bfloat16)
    dt_bias = mx.random.normal((HV,)).astype(mx.bfloat16)
    x = mx.random.normal((1, T, H)).astype(mx.bfloat16)
    state = mx.zeros((1, HV, DV, DK), mx.float32)

    mark("r8 gdn layer start")
    qkv = (x @ qkv_w.T).reshape(1, T, 2 * HK + HV, DK)
    q, k = qkv[:, :, :HK], qkv[:, :, HK:2 * HK]
    vpart = qkv[:, :, 2 * HK:]
    v = vpart.reshape(1, T, HV, DV)
    a = mx.mean(v, axis=-1)
    b = mx.sigmoid(mx.mean(v, axis=-1))
    g = gate(A_log, a, dt_bias)

    # heads-major + chunked plain recurrence (the Zig chunkedRecur op mix)
    qh = mx.transpose(mx.reshape(q, (1, T, HK, DK)), (0, 2, 1, 3))
    kh = mx.transpose(mx.reshape(k, (1, T, HK, DK)), (0, 2, 1, 3))
    vh = mx.transpose(v, (0, 2, 1, 3))
    gh = mx.transpose(g, (0, 2, 1))
    bh = mx.transpose(b, (0, 2, 1))
    rep = HV // HK
    qg = mx.repeat(qh, rep, axis=1)
    kg = mx.repeat(kh, rep, axis=1)

    TP = ((T + CHUNK - 1) // CHUNK) * CHUNK
    pad = TP - T
    if pad:
        qh = mx.pad(qh, [(0, 0), (0, 0), (0, pad), (0, 0)])
        kh = mx.pad(kh, [(0, 0), (0, 0), (0, pad), (0, 0)])
        vh = mx.pad(vh, [(0, 0), (0, 0), (0, pad), (0, 0)])
        gh = mx.pad(gh, [(0, 0), (0, 0), (0, pad)])
        bh = mx.pad(bh, [(0, 0), (0, 0), (0, pad)])
    logg = mx.log(mx.maximum(gh.astype(mx.float32), 1e-30))
    lfull = mx.cumsum(logg, axis=2)

    ys = []
    st = state.astype(mx.float32)
    for c0 in range(0, TP, CHUNK):
        lc = lfull[:, :, c0:c0 + CHUNK]
        qc = qg[:, :, c0:c0 + CHUNK]
        vc = vh[:, :, c0:c0 + CHUNK]
        gc = mx.exp(lc - mx.max(lc, axis=2, keepdims=True)).astype(mx.bfloat16)
        bc = bh[:, :, c0:c0 + CHUNK]
        # within-chunk: decayed state matmul + delta-rule outer updates (f32)
        attn = kg[:, :, c0:c0 + CHUNK] @ mx.transpose(qc, (0, 1, 3, 2))
        decay = mx.exp(lc[..., None] - lc[:, :, None, :])
        A = (attn * decay).astype(mx.float32)
        U = mx.linalg.solve_triangular(
            mx.eye(CHUNK) - A, (vc.astype(mx.float32) * bc[..., None]),
            upper=True, stream=mx.gpu) if False else (vc.astype(mx.float32) * bc[..., None])
        st = st * mx.exp(lc[:, :, -1:, None]).astype(mx.float32)
        y_c = mx.transpose(st @ mx.transpose(qc, (0, 1, 3, 2)), (0, 1, 3, 2))
        ys.append(mx.transpose(y_c, (0, 2, 1, 3)).astype(mx.bfloat16))
        st = st.astype(mx.bfloat16).astype(mx.float32)
        # cadence partial eval every 4 chunks (the Zig eval cadence)
        if (c0 // CHUNK) % 4 == 3:
            mx.eval(st)
    y = mx.concatenate(ys, axis=1)[:, :T]
    out = (mx.reshape(y, (1, T, HV * DV)) @ out_w.T)
    mark("r8 eval gdn layer out")
    mx.eval(out)
    mark("r8 done")


def rung9():
    # Full-attention layer: M-RoPE-like offset rope + kv8 quantized cache slice
    # write + SDPA prefill + out proj at qwen4-ish dims.
    H, KVH, NH, T, DK = 4096, 8, 32, 4304, 128
    wq = mx.random.normal((NH * DK, H)).astype(mx.bfloat16) * 0.02
    wk = mx.random.normal((KVH * DK, H)).astype(mx.bfloat16) * 0.02
    wv = mx.random.normal((KVH * DK, H)).astype(mx.bfloat16) * 0.02
    wo = mx.random.normal((H, NH * DK)).astype(mx.bfloat16) * 0.02
    x = mx.random.normal((1, T, H)).astype(mx.bfloat16)
    cache_k = mx.zeros((1, KVH, 8192, DK), mx.bfloat16)
    cache_v = mx.zeros((1, KVH, 8192, DK), mx.bfloat16)
    kq = mx.zeros((1, KVH, 8192, DK), mx.int8)
    vq = mx.zeros((1, KVH, 8192, DK), mx.int8)
    ks = mx.ones((1, KVH, 8192, 1), mx.bfloat16)
    vs = mx.ones((1, KVH, 8192, 1), mx.bfloat16)

    mark("r9 attn layer start")
    q = (x @ wq.T).reshape(1, T, NH, DK)
    k = ((x @ wk.T).reshape(1, T, KVH, DK))
    v = ((x @ wv.T).reshape(1, T, KVH, DK))
    qr = mx.fast.rope(q, DK, base=10000.0, traditional=False, offset=0, scale=1.0)
    kr = mx.fast.rope(k, DK, base=10000.0, traditional=False, offset=0, scale=1.0)
    # kv cache slice writes (bf16); quantized mirrors kept as packed round trip
    cache_k[:, :, :T] = mx.transpose(kr, (0, 2, 1, 3))
    cache_v[:, :, :T] = mx.transpose(v, (0, 2, 1, 3))
    kqw, kqs, _ = mx.quantize(kr, group_size=32, bits=8)
    kd = mx.dequantize(kqw, kqs, group_size=32, bits=8)
    vqw, vqs, _ = mx.quantize(v, group_size=32, bits=8)
    vd = mx.dequantize(vqw, vqs, group_size=32, bits=8)
    qh = mx.transpose(qr, (0, 2, 1, 3))
    kh = mx.transpose(kd, (0, 2, 1, 3))
    vh = mx.transpose(vd, (0, 2, 1, 3))
    o = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=1.0 / DK ** 0.5)
    out = mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (1, T, NH * DK)) @ wo.T
    mark("r9 eval attn layer")
    mx.eval(out, cache_k, cache_v)
    mark("r9 done")


def rung10():
    # Whole stack: alternating GDN/full layers, hc tails compiled, PLE add,
    # compiled routing, cadence evals, then a decode step (qmm + argmax + host
    # scalar read) -- the first-forward shape end to end.
    L, H, KVH, NH, HK, HV, DK, DV, T = 12, 4096, 8, 32, 16, 32, 128, 128, 4304

    @mx.compile
    def route(x, wg):
        p = mx.softmax((x @ wg.T).astype(mx.float32), axis=-1)
        idx = mx.argpartition(p, kth=-10, axis=-1)[..., -10:]
        top = mx.take_along_axis(p, idx, axis=-1)
        return (top / mx.sum(top, axis=-1, keepdims=True)).astype(mx.bfloat16), idx

    @mx.compile
    def hc_silu(x):
        return mx.silu(x).astype(mx.bfloat16)

    rng = 0.02
    wqkv = mx.random.normal((2 * HK * DK + HV * DV, H)).astype(mx.bfloat16) * rng
    wgq = mx.random.normal((NH * DK, H)).astype(mx.bfloat16) * rng
    wgk = mx.random.normal((KVH * DK, H)).astype(mx.bfloat16) * rng
    wgv = mx.random.normal((KVH * DK, H)).astype(mx.bfloat16) * rng
    wgo = mx.random.normal((H, NH * DK)).astype(mx.bfloat16) * rng
    wgdn_o = mx.random.normal((H, HV * DV)).astype(mx.bfloat16) * rng
    wgate = mx.random.normal((128, H)).astype(mx.bfloat16)
    w_up = mx.random.normal((512, H)).astype(mx.bfloat16) * rng
    w_dn = mx.random.normal((H, 256)).astype(mx.bfloat16) * rng
    A_log = mx.random.normal((HV,)).astype(mx.bfloat16)
    dt_bias = mx.random.normal((HV,)).astype(mx.bfloat16)
    emb = mx.random.normal((256000, H)).astype(mx.bfloat16) * 0.01
    cache_k = mx.zeros((1, KVH, 8192, DK), mx.bfloat16)
    cache_v = mx.zeros((1, KVH, 8192, DK), mx.bfloat16)
    ids = mx.random.randint(0, 255999, (1, T))
    lm_w = mx.quantize(mx.random.normal((256000, 512)).astype(mx.bfloat16),
                       group_size=64, bits=4)

    mark("r10 stack start")
    x = emb[ids]
    h = mx.tile(x, (1, 1, 4))
    for li in range(L):
        full = li % 2 == 0
        mixed = mx.reshape(h, (1, T, 4, H))[:, :, 0]
        if full:
            q = mx.fast.rope((mixed @ wgq.T).reshape(1, T, NH, DK), DK,
                        base=10000.0, traditional=False, offset=0, scale=1.0)
            k = mx.fast.rope((mixed @ wgk.T).reshape(1, T, KVH, DK), DK,
                        base=10000.0, traditional=False, offset=0, scale=1.0)
            v = (mixed @ wgv.T).reshape(1, T, KVH, DK)
            cache_k[:, :, :T] = mx.transpose(k, (0, 2, 1, 3))
            cache_v[:, :, :T] = mx.transpose(v, (0, 2, 1, 3))
            o = mx.fast.scaled_dot_product_attention(
                mx.transpose(q, (0, 2, 1, 3)),
                mx.transpose(k, (0, 2, 1, 3)),
                mx.transpose(v, (0, 2, 1, 3)), scale=0.09)
            attn = mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (1, T, NH * DK)) @ wgo.T
        else:
            qkv = (mixed @ wqkv.T).reshape(1, T, 2 * HK + HV, DK)
            a = mx.mean(qkv[:, :, HK:2 * HK], axis=-1)
            b = mx.sigmoid(mx.mean(qkv[:, :, 2 * HK:], axis=-1))
            g = (-mx.exp(A_log.astype(mx.float32)) *
                 mx.logaddexp(0.0, (a + dt_bias).astype(mx.bfloat16))).astype(mx.bfloat16)
            qh = mx.repeat(mx.transpose(mx.reshape(qkv[:, :, :HK], (1, T, HK, DK)), (0, 2, 1, 3)), HV // HK, axis=1)
            kh2 = mx.repeat(mx.transpose(mx.reshape(qkv[:, :, HK:2 * HK], (1, T, HK, DK)), (0, 2, 1, 3)), HV // HK, axis=1)
            vh2 = mx.transpose(qkv[:, :, 2 * HK:], (0, 2, 1, 3))
            gh = mx.transpose(g, (0, 2, 1)).astype(mx.float32)
            st = vh2.astype(mx.float32) * 0.0
            dec = mx.exp(gh)[:, :, :, None]
            kv = (st * dec) @ mx.transpose(kh2, (0, 1, 3, 2))
            attn = mx.transpose((kv @ mx.transpose(qh, (0, 1, 3, 2))), (0, 2, 1, 3))
            attn = mx.reshape(attn, (1, T, HV * DV)) @ wgdn_o.T
        gate_in = mixed @ w_up.T
        moe_t, idx = route(mixed, wgate)
        expert = mx.mean(mx.reshape(gate_in * moe_t.repeat(4, axis=-1)[:, :, :512], (1, T, 4, 128)), axis=2)
        ff = (mx.reshape(expert, (1, T, 4, 128)).mean(axis=2) @ w_dn.T)
        inj = hc_silu(attn + ff)
        h = h + mx.tile(inj, (1, 1, 4))
        if li % 4 == 3:
            mx.eval(h)  # cadence
    last = mx.reshape(h[:, -1], (1, 4, H))[:, 0]
    lw = lm_w[0]
    logits = mx.quantized_matmul(last, lw, lm_w[1], lm_w[2], group_size=64, bits=4)
    tok = mx.argmax(logits, axis=-1)
    mark("r10 eval stack + decode head")
    mx.eval(h, tok)
    tid = int(tok.item())
    mark("r10 done tok=%d" % tid)


def rung11():
    # Faithful chunkedRecur op pattern (qwen4 GDN prefill, Linux plain path):
    # per chunk: decay mul, TWO batched GEMV matmuls [1,32,128,1], sub/mul/
    # outer/add, bf16 state round-trip per chunk, PART CONCAT at the end, one
    # EVAL PER CHUNK (the Zig bounds the lazy graph per chunk).
    HK, HV, DK, DV, T, CHUNK = 16, 32, 128, 128, 4304, 64
    TP = ((T + CHUNK - 1) // CHUNK) * CHUNK
    pad = TP - T
    q = mx.random.normal((1, TP, HV, DK)).astype(mx.bfloat16) * 0.05
    k = mx.random.normal((1, TP, HV, DK)).astype(mx.bfloat16) * 0.05
    v = mx.random.normal((1, TP, HV, DV)).astype(mx.bfloat16) * 0.05
    g = mx.random.normal((1, TP, HV)).astype(mx.bfloat16) - 2.0
    beta = mx.sigmoid(mx.random.normal((1, TP, HV))).astype(mx.bfloat16)
    state = mx.zeros((1, HV, DV, DK), mx.float32)

    mark("r11 chunked recurrence start")
    logg = mx.log(mx.maximum(g.astype(mx.float32), 1e-30))
    lfull = mx.cumsum(logg, axis=1)
    ys = []
    st = state
    for c0 in range(0, TP, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        lc = lfull[:, sl]
        k_t = mx.transpose(k[:, sl], (0, 1, 3, 2))          # [1,Hv,Dk,C]
        q_t = mx.transpose(q[:, sl], (0, 1, 3, 2))          # [1,Hv,Dk,C]
        v_t = mx.transpose(v[:, sl], (0, 1, 3, 2))          # [1,Hv,Dv,C]
        dec = mx.exp(lc[..., None]).astype(mx.bfloat16)     # [1,C,Hv,1]
        # per-token loop inside the chunk, the serial body with f32<->bf16
        for ti in range(CHUNK):
            dec_t = dec[:, ti][:, :, None]                   # [1,Hv,1,1]
            k1 = k[:, c0 + ti][..., None]                    # [1,Hv,Dk,1]
            q1 = q[:, c0 + ti][..., None]
            v1 = v[:, c0 + ti][..., None]                    # [1,Hv,Dv,1]
            decayed = st * dec_t
            kv = decayed @ k1                                # [1,Hv,Dv,1]
            delta = (v1 - kv) * beta[:, c0 + ti][..., None, None]
            st = decayed + mx.transpose(k1, (0, 1, 3, 2)) @ delta
            y = (st @ q1)                                    # [1,Hv,Dv,1]
            ys.append(y)
            st = st.astype(mx.bfloat16).astype(mx.float32)   # the rounding contract
        _ = k_t, q_t, v_t, lc
    y = mx.concatenate([mx.reshape(t, (1, 1, HV, DV)) for t in ys], axis=1)
    y = mx.reshape(y[:, :T], (1, T, HV * DV))
    out_w = mx.random.normal((4096, HV * DV)).astype(mx.bfloat16) * 0.02
    out = y @ out_w.T
    mark("r11 eval recurrence+proj")
    mx.eval(out)
    mark("r11 done")


def rung12():
    # serialRecur small width (t=5, one eval, no intermediate evals).
    HKV = 32
    DK, DV, t = 128, 128, 5
    q = mx.random.normal((1, t, HKV, DK)).astype(mx.bfloat16) * 0.05
    k = mx.random.normal((1, t, HKV, DK)).astype(mx.bfloat16) * 0.05
    v = mx.random.normal((1, t, HKV, DV)).astype(mx.bfloat16) * 0.05
    g = mx.random.normal((1, t, HKV)).astype(mx.bfloat16) - 2.0
    beta = mx.sigmoid(mx.random.normal((1, t, HKV))).astype(mx.bfloat16)
    st = mx.zeros((1, HKV, DV, DK), mx.float32)
    ys = []
    for ti in range(t):
        dec = mx.exp(g[:, ti].astype(mx.float32))[:, :, None, None]
        k1 = k[:, ti][..., None]
        q1 = q[:, ti][..., None]
        v1 = v[:, ti][..., None]
        decayed = st * dec
        kv = decayed @ k1
        delta = (v1 - kv) * beta[:, ti][..., None, None]
        st = decayed + mx.transpose(k1, (0, 1, 3, 2)) @ delta
        ys.append(st @ q1)
        st = st.astype(mx.bfloat16).astype(mx.float32)
    y = mx.concatenate([mx.reshape(s, (1, 1, HKV, DV)) for s in ys], axis=1)
    mark("r12 eval serial small")
    mx.eval(y)
    mark("r12 done")


if __name__ == "__main__":
    rung = sys.argv[1] if len(sys.argv) > 1 else "0"
    fn = {"0": rung0, "1": rung1, "2": rung2, "3": rung3,
          "4": rung4, "5": rung5, "6": rung6, "7": rung7,
          "8": rung8, "9": rung9, "10": rung10,
          "11": rung11, "12": rung12}[rung]
    mark("rung %s enter" % rung)
    fn()
    mark("rung %s COMPLETE" % rung)
