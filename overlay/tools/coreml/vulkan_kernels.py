"""Metal kernel factories for the vulkan_encoder, split out of it
(issue #17): a leaf concern with no back-dependencies beyond mlx.
All names are re-exported by vulkan_encoder for its handlers."""

from functools import cache

import mlx.core as mx


@cache
def _leftover_chain_kernel():
    """Reduces the batched matmul's fp32 block partials with the landed
    leftover-linear rounding: every 16-wide block's partial rounds to fp16,
    then accumulates in fp16 ascending. One thread per output element, so
    the chain inside the kernel is the same strictly sequential fp16 sum
    the 5688f8bd chunk loop performed dispatch by dispatch."""
    return mx.fast.metal_kernel(
        name="encoder_leftover_fp16_chain_f32",
        input_names=["partials"],
        output_names=["reduced"],
        source="""
            uint n = partials_shape[2];
            uint mn = partials_shape[1] * n;
            uint blocks = partials_shape[0];
            uint index = thread_position_in_grid.x;
            half acc = half(partials[index]);
            for (uint block = 1u; block < blocks; ++block) {
                acc = acc + half(partials[index + block * mn]);
            }
            reduced[index] = acc;
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _leftover_chain_bias_kernel():
    """The leftover chain with the linear's bias folded into the final
    store. The chain half is byte-identical to _leftover_chain_kernel:
    each block's partial rounds to fp16 (the coopmat path's fp16 partials
    pass through exactly; the f32 fallback path's fp32 partials round
    here, the same single RNE the stock chain applies on read) and
    accumulates in fp16 ascending. The epilogue reproduces the removed
    elementwise dispatch's bytes: the stored fp16 chain output widened,
    added to the fp32 bias, rounded once. One thread per output PAIR; the
    bias index is derived from the in-row pair, never from the flattened
    word (an earlier variant indexed bias with the flattened word and
    read out of bounds past the bias buffer for every row past the
    first)."""
    return mx.fast.metal_kernel(
        name="encoder_leftover_fp16_chain_bias_f32",
        input_names=["partials", "bias"],
        output_names=["reduced"],
        source="""
            uint pairs = partials_shape[2] / 2u;
            uint words_row = partials_shape[1] * pairs;
            uint blocks = partials_shape[0];
            uint index = thread_position_in_grid.x;
            uint row = index / pairs;
            uint pair = index - row * pairs;
            uint word = row * pairs + pair;
            uint col = word * 2u;
            half a0 = half(partials[col]);
            half a1 = half(partials[col + 1u]);
            for (uint block = 1u; block < blocks; ++block) {
                uint base = col + block * words_row * 2u;
                a0 = a0 + half(partials[base]);
                a1 = a1 + half(partials[base + 1u]);
            }
            float v0 = float(a0) + float(bias[pair * 2u]);
            float v1 = float(a1) + float(bias[pair * 2u + 1u]);
            reduced[col] = half(v0);
            reduced[col + 1u] = half(v1);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _leftover_chain_bias_silu_kernel():
    """_leftover_chain_bias_kernel with the feed-forward silu folded in:
    the bias sum rounds to fp16 first (the stored f16 linear output the
    separate silu dispatch read), then the silu math runs in f32 and the
    result rounds once to fp16 -- the same roundings as the removed
    _silu_kernel dispatch. The rounded bias sum goes through the output
    buffer before the silu math: in-register, this compiler elides the
    half() round when the value only feeds float() reads (lincheck
    mismatched 14% of elements; the memory round-trip pins the rounding),
    while the chain half above rounds correctly in both kernels. Each
    thread owns its two columns, so the write-back has no cross-thread
    hazard."""
    return mx.fast.metal_kernel(
        name="encoder_leftover_fp16_chain_bias_silu_f32",
        input_names=["partials", "bias"],
        output_names=["reduced"],
        source="""
            uint pairs = partials_shape[2] / 2u;
            uint words_row = partials_shape[1] * pairs;
            uint blocks = partials_shape[0];
            uint index = thread_position_in_grid.x;
            uint row = index / pairs;
            uint pair = index - row * pairs;
            uint word = row * pairs + pair;
            uint col = word * 2u;
            half a0 = half(partials[col]);
            half a1 = half(partials[col + 1u]);
            for (uint block = 1u; block < blocks; ++block) {
                uint base = col + block * words_row * 2u;
                a0 = a0 + half(partials[base]);
                a1 = a1 + half(partials[base + 1u]);
            }
            reduced[col] = half(float(a0) + float(bias[pair * 2u]));
            reduced[col + 1u] = half(float(a1) + float(bias[pair * 2u + 1u]));
            half w0 = reduced[col];
            half w1 = reduced[col + 1u];
            float s0 = float(w0) * (1.0 / (1.0 + exp(-float(w0))));
            float s1 = float(w1) * (1.0 / (1.0 + exp(-float(w1))));
            reduced[col] = half(s0);
            reduced[col + 1u] = half(s1);
        """,
        compile_options={"math_mode": "safe"},
    )

@cache
def _linear_f16_coopmat_kernel():
    """Batched-blocked fp16 linear on the f32 8x8x8 matrix unit: one
    custom dispatch computes every 16-wide block's fp32 partial (A/B read
    as fp16, widened exactly when staged to shared, identical staging
    layout, k-step order, and coopMatMulAdd sequence as the f32 batched
    coopmat matmul it replaces) and rounds each partial once to fp16 at
    the drain — the same single round-to-nearest-even the fp32 partials
    path applies when the chain kernel reads them. fp16 partials halve
    the dominant partials write+read traffic; bytes are identical by
    construction."""
    return mx.fast.metal_kernel(
        name="encoder_linear_f16_coopmat",
        input_names=["lhs", "rhs"],
        output_names=["dst"],
        header="""
            #extension GL_KHR_cooperative_matrix : require
            #extension GL_KHR_memory_scope_semantics : require
        """,
        source="""
            uint rows = lhs_shape[0];
            uint K = lhs_shape[1];
            uint N = rhs_shape[0];
            uint lane = gl_LocalInvocationID.x;
            uint col_base = gl_WorkGroupID.x * 32u;
            uint row_base = gl_WorkGroupID.y * 32u;
            uint k_base = gl_WorkGroupID.z * 16u;
            uint mn = rows * N;
            uint out_base = gl_WorkGroupID.z * mn + row_base * N + col_base;
            threadgroup float a_s[256];
            threadgroup float b_s[256];
            coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseAccumulator> acc[4][4];
            for (uint rb = 0u; rb < 4u; ++rb) {
              for (uint cb = 0u; cb < 4u; ++cb) {
                acc[rb][cb] = coopmat<float, gl_ScopeSubgroup, 8, 8,
                    gl_MatrixUseAccumulator>(0.0);
              }
            }
            for (uint step = 0u; step < 2u; ++step) {
              uint k8 = k_base + step * 8u;
              for (uint j = 0u; j < 8u; ++j) {
                uint i = lane + 32u * j;
                uint row_local = i / 8u;
                uint row = row_base + row_local;
                uint k = k8 + (i % 8u);
                float v = 0.0;
                if (row < rows) {
                  v = float(lhs[row * K + k]);
                }
                a_s[row_local * 8u + (i % 8u)] = v;
              }
              for (uint j = 0u; j < 8u; ++j) {
                uint i = lane + 32u * j;
                uint col_local = i / 8u;
                uint col = col_base + col_local;
                uint k = k8 + (i % 8u);
                float v = 0.0;
                if (col < N) {
                  v = float(rhs[col * K + k]);
                }
                b_s[(i % 8u) * 32u + col_local] = v;
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseA> mat_a[4];
              coopmat<float, gl_ScopeSubgroup, 8, 8, gl_MatrixUseB> mat_b[4];
              for (uint rb = 0u; rb < 4u; ++rb) {
                coopMatLoad(mat_a[rb], a_s, rb * 64u, 8u,
                    gl_CooperativeMatrixLayoutRowMajor);
              }
              for (uint cb = 0u; cb < 4u; ++cb) {
                coopMatLoad(mat_b[cb], b_s, cb * 8u, 32u,
                    gl_CooperativeMatrixLayoutRowMajor);
              }
              for (uint rb = 0u; rb < 4u; ++rb) {
                for (uint cb = 0u; cb < 4u; ++cb) {
                  acc[rb][cb] = coopMatMulAdd(mat_a[rb], mat_b[cb], acc[rb][cb]);
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint rb = 0u; rb < 4u; ++rb) {
              for (uint cb = 0u; cb < 4u; ++cb) {
                coopMatStore(acc[rb][cb], a_s, cb * 64u, 8u,
                    gl_CooperativeMatrixLayoutRowMajor);
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              uint column = col_base + lane;
              if (column < N) {
                for (uint r = 0u; r < 8u; ++r) {
                  uint row = row_base + rb * 8u + r;
                  if (row < rows) {
                    dst[out_base + (rb * 8u + r) * N + lane] =
                        a_s[(lane / 8u) * 64u + r * 8u + (lane % 8u)];
                  }
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _silu_kernel():
    """silu in one dispatch. The fp32 math and the single fp16 rounding at
    the end replicate the two-dispatch chain exactly: mx.sigmoid lowers to
    `1.0 / (1.0 + exp(-x))` (elementwise.comp case 5) and the product stays
    fp32 until the op boundary cast. Buffer names never use `x` or `y`: the
    translator macros every name (`#define src _b0.data`) and would eat the
    `.x` swizzle of thread_position_in_grid."""
    return mx.fast.metal_kernel(
        name="encoder_silu_fused",
        input_names=["src"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            float v = float(src[index]);
            dst[index] = half(v * (1.0 / (1.0 + exp(-v))));
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _glu_kernel():
    """Conv-module GLU in one dispatch: `a * sigmoid(b)`.
    The gate rounds fp32->fp16 before the product, because the stored
    sigmoid result the mul consumes was itself an fp16 tensor. The rounding
    must go through packHalf2x16: an fp16_t local keeps fp32 precision in
    the following arithmetic (the fused_chain.comp M1-equality lesson)."""
    return mx.fast.metal_kernel(
        name="encoder_glu_fused",
        input_names=["lhs", "rhs"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            float s = 1.0 / (1.0 + exp(-float(rhs[index])));
            float gate = float(unpackHalf2x16(packHalf2x16(vec2(s, 0.0))).x);
            dst[index] = half(float(lhs[index]) * gate);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _ln_cast_kernel():
    return mx.fast.metal_kernel(
        name="encoder_ln_cast",
        input_names=["src"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            dst[index] = float(src[index]);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _ln_sq_kernel():
    """(xf - mean)^2 with the row mean broadcast per 1024-wide row. Same
    sub and square elementwise ops the separate dispatches ran, same fp32
    storage, so the following mx.mean reduce sees identical bits."""
    return mx.fast.metal_kernel(
        name="encoder_ln_centered_square",
        input_names=["xf", "mu"],
        output_names=["t2"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 1024u;
            float t = xf[index] - mu[row];
            t2[index] = t * t;
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _ln_tail_kernel():
    """LayerNorm tail in one dispatch. mx.rsqrt lowers to inversesqrt
    (elementwise.comp case 8); the mul-mul-add order and the single fp16
    rounding at the end match the dispatch chain statement for statement."""
    return mx.fast.metal_kernel(
        name="encoder_ln_tail",
        input_names=["xf", "mu", "va", "ga", "be", "ep"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 1024u;
            uint col = index % 1024u;
            float t = xf[index] - mu[row];
            float rstd = inversesqrt(va[row] + ep[0]);
            precise float pr = t * rstd * float(ga[col]);
            pr = pr + float(be[col]);
            dst[index] = half(pr);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _sm_exp_kernel():
    """shifted exp in one dispatch. The row max runs as ReduceF16 over the
    fp16 input: max is order-insensitive and exact, so widening its fp16
    result is the same fp32 value the cast-then-ReduceF32 arm produced."""
    return mx.fast.metal_kernel(
        name="encoder_softmax_exp",
        input_names=["src", "rmax"],
        output_names=["expd"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 375u;
            expd[index] = exp(float(src[index]) - float(rmax[row]));
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _sm_div_kernel():
    return mx.fast.metal_kernel(
        name="encoder_softmax_div",
        input_names=["expd", "rsum"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            uint row = index / 375u;
            dst[index] = half(expd[index] / rsum[row]);
        """,
        compile_options={"math_mode": "safe"},
    )




@cache
def _pw_kernel(op: str, layout: str, n: int, width: int = 0, time: int = 0):
    """fp16 pointwise `a <op> b` in one dispatch. The chain this replaces
    is cast(fp16->f32) x2, one ElementwiseF32, cast(f32->fp16); the kernel
    widens the identical fp16 operands to fp32, runs the identical fp32
    op, and applies the same single round-to-nearest-even at the boundary,
    so the result is bit-identical. `layout` selects how the rhs is
    addressed: flat for same-shape pairs, bare (the macro of a 0-d
    reference parameter already carries the [0]) for a broadcast scalar,
    or the row-broadcast `(index / width) % time` of the conv-stem mask
    muls, whose per-instance width/time bake into the source. The grid
    convention is thread counts per axis, so oversized tensors spread the
    flat element index across y around the 65535-workgroup x limit."""
    rhs = {
        "flat": "float(rhs[index])",
        "scalar": "float(rhs)",
        "row": f"float(rhs[(index / {width}u) % {time}u])",
    }[layout]
    symbol = {"add": "+", "sub": "-", "mul": "*"}[op]
    stride = min(n, 65535 * 256)
    source = f"""
            uint index = thread_position_in_grid.x
                       + thread_position_in_grid.y * {stride}u;
            if (index >= {n}u) {{
                return;
            }}
            dst[index] = half(float(lhs[index]) {symbol} {rhs});
        """
    return mx.fast.metal_kernel(
        name=f"encoder_pw_{op}_{layout}",
        input_names=["lhs", "rhs"],
        output_names=["dst"],
        source=source,
        compile_options={"math_mode": "safe"},
    )


def _pw(op: str, x: mx.array, y: mx.array) -> mx.array | None:
    """Run the fused pointwise kernel for `op` in (add, sub, mul), or
    return None when the operand pair does not match one of the three
    exact broadcast layouts the pinned program uses; the caller then
    falls back to the dispatch chain the kernel replaces."""
    if x.dtype != mx.float16 or y.dtype != mx.float16:
        return None
    if y.size == 1:
        layout = "scalar"
        y = mx.reshape(y, ())
    elif x.shape == y.shape:
        layout = "flat"
    elif (
        y.ndim == 4 and y.shape[0] == 1 and y.shape[1] == 1 and y.shape[3] == 1
        and x.ndim == 4 and x.shape[0] == 1 and x.shape[2] == y.shape[2]
    ):
        layout = "row"
    else:
        return None
    n = x.size
    threads_x = min(n, 65535 * 256)
    threads_y = (n + threads_x - 1) // threads_x
    out = _pw_kernel(op, layout, n,
                     x.shape[3] if layout == "row" else 0,
                     y.shape[2] if layout == "row" else 0)(
        inputs=[x, y],
        output_shapes=[(n,)],
        output_dtypes=[mx.float16],
        grid=(threads_x, threads_y, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )[0]
    return mx.reshape(out, x.shape)


@cache
def _sigmoid_kernel():
    """sigmoid in one dispatch, the silu kernel without its multiply:
    mx.sigmoid lowers to `1.0 / (1.0 + exp(-x))` in fp32 with one fp16
    rounding at the op boundary, exactly as the dispatch chain ran it."""
    return mx.fast.metal_kernel(
        name="encoder_sigmoid_fused",
        input_names=["src"],
        output_names=["dst"],
        source="""
            uint index = thread_position_in_grid.x;
            float v = float(src[index]);
            dst[index] = half(1.0 / (1.0 + exp(-v)));
        """,
        compile_options={"math_mode": "safe"},
    )


