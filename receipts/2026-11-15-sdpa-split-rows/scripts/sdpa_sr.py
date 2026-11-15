# Split-K verify rows (MLX_OMARCHY_SDPA_SPLIT_ROWS) against the one-token route and the default rows route.
# Part 1, SR_EQ: each of the T rows (causal, N keys in the cache) against the same token decoded alone on N-T+1+r keys.
# Part 2, SR_T: microseconds per call, SPLIT_ROWS=1 against 0, 36 chained calls, median of 24 chains.
import os
import statistics as st
import time

import mlx.core as mx

CHAIN, REPS = 36, 24
mx.random.seed(5)


def sdpa(q, k, v, scale, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def check(qh, kvh, d, n, t):
    scale = d ** -0.5
    k = (mx.random.normal((1, kvh, n, d)) * 0.5).astype(mx.bfloat16)
    v = (mx.random.normal((1, kvh, n, d)) * 0.5).astype(mx.bfloat16)
    q = (mx.random.normal((1, qh, t, d)) * 0.5).astype(mx.bfloat16)
    mx.eval(k, v, q)
    os.environ['MLX_OMARCHY_SDPA_SPLIT_ROWS'] = '1'
    rows = sdpa(q, k, v, scale, 'causal')
    mx.eval(rows)
    os.environ.pop('MLX_OMARCHY_SDPA_SPLIT_ROWS')
    single = mx.concatenate([sdpa(q[:, :, r:r + 1, :], k[:, :, :n - t + 1 + r, :], v[:, :, :n - t + 1 + r, :], scale, None) for r in range(t)], axis=2)
    os.environ['MLX_OMARCHY_SDPA_SPLIT_ROWS'] = '0'
    default = sdpa(q, k, v, scale, 'causal')
    os.environ.pop('MLX_OMARCHY_SDPA_SPLIT_ROWS')
    mx.eval(single, default)
    d_single = float(mx.abs(rows.astype(mx.float32) - single.astype(mx.float32)).max())
    d_default = float(mx.abs(rows.astype(mx.float32) - default.astype(mx.float32)).max())
    print('SR_EQ', 'hd', d, 'N', n, 'T', t, 'rows_vs_single_maxabs', d_single, 'bit_equal_single', d_single == 0.0, 'rows_vs_default_maxabs', d_default, flush=True)


for n in (512, 2048, 4096):
    for t in (2, 3, 4, 8, 16):
        check(32, 8, 128, n, t)
for t in (2, 4):
    check(16, 4, 256, 2048, t)


def timed(qh, kvh, d, n, t, flag):
    scale = d ** -0.5
    kb = (mx.random.normal((1, kvh, n, d)) * 0.5).astype(mx.bfloat16)
    vb = (mx.random.normal((1, kvh, n, d)) * 0.5).astype(mx.bfloat16)
    q0 = (mx.random.normal((1, qh, t, d)) * 0.5).astype(mx.bfloat16)
    mx.eval(kb, vb, q0)
    os.environ['MLX_OMARCHY_SDPA_SPLIT_ROWS'] = flag

    def chain():
        x = q0
        for _ in range(CHAIN):
            x = sdpa(x, kb, vb, scale, 'causal')
        mx.eval(x)

    chain()
    chain()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        chain()
        ts.append((time.perf_counter() - t0) * 1e6 / CHAIN)
    os.environ.pop('MLX_OMARCHY_SDPA_SPLIT_ROWS')
    return st.median(ts)


for n in (512, 2048, 4096):
    for t in (2, 3, 4, 8):
        a = timed(32, 8, 128, n, t, '1')
        b = timed(32, 8, 128, n, t, '0')
        print('SR_T', n, t, 'split_us', round(a, 1), 'default_us', round(b, 1), 'speedup', round(b / a, 2), flush=True)
print('SR_DONE', flush=True)
