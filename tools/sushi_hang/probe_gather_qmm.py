"""gather_qmm / qqmm probes at qwen4-ish shapes (the sushi EXL3/MoE path).
The python surface exposes gather_qmm; the sushi port reaches it via mlx-c.
A hang or fault here would never show in affine-quantized model tests."""
import sys
import time

import mlx.core as mx

T0 = time.time()


def mark(m):
    print("[%6.2f] %s" % (time.time() - T0, m), flush=True)


D, E, H, T = 1024, 8, 4096, 4304

mark("build quantized expert tables")
w = mx.random.normal((E, D, H)).astype(mx.bfloat16) * 0.02
q = mx.quantize(w, group_size=64, bits=4)  # [w, scales, biases]
wq, ws, wb = q
mark("wq %s ws %s" % (wq.shape, ws.shape))

x = mx.random.normal((1, T, H)).astype(mx.bfloat16) * 0.01  # [1,T,H]

mark("p1 plain quantized_matmul [T,H]x[D,H] 4bit g64")
y = mx.quantized_matmul(mx.reshape(x, (T, H)), wq[0], ws[0], wb[0],
                        group_size=64, bits=4)
mx.eval(y)
mark("p1 done %s" % (y.shape,))

mark("p2 gather_qmm with rhs_indices (expert gather + qmm), 4304 rows")
rhs = mx.random.randint(0, E, (1, T))  # per-row expert id
try:
    y2 = mx.gather_qmm(x, wq, ws, wb, rhs_indices=rhs,
                       group_size=64, bits=4)
    mx.eval(y2)
    mark("p2 done %s" % (y2.shape,))
except Exception as e:
    mark("p2 ERR %s %s" % (type(e).__name__, str(e)[:150]))

mark("p3 gather_qmm transposed rhs [D,T] out (expert slices)")
rhs2 = mx.zeros((1, T), mx.int32)
try:
    y3 = mx.gather_qmm(mx.reshape(x, (T, H)), wq, ws, wb, rhs_indices=rhs2,
                       group_size=64, bits=4, transpose_rhs=True)
    mx.eval(y3)
    mark("p3 done %s" % (y3.shape,))
except Exception as e:
    mark("p3 ERR %s %s" % (type(e).__name__, str(e)[:150]))

mark("p4 gather_qmm sorted_indices + topk-style lhs gather")
lhs = mx.random.randint(0, T, (1, 1280))
try:
    y4 = mx.gather_qmm(x, wq, ws, wb, rhs_indices=rhs,
                       lhs_indices=lhs, group_size=64, bits=4,
                       sorted_indices=mx.zeros((1,), mx.bool_))
    mx.eval(y4)
    mark("p4 done %s" % (y4.shape,))
except Exception as e:
    mark("p4 ERR %s %s" % (type(e).__name__, str(e)[:150]))

mark("p5 gather_qmm DECODE width (1 row, the gather_qmm_sub shape)")
x1 = mx.random.normal((1, 1, H)).astype(mx.bfloat16)
try:
    y5 = mx.gather_qmm(x1, wq, ws, wb, rhs_indices=mx.zeros((1, 1), mx.int32),
                       group_size=64, bits=4)
    mx.eval(y5)
    mark("p5 done %s" % (y5.shape,))
except Exception as e:
    mark("p5 ERR %s %s" % (type(e).__name__, str(e)[:150]))

mark("ALL PROBES COMPLETE")
