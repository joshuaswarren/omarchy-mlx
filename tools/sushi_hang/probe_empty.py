"""Zero/empty-workload dispatch probes: the qwen4 text forward splices vision
rows with masked_scatter over an EMPTY mask; PLE/ngram gather arms can see 0
rows. A 0-group dispatch that a driver mishandles would hang the ring."""
import mlx.core as mx

def probe(name, fn):
    try:
        fn()
        print("PASS", name, flush=True)
    except Exception as e:
        print("ERR ", name, type(e).__name__, str(e)[:120], flush=True)

x = mx.random.normal((4, 8)).astype(mx.bfloat16)

probe("masked_scatter empty mask", lambda: mx.eval(
    mx.masked_scatter(x, mx.zeros((4, 8), mx.bool_), mx.zeros((0,), mx.bfloat16))))
probe("gather 0 rows", lambda: mx.eval(mx.take(x, mx.zeros((0,), mx.int32), axis=0)))
probe("take 0 idx", lambda: mx.eval(x[mx.zeros((0,), mx.int32)]))
probe("softmax 0 rows", lambda: mx.eval(mx.softmax(mx.zeros((0, 8), mx.float32), axis=-1)))
probe("matmul 0 rows", lambda: mx.eval(
    mx.zeros((0, 8), mx.bfloat16) @ mx.random.normal((8, 8)).astype(mx.bfloat16)))
probe("argpartition 0", lambda: mx.eval(
    mx.argpartition(mx.zeros((0, 8), mx.float32), kth=-2, axis=-1)))
probe("conv 0 len", lambda: mx.eval(
    mx.conv1d(mx.zeros((1, 0, 8), mx.bfloat16),
              mx.random.normal((3, 8, 8)).astype(mx.bfloat16), stride=1)))
probe("mean over 0", lambda: mx.eval(mx.zeros((4, 0), mx.bfloat16).mean(axis=1)))
probe("cumsum 0", lambda: mx.eval(mx.cumsum(mx.zeros((0,), mx.float32))))
probe("bool take true all", lambda: mx.eval(x[mx.ones((4,), mx.bool_)]))
probe("tile hc", lambda: mx.eval(mx.tile(x, (1, 1, 4))))
probe("empty slice add", lambda: mx.eval(x[:0] + x[:0]))
print("ALL DONE", flush=True)
