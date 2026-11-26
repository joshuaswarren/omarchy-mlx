import mlx.core as mx
x = mx.random.normal((64, 100352)).astype(mx.float32)
t = mx.array(list(range(64)))
tests = {
 "topk2": lambda: mx.topk(x, 2, axis=-1),
 "logsumexp": lambda: mx.logsumexp(x, axis=-1),
 "argmax": lambda: mx.argmax(x, axis=-1),
 "max": lambda: mx.max(x, axis=-1),
 "take_along_axis": lambda: mx.take_along_axis(x, t[:, None], axis=-1),
 "where_mask_max": lambda: mx.max(mx.where(mx.arange(x.shape[1])[None, :] == mx.argmax(x, -1)[:, None], -mx.inf, x), -1),
}
for k, f in tests.items():
    try:
        mx.eval(f()); print(k, "ok")
    except Exception as e:
        print(k, "FAIL", str(e)[:90])
