
import json, math, time
res = {"available": False, "device": None, "mlx_version": None,
       "import_error": None, "ops": []}
try:
    import mlx.core as mx
except Exception as exc:
    res["import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(res))
    raise SystemExit(0)
res["available"] = True
res["device"] = str(mx.default_device())
res["mlx_version"] = getattr(mx, "__version__", None)

import sys as _sys
_sys.path.insert(0, ".")
from mlx_provenance import prepare_probe

try:
    res["provenance"] = prepare_probe(mx)
    res["device"] = str(mx.default_device())
except Exception as exc:
    res["available"] = False
    res["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(res))
    raise SystemExit(0)
if res["provenance"].get("verified") == "mismatch":
    res["available"] = False
    res["error"] = ("refusing to emit correctness numbers: "
                    + res["provenance"]["mismatch"])
    print(json.dumps(res))
    raise SystemExit(0)

def close(a, b, tol=1e-5):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, list) or isinstance(b, list) or not math.isfinite(a):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def json_value(value):
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value

def record(name, fn, expect):
    entry = {"op": name, "pass": False, "expected": expect, "actual": None,
             "detail": None, "ms": None}
    start = time.perf_counter()
    try:
        got = fn()
        got = got.tolist() if hasattr(got, "tolist") else got
        entry["pass"] = bool(close(got, expect))
        if not entry["pass"]:
            entry["actual"] = json_value(got)
            entry["detail"] = f"expected {expect}, got {got}"
    except Exception as exc:
        entry["detail"] = f"{type(exc).__name__}: {exc}"
    entry["ms"] = round((time.perf_counter() - start) * 1000, 3)
    res["ops"].append(entry)

record("add", lambda: mx.array([1.0, 2.0]) + mx.array([3.0, 4.0]), [4.0, 6.0])
record("matmul", lambda: mx.array([[1.0, 2.0], [3.0, 4.0]])
       @ mx.array([[5.0, 6.0], [7.0, 8.0]]),
       [[19.0, 22.0], [43.0, 50.0]])
record("softmax", lambda: mx.softmax(mx.array([1.0, 2.0, 3.0])),
       [0.09003057, 0.24472847, 0.66524096])
record("argmax", lambda: mx.argmax(mx.array([-1.0, 5.0, 3.0])), 1)
record("cumsum", lambda: mx.cumsum(mx.array([1.0, 2.0, 3.0])),
       [1.0, 3.0, 6.0])

xv, wv = [0.5, -0.25], [[1.0, 2.0], [3.0, 4.0]]
xw = [xv[0] * wv[0][j] + xv[1] * wv[1][j] for j in range(2)]
grad_expect = [[2.0 * xw[j] * xv[i] for j in range(2)] for i in range(2)]

def _grad():
    x = mx.array(xv)
    def f(w):
        return mx.sum((x @ w) ** 2)
    return mx.value_and_grad(f)(mx.array(wv))[1]

record("value_and_grad", _grad, grad_expect)
print(json.dumps(res, allow_nan=False))
