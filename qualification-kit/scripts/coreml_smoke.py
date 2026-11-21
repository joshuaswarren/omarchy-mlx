
import json, time
out = {"available": False, "calls": 20, "min_ms": None, "median_ms": None,
       "error": None}
try:
    import coremltools as ct
    import numpy as np
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder
    builder = NeuralNetworkBuilder(
        [("x", datatypes.Array(2)), ("y", datatypes.Array(2))],
        [("z", None)])
    builder.add_elementwise(name="add", input_names=["x", "y"],
                            output_name="z", mode="ADD")
    model = ct.models.MLModel(builder.spec, compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    times = []
    for _ in range(out["calls"]):
        start = time.perf_counter()
        got = model.predict({"x": x, "y": y})["z"]
        times.append((time.perf_counter() - start) * 1000)
        # The NeuralNetwork runtime returns the rank-5 (S,B,C,H,W) array,
        # shape (1, 1, 2, 1, 1); compared unflattened it would broadcast
        # against (2,) and fail on correct values.
        if not np.allclose(np.asarray(got).reshape(-1), x + y, atol=1e-4):
            out["error"] = "wrong result"
            break
    else:
        out["available"] = True
        out["min_ms"] = round(min(times), 3)
        out["median_ms"] = round(sorted(times)[len(times) // 2], 3)
except Exception as exc:
    out["error"] = ("%s: %s" % (type(exc).__name__, exc))[:200]
print(json.dumps(out))
