#!/usr/bin/env python3
# aten.linalg_vector_norm / aten.linalg_norm lowering + fp32/bf16 parity
# across ord=1/2/inf (SUP: PT-NORM-b).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "omx", "tools", "torch-mlx-vulkan"))
import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def _cpu(a):
    return torch.ops.aten.linalg_vector_norm(torch.tensor(a.cpu().numpy())
                                            if hasattr(a, "cpu") else torch.tensor(a),
                                            ord=CUR_ORD.get("v", None),
                                            dim=CUR_DIM.get("v", None),
                                            keepdim=CUR_KEEP.get("v", False))


CUR_ORD = {}
CUR_DIM = {}
CUR_KEEP = {}


def run(x_np, ord_, dim, keepdim, dtype):
    CUR_ORD["v"] = ord_
    CUR_DIM["v"] = dim
    CUR_KEEP["v"] = keepdim
    xt = torch.tensor(x_np.astype(np.float32 if dtype == mx.float32 else np.float16)
                      if dtype == mx.float32 else x_np, dtype=dtype_torch(dtype))
    got = mlx_vulkan._FUNC_SPECS["linalg_vector_norm"](
        mlx_vulkan._mx(xt), ord=ord_, dim=dim, keepdim=keepdim)
    want = torch.ops.aten.linalg_vector_norm(
        xt, ord=ord_ if ord_ is not None else 2,
        dim=dim, keepdim=keepdim)
    return _np(got), want.numpy()


def dtype_torch(dt):
    import mlx.core
    return torch.float32 if dt == mlx.core.float32 else torch.bfloat16


def main():
    failures = []
    rng = np.random.RandomState(7)
    x = rng.randn(3, 5, 4).astype(np.float32) * 3

    for dtype, tag in ((mx.float32, "fp32"), (mx.bfloat16, "bf16")):
        for ord_ in (1, 2, float("inf")):
            for dim, want_shape in ((None, ()), (0, (5, 4)), (1, (3, 4)),
                                    (2, (3, 5)), (-1, (3, 5)),
                                    ((0, 2), (5,)), (1, (3, 4))):
                for keepdim in (False, True):
                    got, want = run(x, ord_, dim, keepdim, dtype)
                    g = np.array(got, dtype=np.float32)
                    w = np.asarray(want, dtype=np.float32)
                    if g.shape != w.shape:
                        failures.append((tag, ord_, dim, keepdim, "shape",
                                         g.shape, w.shape))
                        continue
                    if not np.allclose(g, w, rtol=1e-3, atol=1e-3):
                        failures.append((tag, ord_, dim, keepdim, "value",
                                         g.ravel()[:3], w.ravel()[:3]))

    # aten.linalg_norm alias: defaults ord=2, dim=None, keepdim=False
    a = rng.randn(7).astype(np.float32)
    got = mlx_vulkan._FUNC_SPECS["linalg_norm"](mlx_vulkan._mx(a))
    want = np.linalg.norm(a)
    if not np.allclose(_np(got), want, rtol=1e-4):
        failures.append(("alias", want, _np(got)))

    if failures:
        print("FAILURES:", len(failures))
        for f in failures[:10]:
            print(" ", f)
        sys.exit(1)
    print("test_linalg_vector_norm PASS: fp32+bf16 x ord 1/2/inf x dims x "
          "keepdim + alias")


if __name__ == "__main__":
    main()
