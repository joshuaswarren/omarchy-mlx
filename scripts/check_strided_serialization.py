# Serialization roundtrips for retained strided MLX views; constant
# writes require C-order (patches/mlx-export-dense-constants.patch).
# Every serializer consumes the same actual MLX views (Slice, Transpose,
# AsStrided-born) and expectations are NumPy arithmetic on the host,
# never a backend echo.
import os

os.environ.setdefault("MLX_OMARCHY_ALLOW_NON_APPLE", "1")

import sys
import tempfile

import mlx.core as mx
import numpy as np

SRC_NP = np.arange(48, dtype=np.float32).reshape(6, 8)


def views_np():
    return {
        "gapped": SRC_NP[:, 2:6],
        "transposed": SRC_NP.T,
        "strided": SRC_NP[1::3, ::2],
    }


def views_mx(src):
    return {
        "gapped": src[:, 2:6],
        "transposed": src.transpose(),
        "strided": src[1::3, ::2],
    }


def probe_np(view):
    return np.arange(view.size, dtype=np.float32).reshape(view.shape)


def expect_equal(name, got, want):
    got_np = np.asarray(got)
    if got_np.shape != want.shape or got_np.dtype != want.dtype:
        raise AssertionError(
            f"{name}: got {got_np.shape}/{got_np.dtype}, "
            f"want {want.shape}/{want.dtype}"
        )
    if not np.array_equal(got_np, want):
        raise AssertionError(f"{name}: values differ from NumPy expectation")


def check_safetensors(root, src):
    arrays = views_mx(src)
    arrays["dense"] = mx.array(np.ascontiguousarray(SRC_NP[:, 2:6]))
    path = os.path.join(root, "st.safetensors")
    mx.save_safetensors(path, arrays)
    loaded = mx.load(path)
    for name, v in views_np().items():
        expect_equal(f"safetensors {name}", loaded[name], v)
    expect_equal("safetensors dense", loaded["dense"], SRC_NP[:, 2:6])
    print("PASS safetensors roundtrip (gapped, transposed, strided, dense)")


def check_npy(root, src):
    for name, view in views_mx(src).items():
        path = os.path.join(root, f"{name}.npy")
        mx.save(path, view)
        expect_equal(f"npy {name}", mx.load(path), views_np()[name])
    print("PASS npy roundtrip (gapped, transposed, strided)")


def check_export(root, src):
    views = views_mx(src)
    mx.eval(*views.values())
    for name, view in views.items():
        v_np = views_np()[name]
        want = probe_np(view) + v_np

        def fn(x, _v=view):
            return mx.add(x, _v)

        path = os.path.join(root, f"export_{name}.mlxfn")
        mx.export_function(path, fn, mx.array(probe_np(view)))
        out = mx.import_function(path)(mx.array(probe_np(view)))
        if isinstance(out, (list, tuple)):
            out = out[0]
        expect_equal(f"export {name}", out, want)
    print("PASS export roundtrip (closed-over gapped, transposed, strided)")


def main():
    src = mx.array(SRC_NP)
    with tempfile.TemporaryDirectory() as root:
        check_safetensors(root, src)
        check_npy(root, src)
        check_export(root, src)
    print("ALL STRIDED SERIALIZATION CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
