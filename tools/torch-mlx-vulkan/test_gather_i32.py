"""Q-72 audit pin: index arrays reaching the mlx index kernels are int32.

mx.argsort and mx.argmax emit uint32, and arithmetic on them promotes to
int64. A raw non-i32 index that reaches take/take_along_axis/put_along_axis
is silently wrong or crashes on backends whose index kernels assume i32.
Every gather site in mlx_vulkan must cast (or provably produce int32 - e.g.
mx.sum over a bool comparison). The spy wraps the index-taking entry points
and records the index dtype at each lowered call; the helpers below
exercise every site the audit flagged.
"""
import mlx.core as mx
import mlx_vulkan

RECORD = []


def _spy_all():
    orig = {}
    for name in ("take", "take_along_axis", "put_along_axis"):
        orig[name] = getattr(mx, name)

        def spy(a, i, *args, _n=name, **kw):
            RECORD.append((_n, i.dtype))
            return orig[_n](a, i, *args, **kw)
        setattr(mx, name, spy)
    return orig


def _restore(orig):
    for name, fn in orig.items():
        setattr(mx, name, fn)


def test_gather_index_dtypes():
    orig = _spy_all()
    try:
        # _topk: argsort -> take_along_axis
        mlx_vulkan._topk(mx.array([[3., 1., 2.], [0., 9., 4.]]), 2, 1)
        # _mode: argsort -> take_along_axis; argmax-fed pos (u32+i32 -> i64)
        # -> take_along_axis
        mlx_vulkan._mode(mx.array([[1., 2., 2.], [4., 4., 4.]]), 1)
        mlx_vulkan._mode(mx.array([[1, 2, 2], [4, 4, 4]]), 1)  # integral leg
        # _kthvalue: argsort -> take_along_axis
        mlx_vulkan._kthvalue(mx.array([[5., 1., 3.]]), 2, 1)
        mlx_vulkan._kthvalue(mx.array([[5, 1, 3]]), 2, 1)
        # _gather_logp / the take and gather lambdas in _FUNC_SPECS
        mlx_vulkan._gather_logp(mx.array([[0.1, 0.9]]), mx.array([1]))
        mlx_vulkan._FUNC_SPECS["take"](mx.array([1, 2, 3]),
                                       mx.array([0, 2]), 0)
        mlx_vulkan._FUNC_SPECS["gather"](mx.array([[1., 2.], [3., 4.]]),
                                        1, mx.array([[0], [1]]))
        # _multinomial indexes internally with argsort output
        mlx_vulkan._multinomial(mx.array([0.1, 0.7, 0.2]), 2)
    finally:
        _restore(orig)

    bad = [r for r in RECORD if r[1] != mx.int32]
    assert not bad, f"non-i32 index reached an index kernel: {bad}"
    assert RECORD, "spy saw no gather calls - test is vacuous"
    print("test_gather_index_dtypes PASS")


def test_pdist_arg_order():
    # _pdist gathers with an argsort-built index via getitem (not spied);
    # the parity check keeps the cast from reordering the result.
    import numpy as np
    a = mx.array([[0., 0.], [1., 0.], [0., 2.]])
    got = np.asarray(mlx_vulkan._pdist(a, 2.0))
    want = np.array([1., 2., np.sqrt(5.)], dtype=np.float32)
    assert np.allclose(got, want), f"pdist values {got} != {want}"
    print("test_pdist_arg_order PASS")


if __name__ == "__main__":
    test_gather_index_dtypes()
    test_pdist_arg_order()
