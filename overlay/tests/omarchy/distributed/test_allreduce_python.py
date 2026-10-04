"""Run under two MLX_RANK processes with a two-rank MLX_HOSTFILE."""

import mlx.core as mx


def test_allreduce_dtype_ops():
    group = mx.distributed.init()
    rank = group.rank()
    assert group.size() == 2, f"expected two ranks, got {group.size()}"
    values = [1, 4, -3] if rank == 0 else [10, 2, -1]
    cases = (
        (mx.float32, [11, 6, -4], [10, 4, -1], [1, 2, -3]),
        (mx.float16, [11, 6, -4], [10, 4, -1], [1, 2, -3]),
        (mx.bfloat16, [11, 6, -4], [10, 4, -1], [1, 2, -3]),
        (mx.int32, [11, 6, -4], [10, 4, -1], [1, 2, -3]),
    )
    for dtype, sum_want, max_want, min_want in cases:
        x = mx.array(values, dtype=dtype)
        for name, operation, want in (
            ("sum", mx.distributed.all_sum, sum_want),
            ("max", mx.distributed.all_max, max_want),
            ("min", mx.distributed.all_min, min_want),
        ):
            y = operation(x)
            mx.eval(y)
            got = y.tolist()
            assert got == want, (
                f"rank={rank} dtype={dtype} op={name}: expected {want}, got {got}"
            )
    print(f"DISTRIBUTED_ALLREDUCE_OK rank={rank} size=2 dtypes=4 ops=3 elements=3")


if __name__ == "__main__":
    test_allreduce_dtype_ops()
