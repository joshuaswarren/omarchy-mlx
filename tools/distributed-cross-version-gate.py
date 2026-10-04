#!/usr/bin/env python3
"""Two-rank MLX gate for cross-version collectives and point-to-point transfer.

Run once per rank with matching --sizes-mib. Requires MLX_RANK and MLX_HOSTFILE.
"""

import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1 or any(n < 1 or n > 1024 for n in args.sizes_mib):
        parser.error("iterations must be positive and sizes must be 1..1024 MiB")

    import mlx.core as mx

    device = mx.default_device()
    if device != mx.gpu:
        raise RuntimeError(f"distributed tensor gate requires mx.gpu, got {device}")
    group = mx.distributed.init()
    rank = group.rank()
    if group.size() != 2 or rank not in (0, 1):
        raise RuntimeError(f"expected two ranks, got rank={rank} size={group.size()}")
    print(f"DIST_GATE_START rank={rank} size={group.size()} device={device} mlx={mx.__version__}", flush=True)

    for mib in args.sizes_mib:
        n = mib * 1024 * 1024 // 4
        x = mx.full((n,), rank + 1, dtype=mx.float32)
        mx.eval(x)
        for name, operation, expected in (
            ("all_sum", lambda: mx.distributed.all_sum(x), 3),
            ("all_gather", lambda: mx.distributed.all_gather(x), 1),
            ("sum_scatter", lambda: mx.distributed.sum_scatter(x), 3),
        ):
            warmup = operation()
            mx.eval(warmup)
            del warmup
            samples = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                y = operation()
                mx.eval(y)
                samples.append(time.perf_counter() - start)
            if name == "all_gather":
                valid = mx.all(y[:n] == 1) & mx.all(y[n:] == 2)
            else:
                valid = mx.all(y == expected)
            mx.eval(valid)
            if not valid.item():
                raise AssertionError(f"{name} {mib} MiB rank={rank}: expected every output element to be correct")
            elapsed = sorted(samples)[len(samples) // 2]
            wire_bytes = mib * 1024 * 1024
            print(
                f"DIST_GATE op={name} rank={rank} mib={mib} iterations={args.iterations} "
                f"median_ms={elapsed * 1000:.3f} wire_bytes={wire_bytes} "
                f"MiB_s={wire_bytes / elapsed / 1024 / 1024:.3f}",
                flush=True,
            )

        peer = 1 - rank

        def exchange():
            if rank == 0:
                sent = mx.distributed.send(x, peer)
                mx.eval(sent)
                y = mx.distributed.recv((n,), mx.float32, peer)
                mx.eval(y)
            else:
                y = mx.distributed.recv((n,), mx.float32, peer)
                mx.eval(y)
                sent = mx.distributed.send(x, peer)
                mx.eval(sent)
            valid = mx.all(y == peer + 1)
            mx.eval(valid)
            if not valid.item():
                raise AssertionError(f"send_recv {mib} MiB rank={rank}: bad peer payload")

        exchange()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            exchange()
            samples.append(time.perf_counter() - start)
        elapsed = sorted(samples)[len(samples) // 2]
        wire_bytes = 2 * mib * 1024 * 1024
        print(
            f"DIST_GATE op=send_recv_roundtrip rank={rank} mib={mib} "
            f"iterations={args.iterations} median_ms={elapsed * 1000:.3f} "
            f"wire_bytes={wire_bytes} MiB_s={wire_bytes / elapsed / 1024 / 1024:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
