# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""FUSE-5: the fused-stage greedy head pins the eight-stage head's output.

MLX_OMARCHY_GREEDY_FUSE=1 (read per call, default off) replaces the eight
sequential greedy dispatches with three merged stages (qmm_vec.comp stages
8-10: prologue+bounds+candidates, prune+exact+select, flagged full) plus a
state fill. The token and the stats vector are the pinned identity: fixed
duplicate-row ties and randomized inputs must give the same token and the
same stats flag on and off, and the token must equal the composed
quantized_matmul + logsumexp + subtract + argmax expression. Unlike the
two-stage sketch in 1c8a932f8d, the fused path preserves the prune
decision, so stats equality (not always-full markers) is the check.

Adapted from the red-first pin by w73 (commit 1c8a932f8d); the flag and
the stats contract differ from that draft on purpose.
"""

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

K = 2048
N = 512
GROUPS = K // 64


def _gpu_available():
    try:
        prev = mx.default_device()
        mx.set_default_device(mx.gpu)
        mx.eval(mx.array(np.zeros(1, dtype=np.float32)))
        mx.set_default_device(prev)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(), reason="no Honeykrisp GPU route on this host"
)


def _words(codes):
    """uint32 [N, K/8]: word w packs codes 8w..8w+8, low nibble first."""
    packed = codes.reshape(N, K // 8, 8).astype(np.uint32)
    words = np.zeros((N, K // 8), dtype=np.uint32)
    for nib in range(8):
        words |= packed[:, :, nib] << np.uint32(4 * nib)
    return words


def _plane(codes):
    """uint32 [N, K*3/32]: code e of group g sits at bit 3e of word 6g+3e//32."""
    hi = (codes >> 1).astype(np.uint32).reshape(N, GROUPS, 64)
    plane = np.zeros((N, GROUPS, 6), dtype=np.uint32)
    for e in range(64):
        word, shift = (3 * e) // 32, np.uint32((3 * e) % 32)
        plane[:, :, word] |= hi[:, :, e] << shift
    return plane.reshape(N, K * 3 // 32)


def _inputs(rng, dup_rows=()):
    codes = rng.integers(0, 16, size=(N, K)).astype(np.uint8)
    scales = (rng.uniform(-0.5, 0.5, size=(N, GROUPS)) + 0.6).astype(np.float32)
    biases = (rng.uniform(-0.25, 0.25, size=(N, GROUPS)) - 0.1).astype(np.float32)
    x = rng.uniform(-2.0, 2.0, size=K).astype(np.float32)
    for src, dst in dup_rows:
        codes[dst] = codes[src]
        scales[dst] = scales[src]
        biases[dst] = biases[src]
    return (
        mx.array(x).astype(mx.bfloat16),
        mx.array(_words(codes)),
        mx.array(scales).astype(mx.bfloat16),
        mx.array(biases).astype(mx.bfloat16),
        mx.array(_plane(codes)),
    )


def _greedy(x, w, scales, biases, plane, fused):
    old = os.environ.get("MLX_OMARCHY_GREEDY_FUSE")
    os.environ["MLX_OMARCHY_GREEDY_FUSE"] = "1" if fused else "0"
    try:
        token, stats = mx.fast.greedy_quantized_argmax(
            x, w, scales, biases, plane, group_size=64, bits=4
        )
        mx.eval(token, stats)
    finally:
        if old is None:
            os.environ.pop("MLX_OMARCHY_GREEDY_FUSE", None)
        else:
            os.environ["MLX_OMARCHY_GREEDY_FUSE"] = old
    return int(token.item()), stats.tolist()


def _composed(x, w, scales, biases):
    logits = mx.quantized_matmul(
        x, w, scales, biases, True, group_size=64, bits=4
    )
    logprobs = logits - mx.logsumexp(logits, -1, True)
    return int(mx.argmax(logprobs, -1).item())


def test_route_engages():
    """The op must take the pruned GPU head (composed fallback stats end in 0)."""
    rng = np.random.default_rng(0)
    args = _inputs(rng)
    _, stats = _greedy(*args, fused=False)
    assert stats[2] != 0, f"greedy head fell back to composed ops: stats={stats}"


def test_stats_identical_across_flags():
    """The fused path must report the same survivors/full/route as the 8-stage."""
    rng = np.random.default_rng(1)
    args = _inputs(rng)
    token_off, stats_off = _greedy(*args, fused=False)
    token_on, stats_on = _greedy(*args, fused=True)
    assert stats_off == stats_on, (
        f"fused stats diverge: off={stats_off} on={stats_on}"
    )
    assert token_off == token_on


def test_ties_keep_first_index_across_flags_and_composed():
    """Duplicate the true argmax row: the smallest index must win everywhere."""
    rng = np.random.default_rng(2)
    top, _ = _greedy(*_inputs(np.random.default_rng(2)), fused=False)
    dsts = [i for i in (0, 1, N - 2, N - 1) if i != top][:2]
    args = _inputs(rng, dup_rows=[(top, d) for d in dsts])
    token_off, _ = _greedy(*args, fused=False)
    token_on, _ = _greedy(*args, fused=True)
    composed = _composed(*args[:4])
    expected = min([top] + dsts)
    assert token_off == token_on == composed == expected, (
        f"tie divergence: off={token_off} on={token_on} "
        f"composed={composed} expected={expected}"
    )


@pytest.mark.parametrize("seed", range(8))
def test_randomized_identity_matches_flag_off_and_composed(seed):
    rng = np.random.default_rng(100 + seed)
    args = _inputs(rng)
    token_off, stats_off = _greedy(*args, fused=False)
    token_on, stats_on = _greedy(*args, fused=True)
    composed = _composed(*args[:4])
    assert token_off == token_on == composed, (
        f"seed {seed}: off={token_off} on={token_on} composed={composed}"
    )
    assert stats_off == stats_on, (
        f"seed {seed}: off={stats_off} on={stats_on}"
    )


if __name__ == "__main__":
    # Plain-runner form for guard venvs without pytest: same cases, same
    # order, nonzero exit on the first red.
    import sys

    cases = [test_route_engages, test_stats_identical_across_flags,
             test_ties_keep_first_index_across_flags_and_composed]
    cases += [lambda s=s: test_randomized_identity_matches_flag_off_and_composed(s)
              for s in range(8)]
    failed = 0
    for case in cases:
        try:
            case()
            print(f"PASS {case.__name__ if hasattr(case, '__name__') else case}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL {getattr(case, '__name__', case)}: {error}")
    sys.exit(1 if failed else 0)
