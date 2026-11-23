# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""FUSE-5: the two-stage greedy head pins the eight-stage head's token.

MLX_OMARCHY_GREEDY_TWO_STAGE=1 (read per call, default off) replaces the
eight sequential greedy dispatches with one partial dispatch (every
row's exact logit through the same Q4 word column, in row order) and one
final dispatch (the full path's logsumexp, bf16 subtract, and
first-index argmax, statement for statement). The token is the pinned
identity: fixed duplicate-row ties and randomized inputs must give the
same token flag on and off, and both must equal the composed
quantized_matmul + logsumexp + subtract + argmax expression. The flagged
path always runs the full exact pass, so its stats report
(rows=n, full=1, logsumexp bits); before the implementation exists the
old path ignores the flag and prune-decided tokens report full=0, which
is what fails red below.
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


def _greedy(x, w, scales, biases, plane, two_stage):
    old = os.environ.get("MLX_OMARCHY_GREEDY_TWO_STAGE")
    os.environ["MLX_OMARCHY_GREEDY_TWO_STAGE"] = "1" if two_stage else "0"
    try:
        token, stats = mx.fast.greedy_quantized_argmax(
            x, w, scales, biases, plane, group_size=64, bits=4
        )
        mx.eval(token, stats)
    finally:
        if old is None:
            os.environ.pop("MLX_OMARCHY_GREEDY_TWO_STAGE", None)
        else:
            os.environ["MLX_OMARCHY_GREEDY_TWO_STAGE"] = old
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
    _, stats = _greedy(*args, two_stage=False)
    assert stats[2] != 0, f"greedy head fell back to composed ops: stats={stats}"


def test_two_stage_markers_red_until_implemented():
    """Flagged tokens report the full-path markers; the old path ignores the flag."""
    rng = np.random.default_rng(1)
    args = _inputs(rng)
    _, stats = _greedy(*args, two_stage=True)
    assert stats[0] == N and stats[1] == 1 and stats[2] != 0, (
        f"two-stage markers absent: stats={stats}"
    )


def test_ties_keep_first_index_across_flags_and_composed():
    """Duplicate rows pin tie-breaking: the smallest index must win everywhere."""
    rng = np.random.default_rng(2)
    args = _inputs(rng, dup_rows=[(5, 100), (7, 8), (13, 13), (13, 400)])
    token_off, _ = _greedy(*args, two_stage=False)
    token_on, _ = _greedy(*args, two_stage=True)
    composed = _composed(*args)
    assert token_off == token_on == composed, (
        f"tie divergence: off={token_off} on={token_on} composed={composed}"
    )
    assert token_off in (5, 7, 13), f"unexpected tie token {token_off}"


@pytest.mark.parametrize("seed", range(8))
def test_randomized_identity_matches_flag_off_and_composed(seed):
    rng = np.random.default_rng(100 + seed)
    args = _inputs(rng)
    token_off, _ = _greedy(*args, two_stage=False)
    token_on, _ = _greedy(*args, two_stage=True)
    composed = _composed(*args)
    assert token_off == token_on == composed, (
        f"seed {seed}: off={token_off} on={token_on} composed={composed}"
    )


if __name__ == "__main__":
    # Plain-runner form for guard venvs without pytest: same cases, same
    # order, nonzero exit on the first red.
    import sys

    cases = [test_route_engages, test_two_stage_markers_red_until_implemented,
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
