# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Tie-aware greedy-decode MATCH for the torch-mlx-vulkan parity harness.

The compiled path emits logits at bf16 resolution (measured quantization:
gaps arrive in 0.0625 steps), so a reference top-2 gap below that ulp
collapses to an exact tie on device — Qwen3-0.6B token 5 has a 0.0242
fp32 gap on CPU and a literal 0.0 gap through mlx_vulkan. Which side of a
tie argmax returns is a property of the build's fusion order, not of
correctness, so plain id equality turns a coin flip into a false
regression report (see the #787 bisect: pre-window, branch, and head
wheels all flip the same step).

The contract here: a divergence is defensible only when the reference
itself was nearly indifferent (top-2 gap < TIE_TOL) and the device picked
the reference's other candidate. In that case the reference token is fed
forward so every later step is still compared on the same context.
"""

TIE_TOL = 0.1  # above one bf16 ulp at logit ~10 (0.0625); the next-smallest observed step gap is 0.19


def step_row(last_logits):
    """Reduce a 1D logits tensor to {tok, alt, gap} for the step."""
    vals, idxs = last_logits.topk(2)
    return {"tok": idxs[0].item(), "alt": idxs[1].item(),
            "gap": (vals[0] - vals[1]).item()}


def guided_token(row, ref_row, tol=TIE_TOL):
    """Return (token_to_feed, mark) for one decode step.

    mark is "match" (same argmax), "tie" (reference was within tol and the
    device took the other candidate — feed the reference token so the
    contexts stay aligned), or "diff" (a real divergence — feed what the
    device picked so the failure develops naturally).
    """
    if row["tok"] == ref_row["tok"]:
        return row["tok"], "match"
    if ref_row["gap"] < tol and row["tok"] in (ref_row["tok"], ref_row["alt"]):
        return ref_row["tok"], "tie"
    return row["tok"], "diff"


def match_report(marks):
    """Fold per-step marks into the printed MATCH fields."""
    ties = [i for i, m in enumerate(marks) if m == "tie"]
    diffs = [i for i, m in enumerate(marks) if m == "diff"]
    return {"match": not diffs, "ties": ties,
            "first_diff": diffs[0] if diffs else -1}
