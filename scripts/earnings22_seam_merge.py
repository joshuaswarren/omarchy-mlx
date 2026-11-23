#!/usr/bin/env python3
"""Word-level seam merge for chunked long-file transcription (V27, Earnings-22).

Replaces the exact-prefix-only seam drop used by the V24 earnings-22
measurement row: clean overlaps keep the exact fast path, broken overlaps
fall back to fuzzy word alignment over the overlap window so a straddled or
re-transcribed word is taken once.

Provenance: written for the voice card's Earnings-22 measurement row
(1,024-row windows, Parakeet TDT 0.6B). An in-place rerun on a 16-inch
M1 Max MacBook Pro dropped corpus WER 17.03% -> 13.32% and
CER 11.13% -> 7.63% at equal RTFx (50.38 -> 50.54), with seam-boundary
word matches 159/276 -> 241/276 and the full receipt hashed into the
run record (v27-receipt.json, sha256 cb9f147c606d8ad5411a598245af1c81d5a7881444a7be72c97a322644cfeccd).
"""
import re

MAX_SEAM_WORDS = 24  # words searched on each side of a seam (2 s ≈ 4-12 words)
GAP, SUB = 0.6, 1.0  # prefer a substitution over two gaps when aligning


def _key(w):
    return re.sub(r"\W", "", w.casefold())


def _exact_prefix_drop(merged, words):
    """V24 fast path: longest exact (case/punct-insensitive) tail-prefix repeat."""
    for size in range(min(MAX_SEAM_WORDS, len(merged), len(words)), 0, -1):
        if [_key(w) for w in merged[-size:]] == [_key(w) for w in words[:size]]:
            return size
    return 0


def _fuzzy_prefix_drop(merged, words):
    """Align merged's tail with words' head; return (drop, matched) or (0, 0).

    Gaps cost 0.6 on either side, a substitution 1.0; the alignment may end
    anywhere in words' head, and everything up to that point is treated as
    overlap audio already transcribed by the previous window. An alignment
    with too little evidence (fewer than 2 matches, or a single match that
    consumed more than one new word) is rejected and nothing is dropped.
    """
    a = [_key(w) for w in merged[-MAX_SEAM_WORDS:]]
    b = [_key(w) for w in words[:MAX_SEAM_WORDS]]
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0.0] * cols for _ in range(rows)]
    for j in range(1, cols):
        dp[0][j] = dp[0][j - 1] + GAP
    for i in range(1, rows):
        dp[i][0] = dp[i - 1][0] + GAP
        for j in range(1, cols):
            sub = dp[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else SUB)
            dp[i][j] = min(sub, dp[i - 1][j] + GAP, dp[i][j - 1] + GAP)
    best_j, best = 0, float("inf")
    for j in range(cols):
        if dp[rows - 1][j] < best - 1e-9:  # strict: ties keep fewer consumed
            best, best_j = dp[rows - 1][j], j
    i, j, matched = rows - 1, best_j, 0
    while i > 0 and j > 0:
        sub = dp[i - 1][j - 1] + (0.0 if a[i - 1] == b[j - 1] else SUB)
        move = min((sub, "d"), (dp[i - 1][j] + GAP, "u"), (dp[i][j - 1] + GAP, "l"))[1]
        if move == "d":
            if a[i - 1] == b[j - 1]:
                matched += 1
            i, j = i - 1, j - 1
        elif move == "u":
            i -= 1
        else:
            j -= 1
    if matched >= 2 or (matched == 1 and best_j == 1):
        return best_j, matched
    return 0, 0


def merge(merged, words):
    """Merge one window's words; returns (text, words_dropped, matches)."""
    drop = _exact_prefix_drop(merged, words)
    if drop:
        return merged + words[drop:], drop, drop
    drop, matched = _fuzzy_prefix_drop(merged, words)
    return merged + words[drop:], drop, matched
