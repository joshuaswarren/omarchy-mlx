"""bf16 spacing at a logit magnitude."""
import math


def bf16_ulp(magnitude):
    magnitude = abs(float(magnitude))
    if not math.isfinite(magnitude):
        raise ValueError("magnitude must be finite")
    if magnitude < 2.0 ** -126:
        return 2.0 ** -133
    return math.ldexp(1.0, math.frexp(magnitude)[1] - 8)


def within_one_bf16_ulp(gap, magnitude):
    return float(gap) <= bf16_ulp(magnitude)
