# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Benchmark timing guard for mlx-lm's deferred token yield patch.

The mlx-omarchy mlx-lm patch series enables MLX_OMARCHY_DEFER_TOKENS=8 by
default in mlx_lm.generate.generate_step. When active, next(gen) buffers 8
tokens before yielding the first token, shifting ~7 decode steps out of the
(t_last - t_first) decode window and into prefill while draining 7 zero-ms
gaps at the end. Any runner that computes (N - 1) / (t_last - t_first) then
over-reports decode tok/s by (N - 1) / (N - 8) (+5.8% at N=128, +12.5% at
N=64, +29.2% at N=32) even when total wall time is unchanged.

Standard benchmark runners pin MLX_OMARCHY_DEFER_TOKENS=0, record the active
value in their output, and refuse to print tok/s if the variable is unset
while the installed mlx_lm.generate.generate_step defaults to deferring.
"""

import inspect
import os
import re
import sys

ENV_VAR = "MLX_OMARCHY_DEFER_TOKENS"

_DEFER_DEFAULT_RE = re.compile(
    r'os\.environ\.get\(\s*["\']MLX_OMARCHY_DEFER_TOKENS["\']\s*,'
    r'\s*["\']?(-?\d+)["\']?\s*\)'
)


def _generate_step_source(generate_step_fn=None):
    """Return source text of mlx_lm.generate.generate_step, or None."""
    if isinstance(generate_step_fn, str):
        return generate_step_fn
    fn = generate_step_fn
    if fn is None:
        try:
            from mlx_lm.generate import generate_step as fn
        except Exception:
            return None
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return None


def package_default_defers(generate_step_fn=None):
    """Return (defers, default_value) for mlx_lm.generate.generate_step.

    Returns (True, N) when the installed generate_step reads
    MLX_OMARCHY_DEFER_TOKENS with a positive default N (or when the env var
    appears without a parseable integer default, failing closed). Returns
    (False, 0) for stock upstream mlx-lm or a non-positive default.
    """
    src = _generate_step_source(generate_step_fn)
    if not src or ENV_VAR not in src:
        return False, 0
    m = _DEFER_DEFAULT_RE.search(src)
    if m is None:
        return True, None
    default_val = int(m.group(1))
    return default_val > 0, default_val


def pin_defer_tokens(env=None, value="0"):
    """Pin MLX_OMARCHY_DEFER_TOKENS to value when unset or empty."""
    target = os.environ if env is None else env
    cur = target.get(ENV_VAR)
    if cur is None or not str(cur).strip():
        target[ENV_VAR] = str(value)
    return str(target[ENV_VAR]).strip()


def require_defer_tokens(tool, env=None, *, generate_step_fn=None):
    """Return the recorded MLX_OMARCHY_DEFER_TOKENS value or refuse (exit 3).

    Refuses without emitting any rate when MLX_OMARCHY_DEFER_TOKENS is unset
    (or empty) and the installed mlx_lm.generate.generate_step defaults to a
    positive defer depth. Returns None only when unset on a non-deferring
    package (e.g., stock upstream mlx-lm).
    """
    target = os.environ if env is None else env
    raw = target.get(ENV_VAR)
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    defers, default_val = package_default_defers(generate_step_fn)
    if defers:
        desc = f"default={default_val}" if default_val is not None else "non-zero default"
        sys.stderr.write(
            f"[{tool}] REFUSING TO EMIT TOK/S: {ENV_VAR} is unset while "
            f"mlx_lm.generate.generate_step defers token yield ({desc}). "
            f"Deferred yield shifts decode steps into prefill and inflates "
            f"(N-1)/(t_last-t_first); set {ENV_VAR}=0 before measuring.\n"
        )
        raise SystemExit(3)
    return None
