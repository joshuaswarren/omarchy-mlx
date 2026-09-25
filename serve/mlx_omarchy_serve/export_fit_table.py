"""Export a precomputed model-fit table for static sites (joshuawarren.com/models).

Every number here comes from budget.estimate_required() or catalog fields —
this module adds no new budgeting math. For each catalog model and each
common Apple Silicon unified-memory size it records:

  max_context          the largest context (<= the model's max_tokens) whose
                       estimate_required total fits in N GiB minus the
                       2 GiB safety reserve
  fits                 the same gates the serve CLI applies at the model's
                       default context (resolve_context(ctx, None)):
                       default-context estimate fits in N GiB minus the
                       2 GiB safety reserve, AND capability.min_mem_gib <= N
  peak_estimate_bytes  estimate_required total at max_context when the model
                       fits; otherwise the total at its default context,
                       i.e. what running it out of the box would need

Binary search is valid because estimate_required is monotonic non-decreasing
in context_tokens.

Usage (from serve/):  python3 -m mlx_omarchy_serve.export_fit_table [-o out.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import budget
from .catalog import bundled_path, validate_catalog

MEMORY_SIZES_GIB = [8, 16, 24, 32, 36, 48, 64, 96, 128, 192]


def _max_fitting_context(memory: dict, max_tokens: int, available_bytes: int) -> int:
    if budget.estimate_required(memory, max_tokens).total <= available_bytes:
        return max_tokens
    if budget.estimate_required(memory, 0).total > available_bytes:
        return 0
    lo, hi = 0, max_tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if budget.estimate_required(memory, mid).total <= available_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _fit_cell(entry: dict, gib: int) -> dict:
    available = gib * budget.GiB - budget.SAFETY_RESERVE_BYTES
    max_tokens = entry["context"]["max_tokens"] or budget.DEFAULT_CONTEXT_TOKENS
    default_context = budget.resolve_context(entry["context"], None)
    max_context = _max_fitting_context(entry["memory"], max_tokens, available)
    floor_gib = entry["capability"]["min_mem_gib"]
    fits = (
        max_context >= default_context
        and (floor_gib is None or floor_gib <= gib)
    )
    peak_context = max_context if fits else default_context
    peak = budget.estimate_required(entry["memory"], peak_context).total
    return {"fits": fits, "max_context": max_context, "peak_estimate_bytes": peak}


def _model_row(entry: dict) -> dict:
    quant = entry["quant"]
    return {
        "id": entry["id"],
        "label": entry["repo"].split("/", 1)[1],
        "family": entry["family"],
        "kind": entry["kind"],
        "license": entry["license"],
        "repo": entry["repo"],
        "revision": entry["revision"],
        "recommended": entry["recommended"],
        "generation_status": entry["qualification"]["generation"]["status"],
        "quant": None if quant is None else {
            "bits": quant["bits"], "mode": quant["mode"], "group_size": quant["group_size"],
        },
        "weights_bytes": entry["memory"]["weights_bytes"],
        "kv_bytes_per_token": entry["memory"]["kv_bytes_per_token"],
        "max_tokens": entry["context"]["max_tokens"],
        "min_mem_gib": entry["capability"]["min_mem_gib"],
        "fits": {str(gib): _fit_cell(entry, gib) for gib in MEMORY_SIZES_GIB},
    }


def build_fit_table(catalog_path: Path | None = None) -> dict:
    path = catalog_path or bundled_path()
    cat = json.loads(path.read_text(encoding="utf-8"))
    validate_catalog(cat)
    return {
        "exported_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "catalog": {"version": cat["version"], "generated_at": cat["generated_at"], "source": cat["source"]},
        "budget": {
            "safety_reserve_bytes": budget.SAFETY_RESERVE_BYTES,
            "min_usable_context_tokens": budget.DEFAULT_CONTEXT_TOKENS,
            "workspace_fraction": budget.WORKSPACE_FRACTION,
            "workspace_min_bytes": budget.WORKSPACE_MIN_BYTES,
            "unknown_kv_margin_bytes": budget.UNKNOWN_KV_MARGIN,
        },
        "memory_sizes_gib": MEMORY_SIZES_GIB,
        "models": [_model_row(e) for e in cat["models"]],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog", type=Path, default=None, help="catalog.json path (default: bundled)")
    ap.add_argument("-o", "--out", type=Path, default=None, help="write JSON here (default: stdout)")
    args = ap.parse_args(argv)
    table = build_fit_table(args.catalog)
    text = json.dumps(table, indent=2, sort_keys=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
