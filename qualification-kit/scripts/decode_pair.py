#!/usr/bin/env python3
"""Kit-vs-packaged decode pair gate for qualification kit tags.

Before a qualification tag is cut, the kit's disposable-venv decode and the
packaged install's decode must agree on the same model on the reference M1
Max. The kit installs the pinned wheel plus the mlx-lm patch series; the
packaged install is what users actually run. A drift above five percent
means the two surfaces diverge and the tag is refused.

The pair, delta and verdict are written as JSON so the release receipt can
embed the measured numbers verbatim.
"""

import argparse
import json
import math
import sys
from pathlib import Path

MAX_DELTA = 0.05


def delta_frac(kit_tps, packaged_tps):
    """Relative difference between the kit and packaged decode rates.

    The packaged install is the denominator: it is the surface users run.
    """
    return abs(kit_tps - packaged_tps) / packaged_tps


def finite(value):
    """float(value) when it is a finite number, else None. A bool is not a
    number here, and NaN or an infinity cannot be compared."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def verdict(kit_tps, packaged_tps, max_delta=MAX_DELTA,
            kit_ids=None, packaged_ids=None):
    """Compare two decode tok/s rates. Refuse above max_delta, when a rate
    is missing, non-finite or non-positive, or when max_delta is not a
    finite non-negative number — a gate that cannot read both sides must
    not pass. When both generated-token digests are present they must be
    equal: matching speeds over different output is not a pass."""
    if kit_ids and packaged_ids and kit_ids != packaged_ids:
        return {'verdict': 'refuse', 'reason': 'ids_divergence',
                'delta_frac': None}
    kit = finite(kit_tps)
    pkg = finite(packaged_tps)
    if kit is None or pkg is None:
        return {'verdict': 'refuse', 'reason': 'unreadable_rate',
                'delta_frac': None}
    if kit <= 0 or pkg <= 0:
        return {'verdict': 'refuse', 'reason': 'missing_rate',
                'delta_frac': None}
    limit = finite(max_delta)
    if limit is None or limit < 0:
        return {'verdict': 'refuse', 'reason': 'invalid_max_delta',
                'delta_frac': None}
    d = delta_frac(kit, pkg)
    if not d <= limit:
        return {'verdict': 'refuse', 'reason': 'rate_divergence',
                'delta_frac': d}
    return {'verdict': 'pass', 'reason': 'within_delta', 'delta_frac': d}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--kit-tps', type=float, required=True)
    ap.add_argument('--packaged-tps', type=float, required=True)
    ap.add_argument('--kit-ids', default=None)
    ap.add_argument('--packaged-ids', default=None)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokens', type=int, required=True)
    ap.add_argument('--tag', required=True,
                    help='qualification tag being gated, e.g. qualification-r8')
    ap.add_argument('--out', required=True, help='receipt JSON path')
    ap.add_argument('--max-delta', type=float, default=MAX_DELTA)
    args = ap.parse_args(argv)

    v = verdict(args.kit_tps, args.packaged_tps, args.max_delta,
                kit_ids=args.kit_ids, packaged_ids=args.packaged_ids)
    rec = {
        'schema': 'decode-pair-gate-v1',
        'tag': args.tag,
        'model': args.model,
        'tokens': args.tokens,
        'kit_decode_tps': args.kit_tps,
        'packaged_decode_tps': args.packaged_tps,
        'kit_ids_sha': args.kit_ids,
        'packaged_ids_sha': args.packaged_ids,
        'ids_match': (args.kit_ids == args.packaged_ids
                      if args.kit_ids and args.packaged_ids else None),
        'max_delta_frac': args.max_delta,
        **v,
    }
    # JSON has no NaN or infinity: write a non-finite number as null.
    rec = {k: (None if isinstance(x, float) and not math.isfinite(x) else x)
           for k, x in rec.items()}
    Path(args.out).write_text(json.dumps(rec, indent=2, allow_nan=False)
                              + '\n')
    print(f"decode-pair {rec['verdict']}: kit {args.kit_tps} vs packaged "
          f"{args.packaged_tps} tok/s (delta {v['delta_frac']}, "
          f"max {args.max_delta})")
    if rec['verdict'] == 'refuse':
        print(f"REFUSING tag {args.tag}: {v['reason']}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
