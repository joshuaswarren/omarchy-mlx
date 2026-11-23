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
import sys
from pathlib import Path

MAX_DELTA = 0.05


def delta_frac(kit_tps, packaged_tps):
    """Relative difference between the kit and packaged decode rates.

    The packaged install is the denominator: it is the surface users run.
    """
    return abs(kit_tps - packaged_tps) / packaged_tps


def verdict(kit_tps, packaged_tps, max_delta=MAX_DELTA):
    """Compare two decode tok/s rates. Refuse above max_delta, or when a
    rate is missing or non-positive — a gate that cannot read both sides
    must not pass."""
    try:
        kit = float(kit_tps)
        pkg = float(packaged_tps)
    except (TypeError, ValueError):
        return {'verdict': 'refuse', 'reason': 'unreadable_rate',
                'delta_frac': None}
    if kit <= 0 or pkg <= 0:
        return {'verdict': 'refuse', 'reason': 'missing_rate',
                'delta_frac': None}
    d = delta_frac(kit, pkg)
    if d > max_delta:
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

    v = verdict(args.kit_tps, args.packaged_tps, args.max_delta)
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
    Path(args.out).write_text(json.dumps(rec, indent=2) + '\n')
    print(f"decode-pair {rec['verdict']}: kit {args.kit_tps} vs packaged "
          f"{args.packaged_tps} tok/s (delta {v['delta_frac']}, "
          f"max {args.max_delta})")
    if rec['verdict'] == 'refuse':
        print(f"REFUSING tag {args.tag}: {v['reason']}", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
