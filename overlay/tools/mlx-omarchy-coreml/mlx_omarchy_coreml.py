#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""mlx-omarchy-coreml — Core ML front-end for mlx-omarchy.

Subcommands:

  inspect PACKAGE [--json]     section-39 eligibility report
  check PACKAGE --compute-target TARGET
                               sections 36-37 capability check: exits 0
                               when the target can execute the package,
                               4 when blocked (reason printed), 2 on
                               usage/environment errors
"""

import argparse
import json
import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[1])

if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from coreml.eligibility import eligibility_report  # noqa: E402
from coreml.model import CoreMLError, CoreMLModel  # noqa: E402


def _inspect(args) -> int:
    report = eligibility_report(Path(args.package))
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"package: {report['package']}")
    print(f"total ops: {report['total_ops']}")
    for disposition, count in report["counts"].items():
        print(f"  {disposition}: {count}")
    if report["blocking_ops"]:
        print("blocking ops (compiler-side):")
        for entry in report["blocking_ops"]:
            print(f"  {entry['op']} x{entry['count']}: {entry['reason']}")
    else:
        print("no blocking ops")
    post = report["post_frontend_histogram"]
    print(
        "post-frontend histogram: "
        + ", ".join(f"{op}={count}" for op, count in post.items())
    )
    return 0


def _check(args) -> int:
    model = CoreMLModel.load(args.package)
    eligibility = model.eligibility()
    print(f"package: {model.path}")
    print(
        "eligibility: "
        + ", ".join(f"{k}={v}" for k, v in eligibility.counts.items())
    )
    try:
        model.check_compute_target(args.compute_target)
    except CoreMLError as error:
        print(f"compute_target={args.compute_target!r} REFUSED: {error}")
        return 4
    print(f"compute_target={args.compute_target!r} eligible")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mlx-omarchy-coreml")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="eligibility report")
    inspect.add_argument("package")
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_inspect)

    check = sub.add_parser(
        "check", help="capability check for a compute target"
    )
    check.add_argument("package")
    check.add_argument("--compute-target", required=True)
    check.set_defaults(handler=_check)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # Downstream closed the pipe (e.g. `| head`); exit quietly.
        raise SystemExit(0)
    except CoreMLError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
