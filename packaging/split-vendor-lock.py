#!/usr/bin/env python3
"""Partition a vendor lock into resolvable and --no-deps requirement files.

    split-vendor-lock.py --nodeps requirements-nodeps.in --wheels DIR \
        --lock requirements-lock.txt --resolvable OUT --nodeps-only OUT

pip --require-hashes resolves every line it is given. A --no-deps wheel that
declares a deliberately absent dependency (a "not-vendored" line, e.g.
tflite-runtime) can never resolve, so it installs with pip --no-deps. Every
other lock line — including a --no-deps wheel whose own dependencies are all
vendored — must resolve offline; verify-voice-venv.sh runs exactly that
two-stage install inside a route-less network namespace.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "nodeps_closure", Path(__file__).resolve().with_name("check-nodeps-closure.py"))
closure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closure)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nodeps", required=True)
    ap.add_argument("--wheels", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--resolvable", required=True)
    ap.add_argument("--nodeps-only", required=True)
    args = ap.parse_args()

    pins, exempt = closure.read_nodeps(args.nodeps)
    nodeps_only = set()
    for pin in pins:
        for text in closure.wheel_requires(args.wheels, pin):
            dep = closure.Requirement(text)
            if dep.marker is not None and not dep.marker.evaluate(closure.TARGET):
                continue
            if closure.norm(dep.name) in exempt:
                nodeps_only.add(closure.norm(pin.name))
                break

    resolvable, forced = [], []
    for line in Path(args.lock).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name = closure.norm(line.split("==", 1)[0].strip())
        (forced if name in nodeps_only else resolvable).append(line)

    Path(args.resolvable).write_text("\n".join(resolvable) + "\n")
    Path(args.nodeps_only).write_text("\n".join(forced) + ("\n" if forced else ""))
    print(f"resolvable: {len(resolvable)} line(s); --no-deps: {len(forced)} line(s) "
          f"({', '.join(sorted(nodeps_only)) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
