#!/usr/bin/env python3
"""Fail when a --no-deps wheel needs a package the lock does not carry.

    check-nodeps-closure.py --nodeps requirements-nodeps.in --wheels DIR --lock FILE

pip installs the packages in requirements-nodeps.in without their
dependencies, so their Requires-Dist lines are copied by hand into
requirements-lock.in. This reads each wheel's METADATA and checks every
dependency that applies on the target (Linux, aarch64, CPython 3.14, no
extras) against FILE: the release lock (name==version, version checked
against the specifier) or requirements-lock.in (name present). A dependency
that is deliberately absent is declared in the nodeps file as
"# not-vendored: <name> <reason>".
"""
import argparse
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:  # pip vendors the same library
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version

TARGET = {"platform_system": "Linux", "sys_platform": "linux", "platform_machine": "aarch64",
          "os_name": "posix", "implementation_name": "cpython", "platform_python_implementation": "CPython",
          "python_version": "3.14", "python_full_version": "3.14.0", "extra": ""}


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_nodeps(path):
    pins, exempt = [], {}
    for raw in Path(path).read_text().splitlines():
        found = re.match(r"#\s*not-vendored:\s*(\S+)\s+(\S.*)", raw.strip())
        if found:
            exempt[norm(found.group(1))] = found.group(2)
            continue
        line = raw.split("#", 1)[0].strip()
        if line:
            pins.append(Requirement(line))
    return pins, exempt


def read_lock(path):
    """name -> pinned Version, or None for a specifier/URL line."""
    lock = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pinned = re.match(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)==(\S+)\s+--hash=", line)
        if pinned:
            lock[norm(pinned.group(1))] = Version(pinned.group(2))
        else:
            lock.setdefault(norm(Requirement(line).name), None)
    return lock


def wheel_requires(wheels, pin):
    version = next(iter(pin.specifier)).version
    prefix = f"{norm(pin.name).replace('-', '_')}-{version}-"
    for wheel in Path(wheels).glob("*.whl"):
        if norm(wheel.name.split("-", 1)[0]) == norm(pin.name) and wheel.name.lower().startswith(prefix.lower()):
            with zipfile.ZipFile(wheel) as zf:
                meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
                return Parser().parsestr(zf.read(meta).decode()).get_all("Requires-Dist") or []
    raise SystemExit(f"error: no wheel for {pin} in {wheels}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nodeps", required=True)
    ap.add_argument("--wheels", required=True)
    ap.add_argument("--lock", required=True)
    args = ap.parse_args()
    pins, exempt = read_nodeps(args.nodeps)
    lock = read_lock(args.lock)
    problems = []
    for pin in pins:
        for text in wheel_requires(args.wheels, pin):
            dep = Requirement(text)
            if dep.marker is not None and not dep.marker.evaluate(TARGET):
                continue
            name = norm(dep.name)
            if name in exempt:
                continue
            if name not in lock:
                problems.append(f"{pin.name} needs {dep}, which the lock does not carry")
            elif lock[name] is not None and not dep.specifier.contains(lock[name], prereleases=True):
                problems.append(f"{pin.name} needs {dep}, but the lock pins {name}=={lock[name]}")
    for line in problems:
        print(f"error: {line}", file=sys.stderr)
    if problems:
        return 1
    print(f"ok: the lock covers the dependencies of {', '.join(str(p) for p in pins)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
