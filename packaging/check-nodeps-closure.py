#!/usr/bin/env python3
"""Fail when a --no-deps wheel needs a package the lock does not carry.

    check-nodeps-closure.py --nodeps requirements-nodeps.in --wheels DIR --lock FILE

pip installs the packages in requirements-nodeps.in, each pinned as
name==version, without their dependencies, so their Requires-Dist lines
are copied by hand into requirements-lock.in. This reads each wheel's
METADATA and checks every dependency that applies on the target (Linux,
aarch64, CPython 3.14, no extras) against FILE: the release lock
(name==version, version checked against the specifier) or
requirements-lock.in (name present). A dependency that is deliberately
absent is declared in the nodeps file as "# not-vendored: <name> <reason>".
"""
import argparse
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

try:
    from packaging.requirements import Requirement
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from packaging.version import InvalidVersion, Version
except ImportError:  # pip vendors the same library
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import InvalidWheelFilename, parse_wheel_filename
    from pip._vendor.packaging.version import InvalidVersion, Version

TARGET = {"platform_system": "Linux", "sys_platform": "linux", "platform_machine": "aarch64",
          "os_name": "posix", "implementation_name": "cpython", "platform_python_implementation": "CPython",
          "python_version": "3.14", "python_full_version": "3.14.0", "extra": ""}


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lines(path):
    try:
        return Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SystemExit(f"error: cannot read {path}: {exc}")


def parse(text, where):
    try:
        return Requirement(text)
    except (ValueError, SyntaxError) as exc:  # InvalidRequirement; older packaging: InvalidSpecifier, SyntaxError
        raise SystemExit(f"error: {where}: invalid requirement {text!r}: {str(exc).splitlines()[0]}")


def read_nodeps(path):
    pins, exempt = [], {}
    for number, raw in enumerate(read_lines(path), 1):
        found = re.match(r"#\s*not-vendored:\s*(\S+)\s+(\S.*)", raw.strip())
        if found:
            exempt[norm(found.group(1))] = found.group(2)
            continue
        line = raw.split("#", 1)[0].strip()
        if line:
            pin = parse(line, f"{path}:{number}")
            specs = list(pin.specifier)
            if (len(specs) != 1 or specs[0].operator != "==" or specs[0].version.endswith("*")
                    or pin.extras or pin.marker):
                raise SystemExit(f"error: {path}:{number}: {line!r} must pin one version as name==version")
            pins.append(pin)
    return pins, exempt


def read_lock(path):
    """name -> pinned Version, or None for a specifier/URL line."""
    lock = {}
    for number, raw in enumerate(read_lines(path), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        pinned = re.match(r"([A-Za-z0-9][A-Za-z0-9_.\-]*)==(\S+)\s+--hash=", line)
        if pinned:
            try:
                lock[norm(pinned.group(1))] = Version(pinned.group(2))
            except ValueError as exc:  # InvalidVersion or Python's int-digit limit
                raise SystemExit(f"error: {path}:{number}: {exc}")
        else:
            lock.setdefault(norm(parse(line, f"{path}:{number}").name), None)
    return lock


def wheel_requires(wheels, pin):
    """The Requires-Dist lines of the wheel for pin, a name==version pin from read_nodeps()."""
    try:
        version = Version(next(iter(pin.specifier)).version)
    except ValueError as exc:  # Python's int-digit limit
        raise SystemExit(f"error: nodeps pin {pin.name}: {exc}")
    for wheel in sorted(Path(wheels).glob("*.whl")):
        try:
            name, found, _, tags = parse_wheel_filename(wheel.name)
        except (InvalidWheelFilename, InvalidVersion):  # packaging < 23.2 raises InvalidVersion
            continue
        if name != norm(pin.name) or found != version:
            continue
        if not all(t.interpreter and t.abi and t.platform for t in tags):  # older packaging accepts foo-1---.whl
            continue
        try:
            with zipfile.ZipFile(wheel) as zf:
                meta = next((n for n in zf.namelist()
                             if n.count("/") == 1 and n.endswith(".dist-info/METADATA")), None)
                if meta is None:
                    raise SystemExit(f"error: {wheel} has no .dist-info/METADATA")
                text = zf.read(meta).decode("utf-8")
        except Exception as exc:  # zipfile, its decompressors and decode raise many types
            raise SystemExit(f"error: cannot read {wheel}: {exc}")
        metadata = Parser().parsestr(text)
        if metadata.defects:  # a malformed header line drops the headers after it
            raise SystemExit(f"error: {wheel} METADATA has a malformed header line: {metadata.defects[0]!r}")
        requires = metadata.get_all("Requires-Dist") or []
        for line in requires:
            dep = parse(line, f"{wheel} Requires-Dist")
            try:
                if dep.marker is not None:
                    dep.marker.evaluate(TARGET)
            except (ValueError, KeyError) as exc:  # UndefinedComparison, UndefinedEnvironmentName, InvalidVersion
                raise SystemExit(f"error: {wheel} Requires-Dist: cannot evaluate {line!r}: {exc}")
        return requires
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
            try:  # comparing or printing a version over Python's int-digit limit raises ValueError
                if name not in lock:
                    problems.append(f"{pin.name} needs {dep}, which the lock does not carry")
                elif lock[name] is not None and not dep.specifier.contains(lock[name], prereleases=True):
                    problems.append(f"{pin.name} needs {dep}, but the lock pins {name}=={lock[name]}")
            except ValueError as exc:
                raise SystemExit(f"error: {pin.name} Requires-Dist {text!r}: {exc}")
    for line in problems:
        print(f"error: {line}", file=sys.stderr)
    if problems:
        return 1
    print(f"ok: the lock covers the dependencies of {', '.join(str(p) for p in pins)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
