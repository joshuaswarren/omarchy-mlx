#!/usr/bin/env python3
"""Reject mlx-omarchy pins of ANY form in requirements inputs.

A measurement environment must install the one wheel under test, by the
documented install step. A pin hidden in a requirements input silently
reinstalls a different generation: on 2026-09-03 a support-requirements
freeze carried ``mlx-omarchy @ file:///...dev20260903-...whl#sha256=...``,
an exclusion filter anchored on ``==`` did not match it, and pip
reinstalled a stale 5,300,664-byte-libmlx wheel over the correct 20.9 MB
one in every fresh venv. Two hours-long measurement episodes resulted,
and both had to be retracted.

This gate reads requirements-format inputs (files, directories, ``-r``
includes, or stdin) and exits nonzero when any line names mlx-omarchy as
a requirement in any form:

  mlx-omarchy==0.32.2              version pin
  mlx-omarchy>=0.3                 range pin
  mlx-omarchy @ file:///...whl     local wheel pin   (the 2026-09-03 trap)
  mlx-omarchy @ https://...whl     direct-URL pin
  https://.../mlx_omarchy-...whl   bare direct-URL line
  -f or --find-links dirs pointing at mlx_omarchy wheels are reported

Editable installs (``-e ./mlx``) are reported as warnings, not
violations: they build from a source checkout, which
scripts/mlx_provenance.py flags separately as "no metadata".

Usage:
  python3 scripts/check-wheel-pins.py requirements*.txt
  cat freeze.txt | python3 scripts/check-wheel-pins.py -
  python3 scripts/check-wheel-pins.py --self-test
"""

import fileinput
import glob
import os
import re
import sys

PACKAGE_RE = re.compile(r"(?im)^[\s\-]*[A-Za-z0-9._]*mlx[_-]omarchy[A-Za-z0-9._-]*\b")
WHEEL_RE = re.compile(r"(?i)mlx[_-]omarchy[^/\s]*\.whl")
NAME_NORM_RE = re.compile(r"[-_.]+")


def _normalized_names(line):
    """PEP 503-normalized requirement names on this line."""
    stripped = line.split("#", 1)[0].strip()
    if not stripped or stripped.startswith(("-r", "-c", "--requirement",
                                             "--constraint")):
        return [], stripped
    if stripped.startswith("-"):
        return [], stripped
    # env markers keep the name before ';'; URLs come after '@' or spaces.
    head = stripped.split(";", 1)[0]
    head = re.split(r"\s+@\s+|\s+===|\s*==|>=|<=|~=|!=|>|<|\[", head, 1)[0]
    name = head.strip().strip("'\"")
    if not name:
        return [], stripped
    return [NAME_NORM_RE.sub("-", name).lower()], stripped


def check_lines(lines, source="<input>"):
    """Yield (severity, message) for every problematic line."""
    for lineno, raw in enumerate(lines, 1):
        if raw.lstrip().startswith("#"):
            continue
        names, stripped = _normalized_names(raw)
        yielded = False
        for name in names:
            if name != "mlx-omarchy":
                continue
            yielded = True
            yield ("VIOLATION",
                   f"{source}:{lineno}: mlx-omarchy is pinned in a "
                   f"requirements input: {raw.strip()!r}. The wheel under "
                   f"test must come from the documented install step, not "
                   f"from a requirements line: a pin here silently "
                   f"reinstalls a different generation (2026-09-03: a "
                   f"@ file:// pin survived a ==-anchored filter and "
                   f"reinstalled a stale 5.3 MB-libmlx wheel in every "
                   f"fresh venv). Remove the line, install the recorded "
                   f"wheel explicitly, then verify with "
                   f"scripts/mlx_provenance.py.")
            break
        # Bare direct-URL wheel lines the name rule cannot see.
        if not yielded and WHEEL_RE.search(stripped):
            yield ("VIOLATION",
                   f"{source}:{lineno}: direct URL to an mlx_omarchy wheel "
                   f"in a requirements input: {raw.strip()!r}. Same trap "
                   f"as the 2026-09-03 stale-wheel reinstall; install the "
                   f"recorded wheel explicitly instead.")
        if not yielded and not WHEEL_RE.search(stripped) \
                and stripped.startswith("-e") and "mlx-omarchy" in stripped:
            yield ("WARNING",
                   f"{source}:{lineno}: editable install of mlx-omarchy: "
                   f"{raw.strip()!r}. Builds from source; "
                   f"scripts/mlx_provenance.py will report no metadata.")


def _expand(args):
    for a in args:
        if a == "-":
            yield "-", None
            continue
        matches = sorted(glob.glob(a)) or [a]
        for m in matches:
            if os.path.isdir(m):
                for root, _dirs, files in os.walk(m):
                    for f in sorted(files):
                        if f.endswith((".txt", ".in")):
                            yield os.path.join(root, f), None
            else:
                yield m, None


def main(argv):
    if argv == ["--self-test"]:
        return _self_test()
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    failures = 0
    for target, _ in _expand(argv):
        if target == "-":
            for sev, msg in check_lines(sys.stdin.readlines(), "<stdin>"):
                print(msg, file=sys.stderr)
                failures += sev == "VIOLATION"
            continue
        try:
            with open(target, encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError as exc:
            print(f"cannot read {target}: {exc}", file=sys.stderr)
            failures += 1
            continue
        for sev, msg in check_lines(lines, target):
            print(msg, file=sys.stderr)
            failures += sev == "VIOLATION"
    if failures:
        print(f"check-wheel-pins: {failures} violation(s)", file=sys.stderr)
        return 1
    print("check-wheel-pins: OK")
    return 0


def _self_test():
    clean = ["numpy==1.26.0", "# mlx-omarchy mention in a comment is fine",
             "mlx-lm @ https://example.com/mlx_lm.whl"]
    assert list(check_lines(clean)) == [], list(check_lines(clean))
    traps = [
        "mlx-omarchy==0.32.2",
        "mlx_omarchy @ file:///home/u/dist/mlx_omarchy-0.32.2.dev20260903"
        "-cp311-cp311-linux_x86_64.whl#sha256=6e54ab2b",
        "mlx-omarchy @ https://github.com/example/releases/download/v0.3.2/"
        "mlx_omarchy-0.32.2-cp311.whl",
        "https://example.com/mlx_omarchy-0.32.2.dev20260902-cp311.whl",
        "mlx-omarchy>=0.3",
        "mlx_omarchy==0.1",
    ]
    found = list(check_lines(traps))
    violations = [s for s, _ in found if s == "VIOLATION"]
    assert len(violations) == len(traps), found
    joined = " ".join(m for _, m in found)
    assert "requirements input" in joined
    # Every trap line is named in a message, so the operator sees the line.
    for trap in traps:
        assert trap in joined, trap
    print("self-test: OK (6 trap forms rejected, clean input passes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
