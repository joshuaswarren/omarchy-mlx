#!/usr/bin/env python3
"""Redact M2-specific hostnames / home paths / IPs from a log before
committing. Generic-replace; deterministic."""
import re, sys, pathlib

# M2 host markers (the alias the privacy hook already accepts in committed
# receipt filenames, e.g. g16-qmm-batch-legP-jwm1.json on main; jwm2 is the
# M2 host).
REPLACEMENTS = [
    (re.compile(r"/home/joshuawarren"), "$HOME"),
    (re.compile(r"/Users/joshuawarren"), "$HOME"),
    (re.compile(r"joshuaswarren@users\.noreply\.github\.com"), "<author>"),
    (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<ip>"),
    # 192.168.* / 100.* tailscale / etc are already covered by the IP regex.
    (re.compile(r"jwm2-linux"), "<host>"),
    (re.compile(r"jw14m2-linux"), "<host>"),
    (re.compile(r"jwm1\b"), "<other-host>"),
    (re.compile(r"jwm1-linux"), "<other-host>"),
]

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gateM2-redact.py <file>", file=sys.stderr)
        return 2
    p = pathlib.Path(sys.argv[1])
    s = p.read_text()
    for pat, sub in REPLACEMENTS:
        s = pat.sub(sub, s)
    p.write_text(s)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
