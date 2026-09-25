#!/usr/bin/env python3
"""privacy-check: fail-closed per-commit private-content guard (v5).

Generic patterns only; site inventory via optional untracked
.private-patterns (never published). Reviewed-content allowlist via
tracked .privacy-allow (exact "<blob-sha> <path>" pins; content scan
only — messages and filenames are never exempted). Scans EVERY commit in range
(merges via -m, root via --root): added lines, new filenames, commit
message. Single rev = full history (conservative for new refs).

Usage:
  privacy_check.py <base>..<head>     scan introduced commits
  privacy_check.py <head>             scan full history (new refs)
  privacy_check.py --hook             pre-push hook mode (reads stdin)
  privacy_check.py --staged           pre-commit mode (scan staged added lines)

Exit: 0 clean, 1 blocked/failure (fail-closed).
"""
import os
import re
import subprocess
import sys
import tempfile

ZERO_BYTES = b"0" * 40
ZERO = "0" * 40

GENERIC_PATTERNS = [
    rb"/home/[a-z0-9_.-]+/",
    rb"/Users/[A-Za-z0-9_.-]+/",
    rb"100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}",
    rb"10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",
    rb"172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}",
    rb"192\.168\.[0-9]{1,3}\.[0-9]{1,3}",
]


def die(msg):
    sys.stderr.write("privacy-check: %s\n" % msg)
    sys.exit(1)


def run(args, input_bytes=None):
    p = subprocess.run(args, input=input_bytes, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    if p.returncode != 0:
        die("git failed: %s (%s)" % (args[0:3], p.stderr.decode("utf-8", "replace").strip()))
    return p.stdout


def compile_patterns(repo):
    pats = list(GENERIC_PATTERNS)
    site = os.path.join(repo, os.environ.get("PRIVACY_PATTERNS_NAME", ".private-patterns"))
    if os.path.isfile(site):
        with open(site, "rb") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if line and not line.startswith(b"#"):
                    pats.append(line)
    return [re.compile(p, re.IGNORECASE) for p in pats]


def compile_allows(repo):
    """(blob_sha, path) pairs exempt from the blob-content scan.

    Tracked .privacy-allow pins reviewed content the generic patterns
    flag on synthetic fixture data (e.g. a PII scanner's own unit-test
    literals). Lines: "<40-hex-sha> <path>", "#" comments. A pin is
    exact: any edit to the file makes a new blob and blocks again
    until re-pinned, so the exemption cannot silently cover new
    content. Message and filename scans are never exempted.
    """
    allows = set()
    site = os.path.join(repo, ".privacy-allow")
    if os.path.isfile(site):
        with open(site, "rb") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                sha, _, path = line.partition(b" ")
                if sha and path:
                    allows.add((sha.lower(), path.strip()))
    return allows


def scan(content, label, regexes, hits):
    for rx in regexes:
        n = len(rx.findall(content))
        if n:
            sys.stderr.write("privacy-check: BLOCKED %s pattern %r (%d line(s))\n"
                             % (label, rx.pattern.decode("utf-8", "replace"), n))
            hits.append(label)


def changed_new_blobs(repo, c):
    """(path, new_blob_sha_or_None, commit) for added/modified paths.
    --no-renames forces renames to A+D pairs; -z keeps raw paths intact
    (spaces/newlines preserved); --root -m covers root and merge commits."""
    raw = run(["git", "-C", repo, "diff-tree", "--root", "-m", "-r",
               "--no-renames", "--no-commit-id", "--name-status", "-z", c])
    recs = raw.split(b"\0")
    out = []
    i = 0
    while i < len(recs) - 1:
        meta = recs[i]
        path = recs[i + 1]
        i += 2
        if not meta:
            continue
        fields = meta.split(b" ")
        status = (fields[4] if len(fields) >= 5 else meta)[:1]
        if status in (b"A", b"M", b"T"):
            out.append((path, None, c))
    return out


def main():
    args = sys.argv[1:]
    if not args:
        die("usage: <base>..<head> | <head> | --hook")

    if args[0] == "--staged":
        repo = os.environ.get("PRIVACY_REPO")
        if not repo or not os.path.isdir(repo):
            die("PRIVACY_REPO not set or invalid (fail-closed)")
        os.chdir(repo)
        regexes = compile_patterns(repo)
        raw = run(["git", "diff", "--cached", "-U0"])
        added = b"\n".join(l[1:] for l in raw.split(b"\n")
                           if l.startswith(b"+") and not l.startswith(b"+++"))
        hits = []
        scan(added, "staged content", regexes, hits)
        if hits:
            sys.exit(1)
        print("privacy-check: staged content clean")
        sys.exit(0)

    if args[0] == "--hook":
        stdin = sys.stdin.buffer.read().decode("utf-8", "replace")
        toplevel = run(["git", "rev-parse", "--show-toplevel"],
                       b"").decode().strip()
        os.environ["PRIVACY_REPO"] = toplevel
        blocked = False
        for line in stdin.splitlines():
            parts = line.split()
            if len(parts) != 4:
                die("malformed pre-push stdin line: %r" % line[:120])
            local_ref, local_sha, remote_ref, remote_sha = parts
            if local_sha == ZERO:
                continue  # deletion
            if remote_sha == ZERO:
                ranges = [local_sha]  # new ref: full history, conservative
            else:
                ranges = ["%s..%s" % (remote_sha, local_sha)]
            for rng in ranges:
                p = subprocess.run([sys.executable, os.path.abspath(__file__),
                                    rng],
                                   env={**os.environ,
                                        "PRIVACY_REPO": toplevel},
                                   stdout=subprocess.DEVNULL)
                if p.returncode != 0:
                    blocked = True
        sys.exit(1 if blocked else 0)

    repo = os.environ.get("PRIVACY_REPO")
    if not repo:
        die("PRIVACY_REPO not set (fail-closed)")
    if not os.path.isdir(repo):
        die("PRIVACY_REPO '%s' is not a directory" % repo)
    rng = args[0]
    base_part, sep, head_part = rng.partition("..")
    if sep:
        if not base_part or not head_part:
            die("empty range side: %s" % rng)
        for part in (base_part, head_part):
            if subprocess.run(["git", "-C", repo, "rev-parse", "-q",
                               "--verify", part],
                              stdout=subprocess.DEVNULL).returncode != 0:
                die("bad range: %s" % rng)
    else:
        head_part = rng
        if subprocess.run(["git", "-C", repo, "rev-parse", "-q", "--verify",
                           head_part], stdout=subprocess.DEVNULL).returncode != 0:
            die("bad rev: %s" % rng)

    regexes = compile_patterns(repo)
    allows = compile_allows(repo)
    if os.environ.get("PRIVACY_DEBUG"):
        sys.stderr.write("privacy-check DEBUG: repo=%s range=%s patterns=%d site=%s\n" % (
            repo, rng, len(regexes),
            os.path.isfile(os.path.join(repo, os.environ.get("PRIVACY_PATTERNS_NAME", ".private-patterns")))))
    commits = run(["git", "-C", repo, "rev-list", rng]).decode().split()
    hits = []
    for c in commits:
        msg = run(["git", "-C", repo, "log", "-1", "--format=%B", c])
        scan(msg, "message %s" % c[:9], regexes, hits)
        for path, newsha, cc in changed_new_blobs(repo, c):
            scan(path, "new filename %s" % cc[:9], regexes, hits)
            if newsha is None:
                newsha = run(["git", "-C", repo, "rev-parse",
                              cc + ":" + path.decode("utf-8", "surrogateescape")]).strip()
            if (newsha.lower(), path) in allows:
                continue
            body = run(["git", "-C", repo, "cat-file", "blob",
                        newsha.decode("ascii")])
            scan(body, "blob %s" % path.decode("utf-8", "replace")[:60],
                 regexes, hits)
    if hits:
        sys.exit(1)
    print("privacy-check: clean (%s)" % rng)


if __name__ == "__main__":
    main()
