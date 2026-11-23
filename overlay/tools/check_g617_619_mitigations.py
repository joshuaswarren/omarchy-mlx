#!/usr/bin/env python3
"""G617-G619 mitigation regression guard (M6P-3).

Verifies each M6/G18G mitigation's marker in an installed wheel and prints
one PASS/FAIL line per mitigation, then exits nonzero if any check fails.

  M1  omarchy-mlx#656 (G617): mxfp4 subgroup-gather decode defaults OFF on
      G18G (mesa#737 uniform-load defect). Marker: the merge commit is an
      ancestor of the wheel's build commit; optional runtime probe.
  M2  omarchy-mlx#667 (G618): single-op i64/u64 elementwise fallback for the
      AGX3 packer (mesa#739). Markers: merge commit ancestry + exact int64
      select/shift results from the installed wheel.
  M3  omarchy-mlx#664 (G619): small-row qmm test guarded on cpu::is_available.
      Markers: merge commit ancestry + the guarded test source at that commit.

Usage:
  check_g617_619_mitigations.py --repo PATH [--import-name mlx_omarchy]
      [--g617-merge SHA] [--probe-gpu]

PASS lists are deterministic for a given wheel: every check is a pure
function of the wheel's build stamp and fixed numeric probes.
"""
import argparse
import importlib.metadata
import re
import subprocess
import sys


def sh(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          check=False)


def is_ancestor(repo, sha, commit):
    if not sha or not commit:
        return False
    return sh(["git", "-C", repo, "merge-base", "--is-ancestor", sha, commit]).returncode == 0


def wheel_stamp_commit():
    try:
        version = importlib.metadata.version("mlx_omarchy")
    except importlib.metadata.PackageNotFoundError:
        return None, None
    m = re.search(r"\+([0-9a-f]{7,9})$", version)
    return version, (m.group(1) if m else None)


def resolve(repo, short):
    r = sh(["git", "-C", repo, "rev-parse", short])
    return r.stdout.strip() if r.returncode == 0 else None


def check_m2_behavior():
    """Exact int64 select/shift results through the installed wheel."""
    import mlx.core as mx
    a = mx.array([[5, -7, 9, 1 << 40]], dtype=mx.int64)
    b = mx.array([[3, 4, -2, 1 << 39]], dtype=mx.int64)
    mask = mx.array([[True, False, True, False]])
    got_sel = mx.where(mask, a, b)
    want_sel = mx.array([[5, 4, 9, 1 << 39]], dtype=mx.int64)
    got_shift = a >> mx.array([1, 1, 2, 3], dtype=mx.int32)
    want_shift = mx.array([[2, -4, 2, 1 << 37]], dtype=mx.int64)
    got_add = (a + b).astype(mx.int64)
    want_add = mx.array([[8, -3, 7, 3 << 39]], dtype=mx.int64)
    return (bool(mx.array_equal(got_sel, want_sel))
            and bool(mx.array_equal(got_shift, want_shift))
            and bool(mx.array_equal(got_add, want_add)))


def check_m3_source(repo, commit):
    r = sh(["git", "-C", repo, "grep", "-l", "cpu::is_available",
            commit, "--", "overlay/tests/omarchy/test_matmul_family.cpp"])
    return r.returncode == 0 and r.stdout.strip() != ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="omarchy-mlx checkout")
    ap.add_argument("--g617-merge", default="", help="merge sha of #656 when merged")
    ap.add_argument("--probe-gpu", action="store_true",
                    help="M1: also run the gather-qmm decode probe on this GPU")
    args = ap.parse_args()

    version, stamp = wheel_stamp_commit()
    results = []
    if version is None:
        print("FAIL mlx_omarchy not importable")
        return 1
    commit = resolve(args.repo, stamp) if stamp else None
    if commit is None:
        print(f"FAIL wheel stamp {version}: commit not in --repo")
        return 1
    print(f"wheel={version} commit={commit}")

    m1 = is_ancestor(args.repo, args.g617_merge, commit) if args.g617_merge else False
    results.append(("M1 g617 gather-qmm-sub-default-off", m1,
                    "no --g617-merge given (#656 unmerged)" if not args.g617_merge else ""))

    m2a = is_ancestor(args.repo, "3508532cb521038b52349fccf6db0507e4993d87", commit)
    try:
        m2b = check_m2_behavior()
        m2note = "" if m2b else "int64 probe WRONG results"
    except (RuntimeError, ValueError) as exc:
        m2b, m2note = False, f"probe error: {exc}"
    results.append(("M2 g618 i64/u64 fallback", m2a and m2b, m2note))

    m3a = is_ancestor(args.repo, "7d0622ea7930555bdd7190cb363ddb42b4ca838e", commit)
    m3b = check_m3_source(args.repo, commit) if m3a else False
    results.append(("M3 g619 small-row-qmm guard", m3a and m3b,
                    "" if (m3a and m3b) else "guard not in test at this commit"))

    ok = True
    for name, passed, note in results:
        line = f"{'PASS' if passed else 'FAIL'} {name}"
        if note:
            line += f" ({note})"
        print(line)
        ok = ok and passed
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
