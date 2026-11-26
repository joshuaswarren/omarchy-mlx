#!/usr/bin/env bash
# Apply the TensorFold platform-gate patch series to a clean clone of
# ashhart/TensorFold at the pinned commit (609ca419abecebdc5a059498a613680bd3aa847f,
# tag v0.6.5, identical to main at audit time).
#
#   packaging/tensorfold-linux/apply-platform-gate.sh SRC_DIR [--verify-only]
#
# Idempotent; same semantics as the omlx installer (see
# packaging/omlx-linux/apply-platform-gate.sh).
#
# License: Apache-2.0 (preserved; see LICENSE/patches/tensorfold/LICENSE).
set -euo pipefail

self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patches_dir="$self_dir/patches"

if (($# < 1)); then
    echo "usage: $0 SRC_DIR [--verify-only]" >&2
    exit 1
fi

src="$1"
verify_only=0
if [[ "${2:-}" == "--verify-only" ]]; then
    verify_only=1
fi

if [[ ! -d $src/.git ]]; then
    echo "error: $src is not a git working tree" >&2
    exit 1
fi

PINNED_HEAD="609ca419abecebdc5a059498a613680bd3aa847f"
actual="$(git -C "$src" rev-parse HEAD)"
if [[ "$actual" != "$PINNED_HEAD" ]]; then
    echo "error: $src HEAD is $actual; expected $PINNED_HEAD (ashhart/TensorFold v0.6.5)" >&2
    exit 1
fi

if [[ -f $src/src/tensorfold/_compat_gate.py ]]; then
    echo "error: $src/src/tensorfold/_compat_gate.py already exists; patches applied." >&2
    exit 1
fi

for p in "$patches_dir"/0?-*.patch; do
    [[ -e $p ]] || { echo "error: no patches found in $patches_dir" >&2; exit 1; }
    echo "==> applying $(basename "$p")"
    if (( verify_only )); then
        git -C "$src" apply --check "$p"
    else
        git -C "$src" apply "$p"
    fi
done

if (( verify_only )); then
    echo "==> all patches would apply cleanly to $src"
else
    echo "==> applied $(ls "$patches_dir"/0?-*.patch | wc -l) patches to $src"
    echo "    new file: src/tensorfold/_compat_gate.py"
    echo "    edited files:"
    git -C "$src" diff --stat
fi