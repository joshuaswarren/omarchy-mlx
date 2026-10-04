#!/usr/bin/env bash
# Apply the omlx platform-gate patch series to a clean clone of
# jundot/omlx at the pinned commit (4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40,
# tag v0.7.0).
#
#   packaging/omlx-linux/apply-platform-gate.sh SRC_DIR [--verify-only]
#
# SRC_DIR must be a clean working tree at the pinned commit (see
# receipts/2026-10-04-platform-gate/README.md for the pin verification
# command).  Patches are applied in order with `git apply`; on success the
# repo carries omlx/_compat_gate.py (added) and ~16 gate-site edits.
#
# Idempotent: a second run on an already-patched tree refuses to apply
# the same hunk twice and exits non-zero.  Use `git apply -R` to revert.
#
# License: Apache-2.0 (preserved; see LICENSE/patches/omlx/LICENSE).
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

PINNED_HEAD="4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40"
actual="$(git -C "$src" rev-parse HEAD)"
if [[ "$actual" != "$PINNED_HEAD" ]]; then
    echo "error: $src HEAD is $actual; expected $PINNED_HEAD (jundot/omlx v0.7.0)" >&2
    echo "  pin verification: git -C $src log -1 --format='%H %s'" >&2
    echo "  to re-pin: git -C $src fetch --depth 1 origin v0.7.0 && git -C $src reset --hard $PINNED_HEAD" >&2
    exit 1
fi

# Reject already-patched trees by checking for the helper.
if [[ -f $src/omlx/_compat_gate.py ]]; then
    echo "error: $src/omlx/_compat_gate.py already exists; the patch series has been applied." >&2
    echo "  to re-apply: git -C $src apply -R <this patches dir>" >&2
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
    echo "    new file: omlx/_compat_gate.py"
    echo "    edited files:"
    git -C "$src" diff --stat HEAD~0 2>/dev/null || git -C "$src" diff --stat
fi