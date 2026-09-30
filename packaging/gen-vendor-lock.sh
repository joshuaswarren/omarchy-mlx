#!/usr/bin/env bash
# Write a hash-locked requirements file from a directory of wheels.
#
#   packaging/gen-vendor-lock.sh <wheel-dir> > requirements-lock.txt
#
# Purely local: hashes what is there, resolves nothing. The release-time
# counterpart that downloads the wheels is vendor-wheels.sh.
set -euo pipefail

vendor="${1:?usage: gen-vendor-lock.sh <wheel-dir>}"
if [[ ! -d $vendor ]]; then
  echo "error: $vendor is not a directory" >&2
  exit 1
fi

shopt -s nullglob
wheels=("$vendor"/*.whl)
if (( ${#wheels[@]} == 0 )); then
  echo "error: no wheels in $vendor" >&2
  exit 1
fi

for wheel in "${wheels[@]}"; do
  base="$(basename "$wheel" .whl)" # name-version-tags...
  name="${base%%-*}"
  rest="${base#*-}"
  version="${rest%%-*}" # PEP 440 versions carry no dash
  norm="$(printf '%s' "$name" | sed 's/[-_.]\+/-/g' | tr 'A-Z' 'a-z')"
  hash="$(sha256sum "$wheel" | cut -d' ' -f1)"
  printf '%s==%s --hash=sha256:%s\n' "$norm" "$version" "$hash"
done | sort
