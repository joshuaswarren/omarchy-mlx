#!/usr/bin/env bash
# Verify a vendored wheel directory against its hash-locked manifest.
#
#   packaging/verify-vendor.sh <wheel-dir> <requirements-lock.txt>
#
# Every lock line must resolve to exactly one wheel whose sha256 matches
# the pin. Extra wheels in the directory are named (stale vendor bytes).
# Exit 1 on any missing file, hash mismatch, or unpinned lock line.
set -euo pipefail

vendor="${1:?usage: verify-vendor.sh <wheel-dir> <lock>}"
lock="${2:?usage: verify-vendor.sh <wheel-dir> <lock>}"
if [[ ! -d $vendor || ! -f $lock ]]; then
  echo "error: need a wheel directory and a lock file" >&2
  exit 1
fi

status=0
matched=()
while IFS= read -r line; do
  [[ -n $line ]] || continue
  case "$line" in
    '#'*) continue ;;
  esac
  req="${line%% *}"
  rest="${line#* }"
  if [[ $req == "$line" || $rest != --hash=sha256:* ]]; then
    echo "error: lock line is not name==version --hash=sha256:...: $line" >&2
    status=1
    continue
  fi
  want="${rest#--hash=sha256:}"
  name="${req%%==*}"
  version="${req#*==}"
  found=""
  for wheel in "$vendor"/*.whl; do
    [[ -e $wheel ]] || break
    base="$(basename "$wheel" .whl)"
    fname_norm="$(printf '%s' "${base%%-*}" | sed 's/[-_.]\+/-/g' | tr 'A-Z' 'a-z')"
    if [[ "$fname_norm" == "$name" && "${base#*-}" == "$version-"* ]]; then
      found="$wheel"
      break
    fi
  done
  if [[ -z $found ]]; then
    echo "error: $name==$version is pinned but no wheel provides it" >&2
    status=1
    continue
  fi
  got="$(sha256sum "$found" | cut -d' ' -f1)"
  if [[ $got != "$want" ]]; then
    echo "error: hash mismatch for $(basename "$found"): lock $want, file $got" >&2
    status=1
    continue
  fi
  matched+=("$found")
done <"$lock"

for wheel in "$vendor"/*.whl; do
  [[ -e $wheel ]] || break
  listed=""
  for have in "${matched[@]:-}"; do
    if [[ $have == "$wheel" ]]; then
      listed=1
      break
    fi
  done
  if [[ -z $listed ]]; then
    echo "note: $(basename "$wheel") is vendored but not in the lock" >&2
  fi
done

if (( status )); then
  echo "error: vendor directory does not match the lock" >&2
  exit 1
fi
echo "verified ${#matched[@]} vendored wheel(s) against $lock"
