#!/usr/bin/env bash
# upstream-compat-check.sh - skeleton upstream-drift checker for mlx-omarchy.
#
# Modes:
#   api           compare mlx.lock pin vs upstream main/latest tag; classify
#                 changed files; silent (exit 0) unless drift is actionable.
#   probe COMMIT  download the candidate tarball and dry-run the patch series
#                 exactly as scripts/prepare-mlx.sh applies it (order + fuzz),
#                 plus the overlay-collision check. exit 0 = applies clean.
#   all           api, then probe the pinned commit.
#
# Exit codes: 0 = quiet / clean, 1 = drift detected, 2 = probe or tooling error.
# Never writes inside the repo; reports go to --out (default /tmp).
#
# Scheduling: run `api` weekly from GitHub Actions cron. Do NOT schedule probe
# or builds on the M1/M2 hosts; the deep gate (prepare-mlx.sh into a temp
# MLX_OMARCHY_WORK_DIR + tools/run-upstream-suite.sh --cpp-only) stays manual,
# run only when this checker reports a bump is worth considering. mlx.lock is
# never modified by this script.
set -euo pipefail

REPO="${REPO:-$PWD}"
OUT=""
VERBOSE=0
MODE="api"
CANDIDATE=""
GH="https://api.github.com/repos/ml-explore/mlx"
UP="https://codeload.github.com/ml-explore/mlx/tar.gz"

die() { echo "ERROR: $*" >&2; exit 2; }
need() { command -v "$1" >/dev/null || die "missing dependency: $1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    api|probe|all) MODE="$1"; shift ;;
    --repo) REPO="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    *) CANDIDATE="$1"; shift ;;
  esac
done
need curl; need python3; need patch; need tar; need sha256sum

[[ -f "$REPO/mlx.lock" ]] || die "no mlx.lock under $REPO"
# shellcheck source=/dev/null
source "$REPO/mlx.lock"
[[ -n "${MLX_COMMIT:-}" ]] || die "mlx.lock does not define MLX_COMMIT"

# Patch series in prepare-mlx.sh's exact order, with each patch's fuzz factor.
read_patches() {
  awk '
    { while (match($0, /--fuzz=[0-9]+/)) { fuzz=substr($0, RSTART+7, RLENGTH-7);
        $0=substr($0, RSTART+RLENGTH) }
      if (match($0, /patches\/[A-Za-z0-9._-]+\.patch/))
        print substr($0, RSTART+8, RLENGTH-8), (fuzz=="" ? 0 : fuzz) }
  ' "$REPO/scripts/prepare-mlx.sh"
}

# Files our patch series touches (mlx upstream paths only; mlx-lm patches
# target venv paths and simply never match upstream files).
patch_targets() {
  read_patches | while read -r name _; do
    grep -h '^+++ b/' "$REPO/patches/$name" | sed 's|^+++ b/||'
  done | sort -u
}

gh() { curl -s --max-time 30 -H "Accept: application/vnd.github+json" "$GH$1"; }

# classify <path> -> conflict | backend | info
classify() {
  local p="$1"
  if grep -qxF "$p" <(patch_targets); then echo conflict
  elif [[ "$p" == mlx/backend/* || "$p" == CMakeLists.txt || "$p" == mlx/*.h ]]; then echo backend
  else echo info; fi
}

mode_api() {
  local tip tag ahead files
  tip="$(gh /commits/main | python3 -c 'import json,sys;print(json.load(sys.stdin)["sha"])')"
  tag="$(gh /releases/latest | python3 -c 'import json,sys;print(json.load(sys.stdin).get("tag_name",""))')"
  local cmp; cmp="$(gh "/compare/${MLX_COMMIT}...main")"
  ahead="$(printf '%s' "$cmp" | python3 -c 'import json,sys;print(json.load(sys.stdin)["ahead_by"])')"

  files="$(printf '%s' "$cmp" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for f in d.get("files", []): print(f["filename"])')"

  # Zero-noise gate: silent when up to date or when nothing new is actionable.
  if [[ "$ahead" == "0" ]]; then [[ $VERBOSE == 1 ]] && echo "up to date: $MLX_COMMIT"; return 0; fi
  local interesting=0
  while IFS= read -r f; do [[ -z "$f" ]] && continue
    case "$(classify "$f")" in conflict|backend) interesting=1; break ;; esac
  done <<< "$files"
  if [[ "$tag" != "$MLX_VERSION" && -n "$tag" ]]; then interesting=1; fi
  if [[ $interesting == 0 && $VERBOSE == 0 ]]; then return 0; fi

  local out="${OUT:-/tmp/upstream-drift-$(date +%F).md}"
  {
    echo "# upstream drift: $MLX_VERSION ($MLX_COMMIT) -> main ($tip)"
    echo; echo "- latest release: ${tag:-unknown}; main ahead by $ahead"
    echo; echo "| class | file |"; echo "|---|---|"
    while IFS= read -r f; do [[ -z "$f" ]] && continue
      echo "| $(classify "$f") | $f |"
    done <<< "$files"
    echo; echo "Probe command:"
    echo "  bash $0 probe $tip --repo $REPO"
    echo "Deep gate (manual, Linux host, no GPU needed for --cpp-only):"
    echo "  MLX_OMARCHY_WORK_DIR=$(mktemp -d) scripts/prepare-mlx.sh  # after updating a temp mlx.lock"
    echo "  tools/run-upstream-suite.sh --cpp-only"
  } > "$out"
  echo "$out"
  return 1
}

mode_probe() {
  local commit="${CANDIDATE:?probe needs a commit}"
  PROBE_TMP="$(mktemp -d /tmp/upstream-probe.XXXXXX)"
  tmp="$PROBE_TMP"
  trap 'rm -rf "$tmp"' EXIT
  local tgz="$tmp/mlx-$commit.tar.gz"
  curl -sfL --max-time 300 --retry 3 -o "$tgz" "$UP/$commit" || die "tarball fetch failed for $commit"
  sha256sum "$tgz" | tee "$tmp/tarball.sha256" >&2
  mkdir "$tmp/tree"
  tar -xzf "$tgz" -C "$tmp/tree" --strip-components=1

  # Overlay collision check, same semantics as scripts/prepare-mlx.sh:31-38.
  local collisions=0
  while IFS= read -r f; do
    local rel="${f#"$REPO/overlay/"}"
    if [[ -e "$tmp/tree/$rel" ]]; then echo "OVERLAY COLLISION: $rel" >&2; collisions=1; fi
  done < <(find "$REPO/overlay" -type f)

  # Apply the series sequentially for real on the throwaway tree - later
  # hunks depend on context shifted by earlier patches, so per-patch
  # --dry-run against a pristine tree gives false failures.
  local failed=0 n=0
  while read -r name fuzz; do
    n=$((n+1))
    if ! patch --directory="$tmp/tree" --strip=1 --forward --fuzz="$fuzz" \
        < "$REPO/patches/$name" > "$tmp/$name.apply" 2>&1; then
      echo "PATCH FAIL (fuzz=$fuzz): $name" >&2
      grep -E 'FAILED|hunk' "$tmp/$name.apply" | sed -n '1,6p' >&2
      failed=$((failed+1))
    fi
  done < <(read_patches)

  [[ $collisions == 0 ]] || die "overlay paths already exist upstream"
  if [[ $failed -gt 0 ]]; then
    echo "probe $commit: $failed/$n patches failed" >&2
    exit 2
  fi
  echo "probe $commit: $n/$n patches apply clean" >&2
}

case "$MODE" in
  api)   mode_api ;;
  probe) mode_probe ;;
  all)   mode_api || true; CANDIDATE="$MLX_COMMIT" mode_probe ;;
esac
