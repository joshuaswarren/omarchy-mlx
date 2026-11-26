#!/usr/bin/env bash
# Whole-encoder bundle gate for the wheel build.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/stage-whole-bundle.sh"
#   stage_whole_bundle "$WORK_DIR"
#
# Sourced by scripts/build-wheel.sh; exits the shell on refusal (the same
# contract as the previous inline gate: a refusing build dies).
#
# Contract: when the runtime pin declares parakeet-encoder-whole, the ONLY
# acceptable build carries the exact pinned bundle. A wheel without it
# would not silently fall back - the runtime refuses to run the encoder at
# all (see overlay/tools/coreml/vulkan_encoder.py) - so there is no opt-out:
# unset MLX_OMARCHY_WHOLE_BUNDLE_DIR refuses, and so does a wrong-hash
# bundle. When the pin does not declare the bundle, nothing is staged.

stage_whole_bundle() {
  local work_dir="$1"
  local pin_path="$work_dir/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/parakeet-runtime-pin.json"
  local bundle_staging="$work_dir/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole"
  if [[ -f "$pin_path" ]] && python3 -c "
import json, sys
with open('$pin_path') as fh:
    pin = json.load(fh)
sys.exit(0 if 'parakeet-encoder-whole' in pin.get('assets', {}).get('bundles', {}) else 1)
"; then
    if [[ -z "${MLX_OMARCHY_WHOLE_BUNDLE_DIR:-}" ]]; then
      echo "[bundle] runtime pin declares parakeet-encoder-whole but MLX_OMARCHY_WHOLE_BUNDLE_DIR is unset; refusing to build a wheel that would silently fall back" >&2
      echo "[bundle] supply the bundle dir (manifest.json + program-0.anec) via MLX_OMARCHY_WHOLE_BUNDLE_DIR=/path/to/dir" >&2
      exit 1
    fi
    if [[ ! -f "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/manifest.json" || ! -f "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/program-0.anec" ]]; then
      echo "[bundle] ${MLX_OMARCHY_WHOLE_BUNDLE_DIR} does not contain manifest.json + program-0.anec" >&2
      exit 1
    fi
    local expected_manifest expected_program actual_manifest actual_program
    expected_manifest="$(python3 -c "
import json
with open('$pin_path') as fh:
    print(json.load(fh)['assets']['bundles']['parakeet-encoder-whole']['manifest.json'])
")"
    expected_program="$(python3 -c "
import json
with open('$pin_path') as fh:
    print(json.load(fh)['assets']['bundles']['parakeet-encoder-whole']['program-0.anec'])
")"
    actual_manifest="$(sha256sum "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/manifest.json" | cut -d' ' -f1)"
    actual_program="$(sha256sum "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/program-0.anec" | cut -d' ' -f1)"
    [[ "$actual_manifest" == "$expected_manifest" ]] || {
      echo "[bundle] manifest.json sha mismatch: pin=$expected_manifest actual=$actual_manifest" >&2; exit 1; }
    [[ "$actual_program" == "$expected_program" ]] || {
      echo "[bundle] program-0.anec sha mismatch: pin=$expected_program actual=$actual_program" >&2; exit 1; }
    rm -rf "$bundle_staging"
    mkdir -p "$bundle_staging"
    install -m 0644 "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/manifest.json" "$bundle_staging/manifest.json"
    install -m 0644 "${MLX_OMARCHY_WHOLE_BUNDLE_DIR}/program-0.anec" "$bundle_staging/program-0.anec"
    # Refresh mtime so the CMake install(DIRECTORY) rule treats the
    # bundle as newer than any prior build output (the same rule
    # prepare-mlx.sh applies to the overlay).
    touch "$bundle_staging/manifest.json" "$bundle_staging/program-0.anec"
    echo "[bundle] staged $bundle_staging (manifest $actual_manifest, program $actual_program)"
  else
    echo "[bundle] runtime pin does not declare parakeet-encoder-whole; skipping whole-bundle stage"
  fi
}
