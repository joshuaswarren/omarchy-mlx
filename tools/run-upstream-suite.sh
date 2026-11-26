#!/usr/bin/env bash
# Run pinned upstream suites once per file, retaining failures and raw reports.
# Explicit upstream CPU cases remain CPU cases; this is not GPU-only coverage.
# Usage: tools/run-upstream-suite.sh [--cpp-only | --py-only]
# OUT_DIR, CPP_TIMEOUT, PY_TIMEOUT, WHEEL, PYTEST_VENV, MLX_OMARCHY_WORK_DIR override defaults.
# Development hosts must explicitly set MLX_OMARCHY_ALLOW_NON_APPLE=1.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
OUT_DIR="${OUT_DIR:-$ROOT/receipts/upstream-suite-$(date +%F-%H%M%S)}"
CPP_TIMEOUT="${CPP_TIMEOUT:-900}"
PY_TIMEOUT="${PY_TIMEOUT:-900}"

RUN_CPP=1
RUN_PY=1
for arg in "$@"; do
  case "$arg" in
    --cpp-only) RUN_PY=0 ;;
    --py-only) RUN_CPP=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done
if [[ $RUN_CPP -eq 0 && $RUN_PY -eq 0 ]]; then
  echo "--cpp-only and --py-only are mutually exclusive" >&2
  exit 2
fi
SOURCE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)" || exit 1
git -C "$ROOT" diff --quiet HEAD -- overlay patches scripts mlx.lock || {
  echo "Commit source changes before qualification." >&2
  exit 1
}
mkdir -p "$(dirname "$OUT_DIR")" || exit 1
mkdir "$OUT_DIR" || exit 1
printf '%s\n' "$SOURCE_COMMIT" > "$OUT_DIR/source-commit.txt" || exit 1
MLX_OMARCHY_WORK_DIR="$WORK_DIR" "$ROOT/scripts/prepare-mlx.sh" > "$OUT_DIR/prepare.log" 2>&1 || exit 1
cp "$ROOT/mlx.lock" "$OUT_DIR/mlx.lock" || exit 1
export DEVICE=gpu
fail_total=0

if [[ $RUN_CPP -eq 1 ]]; then
  BUILD_DIR="$OUT_DIR/build-cpp"
  DOCTEST_PIN_ARGS=()
  if [[ -d "$WORK_DIR/build/_deps/doctest-src" ]]; then
    DOCTEST_PIN_ARGS+=(-DFETCHCONTENT_SOURCE_DIR_DOCTEST="$WORK_DIR/build/_deps/doctest-src")
  fi
  cmake -S "$WORK_DIR/mlx" -B "$BUILD_DIR" -G Ninja \
    -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON \
    -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
    -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
    "${DOCTEST_PIN_ARGS[@]}" || exit 1
  cmake --build "$BUILD_DIR" --target tests -j "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}" || exit 1
  TESTS_BIN="$BUILD_DIR/tests/tests"
  sha256sum "$TESTS_BIN" > "$OUT_DIR/cpp-binary.sha256" || exit 1
  mkdir -p "$OUT_DIR/cpp" || exit 1
  cpp_files="allocator_tests.cpp array_tests.cpp arg_reduce_tests.cpp \
autograd_tests.cpp blas_tests.cpp compile_tests.cpp custom_vjp_tests.cpp \
creations_tests.cpp device_tests.cpp einsum_tests.cpp export_import_tests.cpp \
eval_tests.cpp fft_tests.cpp load_tests.cpp linalg_tests.cpp ops_tests.cpp \
random_tests.cpp scheduler_tests.cpp utils_tests.cpp vmap_tests.cpp"
  printf 'file\trc\tresult\n' > "$OUT_DIR/cpp/summary.tsv" || exit 1
  for f in $cpp_files; do
    base="${f%.cpp}"
    log="$OUT_DIR/cpp/$base.log"
    xml="$OUT_DIR/cpp/$base.xml"
    rm -f "$xml" || exit 1
    echo "== [cpp] $f =="
    timeout "$CPP_TIMEOUT" "$TESTS_BIN" \
      -sf="*$f" -r=xml -o="$xml" -d -nc > "$log" 2>&1
    rc=$?
    parsed="$(python3 - "$xml" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
    cases = root.findall(".//TestCase")
    executed = [case for case in cases if case.get("skipped") != "true"]
    failed = []
    for case in executed:
        result = case.find("OverallResultsAsserts")
        if result is None or result.get("test_case_success") != "true":
            failed.append(case.get("name", "?"))
    print(f"executed={len(executed)} passed={len(executed) - len(failed)} "
          f"failed={len(failed)} skipped={len(cases) - len(executed)} "
          f"cases={','.join(failed)}")
    sys.exit(0 if executed and not failed else 1)
except (OSError, ET.ParseError, ValueError) as exc:
    print(f"INVALID-REPORT: {exc}")
    sys.exit(1)
PYEOF
)"
    parse_rc=$?
    printf '%s\t%s\t%s\n' "$base" "$rc" "$parsed" | tee -a "$OUT_DIR/cpp/summary.tsv" || exit 1
    if [[ $rc -ne 0 || $parse_rc -ne 0 ]]; then fail_total=1; fi
    if [[ $rc -eq 124 || $rc -gt 128 ]]; then
      echo "Stopping after timeout or signal; preserve device state before further tests." >&2
      exit 1
    fi
  done
fi

if [[ $RUN_PY -eq 1 ]]; then
  if [[ -z "${WHEEL:-}" ]]; then
    shopt -s nullglob
    wheels=("$ROOT"/dist/mlx_omarchy-*.whl)
    shopt -u nullglob
    if [[ ${#wheels[@]} -ne 1 ]]; then
      echo "Set WHEEL explicitly unless dist contains exactly one wheel." >&2
      exit 1
    fi
    WHEEL="${wheels[0]}"
  fi
  [[ -f "$WHEEL" ]] || { echo "Wheel not found: $WHEEL" >&2; exit 1; }
  VENV_DIR="${PYTEST_VENV:-$OUT_DIR/venv-pytest}"
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3 -m venv "$VENV_DIR" || exit 1
  fi
  "$VENV_DIR/bin/python" -m pip install numpy pytest || exit 1
  "$VENV_DIR/bin/python" -m pip install --force-reinstall --no-deps "$WHEEL" || exit 1
  "$VENV_DIR/bin/python" "$ROOT/scripts/mlx_provenance.py" --expect-wheel "$WHEEL" > "$OUT_DIR/python-provenance.json" || exit 1
  "$VENV_DIR/bin/python" - "$OUT_DIR/python-provenance.json" "$SOURCE_COMMIT" <<'PYPIN' || exit 1
import json
import sys
with open(sys.argv[1]) as source:
    provenance = json.load(source)
stamp = (provenance.get("dist_version") or "").partition("+")[2].split(".")[-1]
# git's default abbreviation length grows with the repository, so the stamp
# is a prefix of at least seven characters, not a fixed width.
if (provenance.get("verified") != "match" or len(stamp) < 7
        or not sys.argv[2].startswith(stamp)):
    raise SystemExit(f"Unqualified wheel provenance: expected source {sys.argv[2]}, got {stamp!r}")
print(f"SOURCE-VERIFIED {sys.argv[2]} wheel={provenance['dist_version']}")
PYPIN
  mkdir -p "$OUT_DIR/py" || exit 1
  shopt -s nullglob
  py_files=("$WORK_DIR/mlx/python/tests"/test_*.py)
  shopt -u nullglob
  [[ ${#py_files[@]} -gt 0 ]] || { echo "No upstream Python test files found." >&2; exit 1; }
  printf 'file\trc\tresult\n' > "$OUT_DIR/py/summary.tsv" || exit 1
  for f in "${py_files[@]}"; do
    base="$(basename "$f")"
    if [[ "$base" == *distributed* ]]; then
      printf '%s\t1\tNOT-RUN: requires a separately qualified distributed environment\n' "$base" | tee -a "$OUT_DIR/py/summary.tsv" || exit 1
      fail_total=1
      continue
    fi
    log="$OUT_DIR/py/$base.log"
    junit="$OUT_DIR/py/$base.xml"
    rm -f "$junit" || exit 1
    echo "== [py] $base =="
    MLX_ENABLE_TF32=0 timeout "$PY_TIMEOUT" "$VENV_DIR/bin/python" -m pytest "$f" \
      -v --no-header -p no:cacheprovider --junitxml="$junit" > "$log" 2>&1
    rc=$?
    parsed="$(python3 - "$junit" <<'PYEOF'
import sys
import xml.etree.ElementTree as ET
try:
    root = ET.parse(sys.argv[1]).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        raise ValueError("no testsuite elements")
    counts = [sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")]
    n, failures, errors, skipped = counts
    if min(counts) < 0 or failures + errors + skipped > n:
        raise ValueError("inconsistent test counts")
    print(f"executed={n - skipped} passed={n - skipped - failures - errors} "
          f"failed={failures + errors} skipped={skipped}")
    sys.exit(0 if n > skipped and failures + errors == 0 else 1)
except (OSError, ET.ParseError, ValueError) as exc:
    print(f"INVALID-REPORT: {exc}")
    sys.exit(1)
PYEOF
)"
    parse_rc=$?
    printf '%s\t%s\t%s\n' "$base" "$rc" "$parsed" | tee -a "$OUT_DIR/py/summary.tsv" || exit 1
    if [[ $rc -ne 0 || $parse_rc -ne 0 ]]; then fail_total=1; fi
    if [[ $rc -eq 124 || $rc -gt 128 ]]; then
      echo "Stopping after timeout or signal; preserve device state before further tests." >&2
      exit 1
    fi
  done
fi
exit "$fail_total"
