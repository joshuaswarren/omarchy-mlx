#!/usr/bin/env bash
# Stub harness for g-hold-g13g.sh: runs the REAL script end to end on this
# box (CPU only) with stubbed python3.14/sudo/udevadm/stat and a fake vendor
# tar, under set -euo pipefail. Cases:
#   1 positive:      2/2 live tests + 4 equal digests -> HOLD_PASS, rc 0
#   2 skipped tests: FAILED (skipped=2)             -> FAIL, rc 1
#   3 vendor mlx:    vendor tar carries upstream mlx -> FAIL, rc 1
# Usage: bash stub-tests/test-g-hold.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/../g-hold-g13g.sh"
TESTS_FILE="$HERE/../../../../tests/test_cpu_pd_hold.py"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/hold-stub.XXXX")"
FAILED=0

make_vendor_tar() { # $1 = dest tar, $2 = include upstream mlx? (0/1)
  local top="$WORK/vendor-src/omarchy-mlx-vendor-wheels-v0.7.29-cp314-aarch64"
  rm -rf "$WORK/vendor-src"
  mkdir -p "$top"
  for w in mlx_lm-0.31.3-py3-none-any.whl transformers-5.16.1-py3-none-any.whl \
           numpy-2.5.3-cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl \
           tokenizers-0.23.2-cp310-abi3-manylinux2014_aarch64.whl; do
    echo stub >"$top/$w"
  done
  if [[ "$2" == 1 ]]; then echo stub >"$top/mlx-0.29.0-cp314-cp314-linux_aarch64.whl"; fi
  tar -cf "$1" -C "$WORK/vendor-src" omarchy-mlx-vendor-wheels-v0.7.29-cp314-aarch64
}

make_python_stub() { # $1 = dest
  cat >"$1" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  -m) case "$2" in
        venv) mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; chmod +x "$3/bin/python"; exit 0 ;;
        pip) exit 0 ;;
      esac ;;
  -c) echo "mlx_lm 0.31.3"; exit 0 ;;
  -) echo "MLX_PIN version=0.32.4.dev202610070139+d86daf9 expected=0.32.4.dev202610070139+d86daf9 ok=True"
     echo "MLX_PIN libmlx_sha=deadbeef record_sha=deadbeef ok=True"; exit 0 ;;
esac
tests_mode=0; ab_mode=0
for a in "$@"; do
  case "$a" in
    *test_cpu_pd_hold.py) tests_mode=1 ;;
    *ab.py) ab_mode=1 ;;
  esac
done
if (( tests_mode )); then
    echo "test_forced (ok) ... ok"
    echo "test_off (ok) ... ok"
    echo "Ran 2 tests in 0.001s"
    if [[ "${STUB_TESTS_MODE:-}" == "skipped" ]]; then
      echo "FAILED (skipped=2)"
    else
      echo "OK"
    fi
    exit 0
fi
if (( ab_mode )); then
    printf 'AB_DIGEST %s ntok 64\n' "$(printf stub | sha256sum | awk '{print $1}')"
    exit 0
fi
exit 0
STUB
  chmod +x "$1"
}

make_fakebin() { # $1 = case dir, $2 = tests mode
  local d="$1/fakebin"
  mkdir -p "$d"
  for t in mktemp tee grep timeout flock date cat sha256sum tar awk install; do
    ln -sf "$(command -v "$t")" "$d/$t"
  done
  printf '#!/usr/bin/env bash\ncase "$1" in install|rm) exit 0 ;; *) exec "$@" ;; esac\n' >"$d/sudo"
  chmod +x "$d/sudo"
  printf '#!/usr/bin/env bash\ncase "$*" in *cpu_dma_latency*) echo "root video 660" ;; *) exec /usr/bin/stat "$@" ;; esac\n' >"$d/stat"
  chmod +x "$d/stat"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$d/udevadm"; chmod +x "$d/udevadm"
  STUB_TESTS_MODE="$2"
  make_python_stub "$d/python3.14"
  export STUB_TESTS_MODE
}

run_case() { # $1 case, $2 tests mode, $3 vendor mlx
  local name="$1"
  local case_dir="$WORK/$name"
  make_fakebin "$case_dir" "$2"
  make_vendor_tar "$case_dir/vendor.tar" "$3"
  mkdir -p "$case_dir"
  cp "$GATE" "$case_dir/g-hold-g13g.sh"
  cp "$TESTS_FILE" "$case_dir/test_cpu_pd_hold.py"
  (
    export PATH="$case_dir/fakebin:$PATH"
    export HOLD_WHEEL="/tmp/stub/mlx_omarchy-0.32.4.dev202610070139+d86daf9-cp314-cp314-linux_aarch64.whl"
    export HOLD_VENDOR_TAR="$case_dir/vendor.tar"
    export HOLD_RULE_SRC="$case_dir/70-omarchy-mlx-cpu-dma-latency.rules"
    export HOLD_RECEIPT="$case_dir/receipt.log"
    export HOLD_MODEL="/tmp/stub/model"
    echo stub >"$HOLD_RULE_SRC"
    echo stub >"$HOLD_WHEEL"
    mkdir -p "$HOLD_MODEL"
    set -euo pipefail
    bash "$case_dir/g-hold-g13g.sh"
  ) >"$case_dir/out.log" 2>&1
  echo $? >"$case_dir/rc"
}

check() { if (( $2 == 0 )); then echo "ok: $1"; else echo "FAIL: $1"; FAILED=1; fi; }

run_case positive "" 0
rc=$(cat "$WORK/positive/rc")
check "positive rc=0" $([[ $rc == 0 ]]; echo $?)
grep -q "HOLD_UDEV root video 660" "$WORK/positive/receipt.log"; check "positive udev line" $?
grep -q "Ran 2 tests" "$WORK/positive/receipt.log"; check "positive Ran 2 tests in receipt" $?
grep -qE "^OK$" "$WORK/positive/receipt.log"; check "positive bare OK" $?
check "positive 4 AB_DIGEST lines" $([[ $(grep -c "AB_DIGEST" "$WORK/positive/receipt.log") == 4 ]]; echo $?)
grep -q "HOLD_AB all 4 digests equal" "$WORK/positive/receipt.log"; check "positive digest rule" $?
grep -q "MLX_PIN libmlx_sha=deadbeef record_sha=deadbeef ok=True" "$WORK/positive/receipt.log"; check "positive mlx pin assert" $?
grep -q "HOLD_PASS" "$WORK/positive/out.log"; check "positive PASS" $?

run_case skipped skipped 0
rc=$(cat "$WORK/skipped/rc")
check "skipped rc=1" $([[ $rc == 1 ]]; echo $?)
grep -q "FAIL" "$WORK/skipped/out.log"; check "skipped FAIL message" $?

run_case vendormlx "" 1
rc=$(cat "$WORK/vendormlx/rc")
check "vendor-mlx rc=1" $([[ $rc == 1 ]]; echo $?)
grep -q "vendor tar carries an upstream mlx wheel" "$WORK/vendormlx/out.log"; check "vendor-mlx FAIL message" $?

echo "== raw outputs =="
for c in positive skipped vendormlx; do echo "--- $c (rc=$(cat "$WORK/$c/rc"))"; tail -6 "$WORK/$c/out.log"; done
if (( FAILED )); then echo "STUB_HARNESS g-hold: FAIL"; exit 1; fi
echo "STUB_HARNESS g-hold: PASS"
rm -rf "$WORK"
