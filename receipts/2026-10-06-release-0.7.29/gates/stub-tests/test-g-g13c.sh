#!/usr/bin/env bash
# Stub harness for g-g13c.sh: runs the REAL script end to end on this box
# with a stubbed python3.14/sudo and a stub mlx-omarchy-info, under
# set -euo pipefail. Covers w7K's cases:
#   1 positive: G13C device line -> chip OK, both RESULT lines, PASS, rc 0
#   2 negative: a non-G13C device -> FAIL chip assertion, rc 1
#   3 negative: missing info binary -> clear FAIL, rc 1
# Usage: bash stub-tests/test-g-g13c.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/../g-g13c.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/g13c-stub.XXXX")"
FAILED=0

make_python_stub() { # $1 = dest path (becomes the venv python too)
  cat >"$1" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  -m) case "$2" in
        venv) mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; chmod +x "$3/bin/python"; exit 0 ;;
        pip) exit 0 ;;
      esac ;;
  -) printf '%s\n' "$STUB_INFO_PATH"; exit 0 ;;
  -c) printf '%s\n' "$STUB_INFO_PATH"; exit 0 ;;
esac
if [[ "${1##*.}" == "py" ]]; then
  if [[ -n "${MLX_OMARCHY_CPU_PD_HOLD:-}" && "${MLX_OMARCHY_CPU_PD_HOLD:-}" != "0" ]]; then
    echo "RESULT during=7 after=0"
  else
    echo "RESULT during=0 after=0"
  fi
  exit 0
fi
exit 0
STUB
  chmod +x "$1"
}

make_fakebin() { # $1 = case dir, $2 = device line to serve ("" = no binary)
  local d="$1/fakebin" dev="$2"
  mkdir -p "$d"
  for t in mktemp tee grep timeout flock date cat; do
    ln -sf "$(command -v "$t")" "$d/$t"
  done
  # sudo: pass through without privilege
  printf '#!/usr/bin/env bash\nexec "$@"\n' >"$d/sudo"; chmod +x "$d/sudo"
  # python3.14 stub: venv/pip/-c dispatch; the venv python copy serves probes
  STUB_INFO_PATH="$d/stub-info"
  make_python_stub "$d/python3.14"
  if [[ -n "$dev" ]]; then
    {
      echo "#!/usr/bin/env bash"
      echo "cat <<'OUT'"
      echo "mlx-omarchy-info"
      printf '  device:            %s\n' "$dev"
      echo "  driver:            Honeykrisp (stub)"
      echo "OUT"
    } >"$STUB_INFO_PATH"
    chmod +x "$STUB_INFO_PATH"
  fi
  # the stub venv python must see the same STUB_INFO_PATH
  export STUB_INFO_PATH
}

run_case() { # $1 = case name, $2 = device line ("" = missing binary)
  local name="$1" dev="$2"
  local case_dir="$WORK/$name"
  make_fakebin "$case_dir" "$dev"
  (
    export PATH="$case_dir/fakebin:$PATH"
    export G13C_WHEEL="/tmp/stub/mlx_omarchy-0.32.4.dev202610070139+d86daf9-cp314-cp314-linux_aarch64.whl"
    export G13C_RECEIPT="$case_dir/receipt.log"
    set -euo pipefail
    bash "$GATE"
  ) >"$case_dir/out.log" 2>&1
  echo $? >"$case_dir/rc"
}

check() { # check <description> <condition result 0/1>
  if (( $2 == 0 )); then echo "ok: $1"; else echo "FAIL: $1"; FAILED=1; fi
}

# case 1: positive
run_case positive "Apple M1 Max (G13C C0)"
rc=$(cat "$WORK/positive/rc")
check "positive rc=0" $([[ $rc == 0 ]]; echo $?)
grep -q "G13C_CHIP_OK G13C" "$WORK/positive/receipt.log"; check "positive chip OK in receipt" $?
grep -q "RESULT during=0 after=0" "$WORK/positive/receipt.log"; check "positive default RESULT raw in receipt" $?
grep -q "RESULT during=7 after=0" "$WORK/positive/receipt.log"; check "positive forced RESULT raw in receipt" $?
grep -q "G13C_PASS" "$WORK/positive/out.log"; check "positive PASS line" $?

# case 2: negative chip
run_case negative "Apple M2 Max (G14C B1)"
rc=$(cat "$WORK/negative/rc")
check "negative rc=1" $([[ $rc == 1 ]]; echo $?)
grep -q "FAIL chip assertion" "$WORK/negative/out.log"; check "negative chip FAIL message" $?
grep -q "G13C_PASS" "$WORK/negative/out.log" && r=1 || r=0; check "negative has no PASS" $r

# case 3: missing info binary
run_case missing ""
rc=$(cat "$WORK/missing/rc")
check "missing rc=1" $([[ $rc == 1 ]]; echo $?)
grep -qE "FAIL.*(mlx-omarchy-info failed|no device line)" "$WORK/missing/out.log"; check "missing info FAIL message" $?

echo "== raw outputs =="
for c in positive negative missing; do echo "--- $c (rc=$(cat "$WORK/$c/rc"))"; cat "$WORK/$c/out.log"; done
if (( FAILED )); then echo "STUB_HARNESS g-g13c: FAIL"; exit 1; fi
echo "STUB_HARNESS g-g13c: PASS"
rm -rf "$WORK"
