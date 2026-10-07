#!/usr/bin/env bash
# Stub harness for the g14 dead-leg fix: the leg server exits immediately and
# the gate must FAIL FAST (liveness wrapper kills the driver, "leg process
# died", rc 1, well under the old 1200 s poll timeout). Runs the REAL staged
# g14-routing.sh with a stubbed python3/flock/timeout environment.
# Usage: bash stub-tests/test-g14-deadleg.sh <staged-g14-routing.sh>
set -u
GATE="${1:?pass the staged g14-routing.sh path}"
GATE="$(cd "$(dirname "$GATE")" && pwd)/$(basename "$GATE")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/g14-deadleg.XXXX")"
FAILED=0
trap 'rm -rf "$WORK"' EXIT

FAKEBIN="$WORK/fakebin"
mkdir -p "$FAKEBIN"
for t in mktemp tee grep date cat basename dirname sleep; do
  ln -sf "$(command -v "$t")" "$FAKEBIN/$t"
done
# python3 stub: setsid line execs the rest directly; -c exits quietly;
# driver-file invocations are answered by the stub driver below.
cat >"$FAKEBIN/python3" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  -c) shift 2; exec "$@" ;;
  *) last="${@: -1}"
     case "$last" in
       *g14-routing-driver.py) exec /bin/bash "$STUB_DRIVER" ;;
       *) exit 0 ;;
     esac ;;
esac
STUB
chmod +x "$FAKEBIN/python3"
# the stub driver: a 300 s poll loop the liveness wrapper must kill
printf '#!/usr/bin/env bash\nsleep 300\n' >"$WORK/g14-routing-driver.py"
# flock/timeout real; GPU_LOCK in the temp tree
touch "$WORK/m2-gpu.lock"
export GPU_LOCK="$WORK/m2-gpu.lock"
export STUB_DRIVER="$WORK/g14-routing-driver.py"

# the gate's launch target: a leg server that dies immediately (dead leg)
mkdir -p "$WORK/gate-home/.local/bin"
printf '#!/usr/bin/env bash\nexit 0\n' >"$WORK/gate-home/.local/bin/mlx-omarchy-chat"
chmod +x "$WORK/gate-home/.local/bin/mlx-omarchy-chat"

START=$(date +%s)
(
  export PATH="$FAKEBIN:/usr/bin:/bin"
  export TAG=v0.7.29-test
  export GATE_ROOT="$WORK/gates"
  export GATES_DIR="$WORK/stub-gates"
  export HF_CACHE="$WORK/hf"
  export GATE_INSTALL_PATH="/usr/bin:/bin"
  mkdir -p "$GATES_DIR" "$GATE_ROOT"
  printf '#!/usr/bin/env bash\nsleep 300\n' >"$GATES_DIR/g14-routing-driver.py"
  exec /usr/bin/env bash "$GATE"
) >"$WORK/out.log" 2>&1
rc=$?
END=$(date +%s)
wall=$((END-START))

check() { if (( $2 == 0 )); then echo "ok: $1"; else echo "FAIL: $1"; FAILED=1; fi; }
check "rc=1 on dead leg" $([[ $rc == 1 ]]; echo $?)
if (( wall < 60 )); then check "fail-fast (wall=${wall}s < 60)" 0; else check "fail-fast (wall=${wall}s >= 60)" 1; fi
grep -q "leg A process died during the driver run" "$WORK/out.log"; check "liveness message" $?
grep -q "leg B" "$WORK/out.log" && r=1 || r=0; check "stopped at leg A (no leg B)" $r

echo "== raw (rc=$rc wall=${wall}s) =="
cat "$WORK/out.log"
if (( FAILED )); then echo "STUB_HARNESS g14-deadleg: FAIL"; exit 1; fi
echo "STUB_HARNESS g14-deadleg: PASS"
