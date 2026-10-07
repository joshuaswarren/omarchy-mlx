#!/usr/bin/env bash
# Gate 7c — packaged ANE worker verify (runs ON the jw16 ANE host, inside a
# gpuwin window driven by run-all.sh or the operator). Builds a PRIVATE venv
# from the DRAFT wheel + vendor tar and runs the packaged golden e2e verify
# with the venv's python (the data-file shebang resolves the system python,
# which lacks the installed packages — invoke through the venv explicitly).
# Never touches the serving venv; SERVING_VENV env names it for the guard.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
P="$GATE_HOME/.local/share/mlx-omarchy"
LOG="$LOG_DIR/g7c-ane-worker-verify.log"
VDIR="/tmp/${TAG}-vendor-g7c"
: > "$LOG"
gate_begin "$LOG"
gate_log "$LOG" "loadavg_before=$(cat /proc/loadavg)"
gate_log "$LOG" "psi_cpu_before=$(cat /proc/pressure/cpu 2>/dev/null)"

if [[ -n "$SERVING_VENV" ]] && [[ "$(realpath "$SERVING_VENV" 2>/dev/null || echo none)" == "$P/venv" ]]; then
  gate_log "$LOG" "REFUSE: would clobber serving venv"
  exit 99
fi
gate_log "$LOG" "SERVING_VENV=${SERVING_VENV:-unset} PRIVATE_VENV=$P/venv"

gate_require_asset "$(gate_wheel)" "aarch64 wheel"
gate_require_asset "$(gate_vtar)" "vendor tar"
rm -rf "$GATE_HOME" "$VDIR"
mkdir -p "$GATE_HOME" "$VDIR"
tar -xf "$(gate_vtar)" -C "$VDIR" --strip-components=1
gate_log "$LOG" "vendor_wheel_count=$(ls "$VDIR"/*.whl 2>/dev/null | wc -l)"
gate_log "$LOG" "lock_lines=$(wc -l < "$VDIR/requirements-lock.txt")"

"$PY_AARCH64" -m venv "$P/venv" 2>>"$LOG"
"$P/venv/bin/pip" install --quiet --no-index --find-links "$VDIR" \
  -r "$VDIR/requirements-lock.txt" 2>>"$LOG"
gate_log "$LOG" "INSTALL_EXIT $?"

"$P/venv/bin/python3" - <<'PY' 2>&1 | tee -a "$LOG"
import importlib.metadata as m
print("mlx_omarchy", m.version("mlx_omarchy"))
import numpy
print("numpy", numpy.__version__)
import google.protobuf
print("protobuf", google.protobuf.__file__)
PY

SITE=$(echo "$P"/venv/lib/python3.*/site-packages/mlx/bin)
ls -la "$SITE/mlx-omarchy-ane-worker" "$SITE/mlx-omarchy-parakeet" "$SITE/mlx-omarchy-info" 2>&1 | tee -a "$LOG"

"$P/venv/bin/python3" "$SITE/mlx-omarchy-parakeet" verify 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
gate_log "$LOG" "VERIFY_EXIT $RC"

# Packaged-flow ABI + GC regression (2026-10-04 wave-scheduler crash): the
# trace snapshot must be read through the SHIPPED mirror module with its ABI
# size guard, then mlx arrays, a short eval chain, and repeated GC passes
# with the wave scheduler ON — the exact flow whose silent struct-growth
# overflow killed the worker at its first GC pass.
gate_log "$LOG" "G7C_TRACE_ABI_BEGIN"
MLX_OMARCHY_WAVE_SCHED=1 "$P/venv/bin/python3" - "$(dirname "$SITE")" <<'PY' 2>&1 | tee -a "$LOG"
import ctypes, gc, os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "coreml"))
from trace_abi import TraceSnapshot, trace_snapshot, library_path
lib = ctypes.CDLL(library_path())
probe = lib.mlx_omarchy_trace_snapshot_abi_size
probe.argtypes = []
probe.restype = ctypes.c_uint64
size = int(probe())
mirror = ctypes.sizeof(TraceSnapshot)
assert size == mirror, f"ABI drift: libmlx writes {size} bytes, mirror is {mirror}"
import mlx.core as mx
before = trace_snapshot()
xs = [mx.ones((64, 64), dtype=mx.float32) for _ in range(8)]
acc = xs[0]
for x in xs[1:]:
    acc = acc + x
mx.eval(acc)
del xs, acc
for _ in range(5):
    gc.collect()
after = trace_snapshot()
print(f"trace_abi ok size={size} wave={os.environ.get('MLX_OMARCHY_WAVE_SCHED')} "
      f"dispatches={after['gpu_primitive_dispatches'] - before['gpu_primitive_dispatches']} "
      f"emitted={after['barriers_emitted'] - before['barriers_emitted']} "
      f"skipped={after['barriers_skipped'] - before['barriers_skipped']}")
PY
RCABI=${PIPESTATUS[0]}
gate_log "$LOG" "G7C_TRACE_ABI_EXIT $RCABI"
[[ $RCABI -ne 0 ]] && RC=$RCABI

REPORT=$(ls -t "$HOME"/.cache/mlx-omarchy/parakeet-reference/transcriptions/*/transcribe-report.json \
             "$HOME"/.cache/mlx-omarchy/transcriptions/*/transcribe-report.json 2>/dev/null | head -1)
gate_log "$LOG" "REPORT_PATH=$REPORT"
if [[ -n "$REPORT" && -s "$REPORT" ]]; then
  RDIR=$(dirname "$REPORT")
  "$P/venv/bin/python3" - "$REPORT" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
ex = d.get("execution", {})
print("status", "=", d.get("status"))
print("ane_mode", "=", ex.get("ane_mode"))
print("cpu_tensor_events", "=", ex.get("cpu_tensor_events"))
print("emissions", "=", d.get("emissions"))
print("transcript_sha256", "=", d.get("transcript_sha256"))
print("total_pipeline_ms", "=", d.get("timing", {}).get("total_pipeline_ms"))
for c in d.get("verification", {}).get("checks", []):
    print("  check", c.get("check"), "pass=", c.get("pass"))
PY
  cp "$REPORT" "$LOG_DIR/g7c-e2e-report.json" 2>/dev/null || true
  cp "$RDIR/transcript.txt" "$LOG_DIR/g7c-transcript.txt" 2>/dev/null || true
fi
gate_log "$LOG" "POST_RUN active=$(systemctl is-active llm-inference.service 2>/dev/null)"
gate_log "$LOG" "GATE7C_EXIT $RC"
exit "$RC"
