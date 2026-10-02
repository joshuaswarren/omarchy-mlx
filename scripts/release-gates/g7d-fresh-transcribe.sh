#!/usr/bin/env bash
# Gate 7d — fresh-image parakeet transcribe end to end (NEW after the real
# fresh-Omarchy defect: the installed `mlx-omarchy-parakeet transcribe` bound
# the system python and refused with "missing runtime dependencies: numpy,
# google.protobuf"; fixed by the staged launcher + CLI self-heal).
#
# What it proves, as a user would hit it:
#   ane mode (ANE host, e.g. inside a gpuwin window on jw16):
#     - install from the DRAFT assets exactly like a user (private venv from
#       wheel + vendor lock), fresh home, no manual pip;
#     - assert the SYSTEM interpreter is dep-less (the fresh-image property);
#     - invoke the installed CLI entry as a user would — the staged launcher
#       (G7D_LAUNCHER, from install.sh --system) or the packaged data file via
#       its `env python3` shebang — twice: plain and `python3 -S` (the
#       verbatim defect form);
#     - transcribe the packaged golden fixture; require exit 0, report
#       status=match with every pin check green, ane_mode true,
#       cpu_tensor_events 0, and transcript sha == the pinned golden sha
#       (cross-checked against the wheel's own parakeet-reference.lock);
#     - the old failure string must NOT appear anywhere.
#   cpu mode (dev box, no ANE): proves the dependency boundary ONLY, stopping
#     right after the dependency probe: stage the installed layout under a
#     throwaway prefix (owning venv python with numpy+protobuf), drive the
#     CLI's own probe under `python3 -S`; PASS = the failure string is absent
#     and the probe crossed into the owning venv (downstream refusal or clean
#     pass). A pre-fix CLI refuses with the failure string -> gate FAILs.
#     Set G7D_CPU_ONLY=1 (or G7D_MODE=cpu).
#
# The old-failure demonstration against published v0.7.14 assets: run this
# gate with ASSETS_DIR pointing at those assets (no overlay); it must FAIL
# with the old string in the log. Then rerun with G7D_OVERLAY_CLI=<fixed CLI>
# to see PASS.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
G7D_MODE="${G7D_MODE:-${G7D_CPU_ONLY:+cpu}}"
G7D_MODE="${G7D_MODE:-ane}"
G7D_CACHE="${G7D_CACHE:-$HOME/.cache/mlx-omarchy}"   # staged reference cache root (g7c stages it)
G7D_LAUNCHER="${G7D_LAUNCHER:-}"                      # optional staged launcher to prefer
G7D_OVERLAY_CLI="${G7D_OVERLAY_CLI:-}"                # optional fixed-CLI overlay (proof runs)
FAILSTR="missing runtime dependencies for transcribe"
# Golden transcript pin, cross-checked against the wheel's own lock below.
GOLDEN_TRANSCRIPT_SHA="db501a8c080380ea027ffa50a4b4956c39df77cb692c4fb78e556311a11a0790"

LOG="$LOG_DIR/g7d-fresh-transcribe.log"
: > "$LOG"
gate_begin "$LOG"
gate_log "$LOG" "mode=$G7D_MODE gate7d_home=$GATE7D_HOME cache=$G7D_CACHE"

gate_refuse_existing "$GATE7D_HOME"
if [[ -n "$SERVING_VENV" ]]; then
  P="$GATE7D_HOME/.local/share/mlx-omarchy"
  if [[ "$(realpath "$SERVING_VENV" 2>/dev/null || echo none)" == "$P/venv" ]]; then
    gate_log "$LOG" "REFUSE: would clobber serving venv"; exit 99
  fi
fi

VPY=""   # venv python of the installed layout (ane mode)
CLI=""   # the installed CLI data file

if [[ "$G7D_MODE" == "ane" ]]; then
  # --- user-like install from the DRAFT assets -----------------------------
  gate_require_asset "$(gate_wheel)" "aarch64 wheel"
  gate_require_asset "$(gate_vtar)" "vendor tar"
  VDIR="/tmp/${TAG}-vendor-g7d"
  P="$GATE7D_HOME/.local/share/mlx-omarchy"
  rm -rf "$GATE7D_HOME" "$VDIR"
  mkdir -p "$GATE7D_HOME" "$VDIR"
  tar -xf "$(gate_vtar)" -C "$VDIR" --strip-components=1
  "$PY_AARCH64" -m venv "$P/venv" 2>>"$LOG"
  "$P/venv/bin/pip" install --quiet --no-index --find-links "$VDIR" \
    -r "$VDIR/requirements-lock.txt" 2>>"$LOG"
  INSTALL_RC=$?
  gate_log "$LOG" "INSTALL_EXIT $INSTALL_RC"
  [[ $INSTALL_RC -eq 0 ]] || { gate_log "$LOG" "GATE7D_EXIT 1 (install failed)"; exit 1; }
  "$P/venv/bin/python3" -c 'import importlib.metadata as m; print("mlx_omarchy", m.version("mlx_omarchy"))' 2>&1 | tee -a "$LOG"
  SITEBASE=$(echo "$P"/venv/lib/python3.*/site-packages)
  CLI="$SITEBASE/mlx/bin/mlx-omarchy-parakeet"
  VPY="$P/venv/bin/python3"
  [[ -s "$CLI" ]] || { gate_log "$LOG" "REFUSING: installed CLI not found at $CLI"; exit 1; }

  if [[ -n "$G7D_OVERLAY_CLI" ]]; then
    gate_log "$LOG" "overlay_before_sha $(sha256sum "$CLI" | awk '{print $1}')"
    cp "$G7D_OVERLAY_CLI" "$CLI"
    gate_log "$LOG" "overlay_after_sha  $(sha256sum "$CLI" | awk '{print $1}')"
  fi

  # The staged reference cache: g7c stages it (or a prior `download` run);
  # `download --json` verifies hashes and fetches only what is missing.
  "$VPY" "$CLI" download --json >>"$LOG" 2>&1
  CACHE_RC=$?
  gate_log "$LOG" "CACHE_STAGE_EXIT $CACHE_RC"
  [[ $CACHE_RC -eq 0 ]] || { gate_log "$LOG" "REFUSING: reference cache staging failed"; exit 1; }
  gate_log "$LOG" "CACHE_STAGED $G7D_CACHE"

  # Fresh-image property: the SYSTEM interpreter (what `env python3`
  # resolves) must lack the transcribe deps, or nothing is proven.
  PY_SYS="$(command -v python3)"
  if "$PY_SYS" -S -c 'import numpy, google.protobuf' 2>/dev/null; then
    gate_log "$LOG" "REFUSING: $PY_SYS already has numpy+protobuf; not a fresh-image surface"
    exit 1
  fi
  gate_log "$LOG" "SYSTEM_INTERPRETER_DEPLESS $PY_SYS"

  ENTRY="${G7D_LAUNCHER:-$CLI}"
  gate_log "$LOG" "ENTRY $ENTRY (launcher=${G7D_LAUNCHER:+set}${G7D_LAUNCHER:-none: packaged data file})"
  RC=0
  GATE_START=$(date +%s)  # reports older than this are stale, never ours
  run_transcribe() { # run_transcribe <label> <cmd...>
    local label="$1"; shift
    gate_log "$LOG" "--- transcribe ($label) ---"
    OUT=$(env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE7D_HOME" \
      MLX_OMARCHY_CACHE_DIR="$G7D_CACHE" TERM=dumb timeout -k 30 900 "$@" 2>&1)
    local rc=$?
    printf '%s\n' "$OUT" >>"$LOG"
    gate_log "$LOG" "TRANSCRIBE_EXIT[$label] $rc"
    if printf '%s\n' "$OUT" | grep -q "$FAILSTR"; then
      gate_log "$LOG" "OLD_FAILURE_STRING PRESENT[$label] — DEFECT"
      RC=1
    fi
    [[ $rc -eq 0 ]] || RC=1
  }
  run_transcribe entry "$ENTRY" transcribe
  run_transcribe bare-S "$PY_SYS" -S "$ENTRY" transcribe

  # Golden contract: the report OF THIS RUN must match the pinned transcript.
  "$VPY" - "$G7D_CACHE" "$GOLDEN_TRANSCRIPT_SHA" "$LOG" "$GATE_START" <<'PY' || RC=1
import glob, hashlib, json, os, sys
cache, golden, log, start = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
reports = sorted(
    [p for p in glob.glob(f"{cache}/parakeet-reference/transcriptions/*/transcribe-report.json")
     + glob.glob(f"{cache}/transcriptions/*/transcribe-report.json")
     if os.path.getmtime(p) >= start - 5],
    key=os.path.getmtime)
def line(*a):
    print(*a); open(log, "a").write(" ".join(str(x) for x in a) + "\n")
if not reports:
    line("GOLDEN_FAIL no transcribe-report.json written by THIS run"); sys.exit(1)
report = reports[-1]
d = json.load(open(report))
ex = d.get("execution", {})
line("REPORT", report)
line("status =", d.get("status"),
     "ane_mode =", ex.get("ane_mode"),
     "cpu_tensor_events =", ex.get("cpu_tensor_events"),
     "checks_failed =", d.get("checks_failed"))
checks = d.get("verification", {}).get("checks", [])
for c in checks:
    line("  check", c.get("check"), "pass =", c.get("pass"))
tdir = os.path.dirname(report)
sha = hashlib.sha256(open(os.path.join(tdir, "transcript.txt"), "rb").read()).hexdigest()
line("transcript_sha256 =", sha)
line("golden_pin =", golden)
lock = glob.glob(os.path.join(os.path.dirname(report), "..", "..", "*", "*", "parakeet-reference.lock"))
lock_sha = None
for cand in glob.glob(f"{cache}/**/parakeet-reference.lock", recursive=True):
    try:
        lock_sha = json.load(open(cand)).get("macos_reference_paths", {}).get("transcript.txt")
        if lock_sha:
            line("wheel_lock_pin =", lock_sha, "(", cand, ")")
            break
    except Exception:
        pass
ok = (d.get("status") == "match"
      and not d.get("checks_failed")
      and all(c.get("pass") for c in checks)
      and ex.get("ane_mode") is True
      and ex.get("cpu_tensor_events") == 0
      and sha == golden
      and (lock_sha is None or lock_sha == golden))
line("GOLDEN_MATCH", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
  gate_log "$LOG" "GATE7D_EXIT $RC"
  exit "$RC"
fi

# --- cpu mode: dependency boundary only (no ANE, no device open) -----------
REPO_ROOT="$(cd "$GATES_DIR/../.." && pwd)"
SRC_CLI="${G7D_CLI:-$REPO_ROOT/overlay/tools/mlx-omarchy-parakeet/mlx_omarchy_parakeet.py}"
[[ -s "$SRC_CLI" ]] || SRC_CLI="$G7D_CLI"
[[ -s "$SRC_CLI" ]] || { gate_log "$LOG" "REFUSING: no CLI source (set G7D_CLI or G7D_OVERLAY_CLI)"; exit 1; }
gate_log "$LOG" "cli_under_test $SRC_CLI sha=$(sha256sum "$SRC_CLI" | awk '{print $1}')"

rm -rf "$GATE7D_HOME"
"$PY_AARCH64" -m venv "$GATE7D_HOME" 2>>"$LOG" \
  || python3 -m venv "$GATE7D_HOME" 2>>"$LOG"
"$GATE7D_HOME/bin/pip" install --quiet numpy protobuf >>"$LOG" 2>&1
gate_log "$LOG" "OWNING_VENV_DEPS_EXIT $?"
SITEBASE=$(echo "$GATE7D_HOME"/lib/python3.*/site-packages)
mkdir -p "$SITEBASE/mlx/bin"
cp -r "$REPO_ROOT/overlay/tools/coreml" "$SITEBASE/mlx/coreml" 2>/dev/null
cp "$SRC_CLI" "$SITEBASE/mlx/bin/mlx-omarchy-parakeet"
CLI="$SITEBASE/mlx/bin/mlx-omarchy-parakeet"

# Dep-less entry interpreter, exactly the defect surface (`python3 -S`).
PY_SYS="$(command -v python3)"
if "$PY_SYS" -S -c 'import numpy, google.protobuf' 2>/dev/null; then
  gate_log "$LOG" "REFUSING: $PY_SYS -S has the deps; boundary not exercisable here"
  exit 1
fi
gate_log "$LOG" "SYSTEM_INTERPRETER_DEPLESS $PY_SYS -S"

OUT=$("$PY_SYS" -S - "$CLI" <<'PY' 2>&1
import importlib.util, sys
from importlib.machinery import SourceFileLoader
cli = sys.argv[1]
sys.argv = ["mlx-omarchy-parakeet", "transcribe"]
loader = SourceFileLoader("mop_cli", cli)  # the installed file has no .py suffix
spec = importlib.util.spec_from_loader("mop_cli", loader)
mod = importlib.util.module_from_spec(spec)
loader.exec_module(mod)
mod._check_runtime_deps()
print("DEP_PROBE_PASS")
PY
)
rc=$?
printf '%s\n' "$OUT" >>"$LOG"
gate_log "$LOG" "PROBE_EXIT $rc"
if printf '%s\n' "$OUT" | grep -q "$FAILSTR"; then
  gate_log "$LOG" "OLD_FAILURE_STRING PRESENT — DEFECT (dependency boundary NOT crossed)"
  RC=1
elif printf '%s\n' "$OUT" | grep -q "DEP_PROBE_PASS\|error:"; then
  gate_log "$LOG" "DEP_BOUNDARY_CROSSED (probe passed under the owning venv; downstream: $(printf '%s' "$OUT" | grep 'error:' | head -1))"
  RC=0
else
  gate_log "$LOG" "UNEXPECTED probe outcome"
  RC=1
fi
gate_log "$LOG" "GATE7D_EXIT $RC"
exit "$RC"
