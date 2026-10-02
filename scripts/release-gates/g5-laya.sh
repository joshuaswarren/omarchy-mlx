#!/usr/bin/env bash
# Gate 5 — Laya completeness: the converted manifests from gates 2 and 3 pin
# the upstream revision.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g5-laya.log"
: > "$LOG"

python3 - "$ASSIST_9B" "$ASSIST_4B" >>"$LOG" 2>&1 <<'PY'
import json, pathlib, sys
found = []
for base in sys.argv[1:]:
    for p in pathlib.Path(base).rglob("manifest.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        rev = m.get("source_revision") or (m.get("source") or {}).get("revision")
        if rev:
            found.append((str(p), rev))
for p, rev in found:
    print("LAYA_MANIFEST", p, rev)
ok = any(rev == "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851" for _, rev in found)
print("GATE5_LAYA", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
RC=$?
gate_log "$LOG" "GATE5_EXIT $RC"
exit "$RC"
