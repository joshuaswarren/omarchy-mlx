#!/bin/bash
# MatmulGap H34 (one gpu-turn ticket, about 20 min): base arm vs the PR wheel at LLM-like shapes, three rounds
# BASE PR / PR BASE / BASE PR, idle gap before every arm run (60 s on the M2 Max, 30 s elsewhere).
# Usage: gen_ticket.sh g13g|g14c|g13c. Base arm: C2 (venv-c2) on g13g and g14c, M (venv-h13main) on g13c, because the PR
# changes no G13C row and the question there is the shape generality of the rows already landed (H34 amendment 1).
set -u
CHIP=${1:?chip g13g|g14c|g13c}
H=/var/tmp/w7q-h8
D=$HOME/.local/share/coreglass/vulkan-v3-6543eeb7df
export VK_ICD_FILENAMES=$D/honeykrisp_icd.aarch64.json MESA_SHADER_CACHE_DISABLE=1
GAP=30
if [ "$CHIP" = g14c ]; then GAP=60; fi
BASE=C2
BASEV=venv-c2
if [ "$CHIP" = g13c ]; then BASE=M; BASEV=venv-h13main; fi
if [ ! -x "$H/$BASEV/bin/python" ] || [ ! -x "$H/venv-pr/bin/python" ]; then echo "$BASEV or venv-pr missing, nothing run"; exit 2; fi
shopt -s nullglob
LIBS=("$D"/libvulkan_*.so)
shopt -u nullglob
if [ "${#LIBS[@]}" -ne 1 ]; then echo "driver library: expected 1 match in $D, found ${#LIBS[@]}"; exit 2; fi
DRV=$(sha256sum "${LIBS[0]}" | cut -c1-16)
echo "start $(date -u +%FT%TZ) host $(hostname) $(uname -r) load $(cut -d' ' -f1 /proc/loadavg) driver $DRV base $BASE"
ROUND=0
run() { # arm venv
  sleep "$GAP"
  timeout -k 20 900 "$H/$2/bin/python" "$H/gen_grid.py" "$1" "$CHIP" > "$H/gen-$CHIP-$1-r$ROUND.jsonl" 2> "$H/gen-$CHIP-$1-r$ROUND.err"
  echo "$1 r$ROUND rc=$? cells $(grep -c '"k": "cell"' "$H/gen-$CHIP-$1-r$ROUND.jsonl") $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)"
}
ROUND=1; run "$BASE" "$BASEV"; run PR venv-pr
ROUND=2; run PR venv-pr; run "$BASE" "$BASEV"
ROUND=3; run "$BASE" "$BASEV"; run PR venv-pr
echo "end $(date -u +%FT%TZ)"
