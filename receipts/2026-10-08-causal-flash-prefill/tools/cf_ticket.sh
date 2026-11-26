#!/bin/bash
# MatmulGap H35 tickets (one gpu-turn ticket each, no lock call inside).
#   cf_ticket.sh bringup            : device test case for the causal coopmat flash kernel (both ROWP variants), no timing
#   cf_ticket.sh select             : ROWP selection at L = 768 and 1536 (composed, ROWP 1, ROWP 2; 3 repeats, 30 s gaps)
#   cf_ticket.sh gate <rowp>        : gate micro at L = 512, 1024, 2048 (composed and the frozen variant; 3 repeats, 30 s gaps)
# Results go to $H/cf-<mode>-<host>.log. Venv: venv-cf. Test tree: tbuild-cf.
set -u
H=${WORK_DIR:?set WORK_DIR to the directory with the venvs, the test tree, sdpa_micro.py and the logs}
D=$HOME/.local/share/coreglass/vulkan-v3-6543eeb7df
export VK_ICD_FILENAMES=$D/honeykrisp_icd.aarch64.json MESA_SHADER_CACHE_DISABLE=1
HOST=$(hostname)
shopt -s nullglob
LIBS=("$D"/libvulkan_*.so)
shopt -u nullglob
if [ "${#LIBS[@]}" -ne 1 ]; then echo "driver library: expected 1 match in $D, found ${#LIBS[@]}"; exit 2; fi
DRV=$(sha256sum "${LIBS[0]}" | cut -c1-16)
MODE=${1:?mode bringup|select|gate}
LOG=$H/cf-$MODE-$HOST.log
PY=$H/venv-cf/bin/python
echo "start $(date -u +%FT%TZ) host $HOST kernel $(uname -r) load $(cut -d' ' -f1 /proc/loadavg) driver $DRV wheel $($PY -c 'import mlx.core as mx; print(mx.__version__)' 2>&1 | tail -1)" | tee "$LOG"
micro() { # arm label, env assignments..., length list
  local label=$1 lens=$2
  shift 2
  sleep "${GAP:-30}"
  echo "$label L=$lens: $(env "$@" timeout 200 "$PY" "$H/sdpa_micro.py" causal "$lens" 2>&1 | tail -1 | cut -c1-900)" | tee -a "$LOG"
}
case $MODE in
  bringup)
    BIN=$H/tbuild-cf/tests/omarchy/omarchy_sdpa_prefill_flash_tests
    echo "bin $(sha256sum "$BIN" | cut -c1-16)" | tee -a "$LOG"
    FULL=$H/cf-bringup-full-$HOST.log
    timeout -k 20 800 "$BIN" -tc='causal coopmat flash*' > "$FULL" 2>&1
    echo "rc=$? full output $FULL: $(grep -E '^\[doctest\] (test cases|assertions)' "$FULL" | tr '\n' ' ')" | tee -a "$LOG"
    ;;
  select)
    for rep in 1 2 3; do
      micro composed 768,1536 X35=1
      micro rowp1 768,1536 MLX_OMARCHY_SDPA_CAUSAL_FLASH=1 MLX_OMARCHY_SDPA_CAUSAL_FLASH_ROWP=1
      micro rowp2 768,1536 MLX_OMARCHY_SDPA_CAUSAL_FLASH=1 MLX_OMARCHY_SDPA_CAUSAL_FLASH_ROWP=2
    done
    ;;
  gate)
    ROWP=${2:?rowp}
    for rep in 1 2 3; do
      micro composed 512,1024,2048 X35=1
      micro "flash-rowp$ROWP" 512,1024,2048 MLX_OMARCHY_SDPA_CAUSAL_FLASH=1 MLX_OMARCHY_SDPA_CAUSAL_FLASH_ROWP=$ROWP
    done
    ;;
  *) echo "usage"; exit 2 ;;
esac
echo "end $(date -u +%FT%TZ)" | tee -a "$LOG"
