#!/bin/bash
# Runs the omarchy_sdpa_split_tests and omarchy_sdpa_decode_fused_tests binaries built on the Apple M2 Max machine at the PR commit 6943e37a (sha256 f11e9552c6a13b0c3b4d7baa16611acf7e2934aee0857550e74b38a83f3ae11f
# and 825528a0a3dba269870f83a339b158f3ecfc55601e5fccb18d5538121a79099d, copied here) on THIS Mac, with the Apple GPU through the installed honeykrisp driver, and fails when any GPU case printed Skipping
# or a summary does not show 0 failed. The split binary reads the device name it runs on: its default-rule case expects the fused route below the part threshold and the split route above it.
# usage: sdpa-run-bin.sh OUT      Writes only OUT.
set -uo pipefail
out=${1:?out dir}
D=$HOME/work
export VK_DRIVER_FILES=/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json
test ! -e "$out" || { echo "ABORT $out exists"; exit 1; }
mkdir -p "$out"
echo "stage run boot $(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel $(uname -r) driver $(sha256sum /usr/lib/omarchy-mlx/vulkan/libvulkan_driver.so | cut -c1-16) start $(date -u +%FT%TZ) load1 $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$out/run.log"
rc=0
run_one() {
  local name=$1 want=$2 bin one
  bin=$D/$name-6943e37a
  [ "$(sha256sum "$bin" | cut -c1-64)" = "$want" ] || { echo "ABORT $name binary hash differs" | tee -a "$out/run.log"; rc=1; return; }
  sha256sum "$bin" | tee -a "$out/run.log"
  timeout 900 "$bin" > "$out/$name.txt" 2>&1
  one=$?
  tail -5 "$out/$name.txt" | tee -a "$out/run.log"
  echo "$name rc $one" | tee -a "$out/run.log"
  if grep -q '^Skipping' "$out/$name.txt"; then
    echo "NOT CONFIRMED $name (a GPU case printed Skipping: no Vulkan device was used)" | tee -a "$out/run.log"; rc=1
  elif grep -Eq 'test cases: +[0-9]+ \| +[0-9]+ passed \| +0 failed' "$out/$name.txt" && grep -Eq 'assertions: +[0-9]+ \| +[0-9]+ passed \| +0 failed' "$out/$name.txt"; then
    echo "ALL CASES RAN, 0 FAILED, no case skipped: $name $(grep -E 'assertions:' "$out/$name.txt" | head -1)" | tee -a "$out/run.log"
    [ "$one" = 0 ] || rc=1
  else
    echo "NOT CONFIRMED $name (no all-passed summary)" | tee -a "$out/run.log"; rc=1
  fi
}
run_one omarchy_sdpa_split_tests f11e9552c6a13b0c3b4d7baa16611acf7e2934aee0857550e74b38a83f3ae11f
run_one omarchy_sdpa_decode_fused_tests 825528a0a3dba269870f83a339b158f3ecfc55601e5fccb18d5538121a79099d
echo "done $(date -u +%FT%TZ) rc $rc" | tee -a "$out/run.log"
exit $rc
