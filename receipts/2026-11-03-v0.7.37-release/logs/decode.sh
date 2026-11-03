#!/usr/bin/env bash
# Release step 6: one pinned 4-bit decode on the UPLOADED aarch64 wheel, installed from the draft's vendored set into a fresh venv.
# Same method as w7Q's v0.7.36 run (decode-rel0736b.sh): Qwen3.8-2B 4-bit, greedy, seed 0, 128 tokens, MLX_DISABLE_COMPILE=1, five separate runs.
# usage: decode.sh ASSETS_DIR TREE OUT_DIR   (OUT_DIR must not exist; TREE is the checkout of the tag; the ICD is the system Honeykrisp ICD)
set -uo pipefail
A=${1:?assets dir}; WT=${2:?tag tree}; B=${3:?output dir}
set -- "$A"/mlx_omarchy-*cp314*linux_aarch64.whl
[ "$#" -eq 1 ] && [ -f "$1" ] || { echo "expected exactly one aarch64 wheel in $A"; exit 2; }
WHEEL="$1"
set -- "$A"/omarchy-mlx-vendor-wheels-v*-cp314-aarch64.tar
[ "$#" -eq 1 ] && [ -f "$1" ] || { echo "expected exactly one vendor tar in $A"; exit 2; }
VTAR="$1"
MODEL="$HOME/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381"
PROMPT="What is the capital of France? Answer in one word."
EXPECT_IDS=775bf034e185c8f9
ICD=/usr/share/vulkan/icd.d/a[redacted]_icd.json
[ ! -e "$B" ] || { echo "refusing: $B exists"; exit 2; }
[ -f "$MODEL/config.json" ] || { echo "model snapshot missing"; exit 2; }
[ -f "$ICD" ] || { echo "ICD file missing: $ICD"; exit 2; }
[ -f "$WT/scripts/caps_sim_guard.py" ] || { echo "caps_sim_guard.py missing in the tag tree"; exit 2; }
mkdir -p "$B/vendor"
echo "start $(date -u +%T) free $(df -BG --output=avail "$HOME" | tail -1) load $(cut -d' ' -f1 /proc/loadavg)"
tar -xf "$VTAR" -C "$B/vendor" --strip-components=1
python3.14 -m venv "$B/venv"
"$B/venv/bin/pip" install --quiet --no-index --find-links "$B/vendor" -r "$B/vendor/requirements-lock.txt" 2> "$B/pip.err"
echo "INSTALL_EXIT $?"
"$B/venv/bin/python" -c 'import importlib.metadata as m; print("installed", m.version("mlx_omarchy"), "mlx_lm", m.version("mlx_lm"))'
ok=0
for i in 1 2 3 4 5; do
  echo "== run $i $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)"
  out="$(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MLX_DISABLE_COMPILE=1 VK_DRIVER_FILES="$ICD" timeout 240 "$B/venv/bin/python" "$WT/scripts/bench_decode.py" \
      --model "$MODEL" --prompt "$PROMPT" --tokens 128 --temp 0 --seed 0 --wheel "$WHEEL" 2>&1)"; rc=$?
  echo "$out" | grep -E "decode |prefill|generated_ids|provenance|PROVENANCE|verified|ERROR|Error|refus|uclamp" | cut -c1-230
  echo "bench_decode exit $rc"
  [ "$rc" -eq 0 ] || echo "$out" | tail -6 | cut -c1-230
  [ "$rc" -eq 0 ] && ok=$((ok+1))
done
echo "runs completed: $ok of 5; expected ids digest from the v0.7.35 and v0.7.36 receipts: sha256:$EXPECT_IDS"
echo DECODE_DONE
