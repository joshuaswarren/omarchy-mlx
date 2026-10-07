#!/usr/bin/env bash
# g-hold — on the T8103/G13G host (the power-experiments lane runs this; the
# lane's own fleet bench may substitute for the A/B leg as long as the
# digest-equality discriminator is preserved).
#
# Verifies, on the RELEASE wheel:
#   1. the udev rule grants group video 0660 without a reboot (any
#      pre-existing rule file is recorded and restored byte-exactly; a rule
#      this run created is removed — never delete a pre-existing rule),
#   2. tests/test_cpu_pd_hold.py really runs 2 tests and prints a bare OK
#      ("Ran 2 tests" + "^OK$"; skipped=2 is a FAIL),
#   3. a greedy A/B (hold default vs MLX_OMARCHY_CPU_PD_HOLD=0), two runs per
#      arm, TOKEN IDS hashed from generate_step (mlx-lm 0.31.3 pinned from
#      the vendor tar; default sampler, no temp kwarg): all four digests must
#      be equal and non-empty, printed raw.
#
# w7K review fixes: H1 mlx-lm installed from the vendor tar + token-id
# digests from generate_step; H2 zero-test and skipped runs FAIL; H3 the live
# tests run under the same ticket/lock as the A/B; H4 four raw digests
# asserted equal and non-empty; every python leg under `timeout -k 30 300`;
# all lines teed to a receipt file.
#
# Usage (on the host):
#   HOLD_WHEEL=/tmp/v0.7.29-assets/mlx_omarchy-*-cp314-cp314-linux_aarch64.whl \
#   HOLD_VENDOR_TAR=/tmp/v0.7.29-assets/omarchy-mlx-vendor-wheels-v0.7.29-cp314-aarch64.tar \
#   HOLD_RULE_SRC=$HOME/rel0729/gates/70-omarchy-mlx-cpu-dma-latency.rules \
#     bash g-hold-g13g.sh
#   (HOLD_MODEL defaults to /var/tmp/MesaParity/model)
# Expected PASS (also in the receipt):
#   HOLD_UDEV root video 660
#   HOLD_TESTS Ran 2 tests ... OK (2 run, 0 skipped)
#   HOLD_AB 4 digests equal: <d> <d> <d> <d>
#   HOLD_PASS
set -euo pipefail
WHEEL="${HOLD_WHEEL:?set HOLD_WHEEL to the release wheel path on this host}"
VENDOR_TAR="${HOLD_VENDOR_TAR:?set HOLD_VENDOR_TAR to the release vendor tar (pinned mlx-lm comes from it)}"
MODEL="${HOLD_MODEL:-/var/tmp/MesaParity/model}"
RULE_SRC="${HOLD_RULE_SRC:?set HOLD_RULE_SRC to packaging/udev/70-omarchy-mlx-cpu-dma-latency.rules from the tag}"
RULE=/etc/udev/rules.d/70-omarchy-mlx-cpu-dma-latency.rules
if [[ -z "${GPU_LOCK:-}" ]]; then
  if grep -aq "t6021\|t6011\|t6010" /proc/device-tree/compatible 2>/dev/null; then
    GPU_LOCK=/tmp/m2-gpu.lock
  else
    GPU_LOCK=/tmp/m1-gpu.lock
  fi
fi
RECEIPT="${HOLD_RECEIPT:-$PWD/g-hold-receipt-$(date -u +%Y%m%dT%H%M%SZ).log}"
: > "$RECEIPT"
log() { tee -a "$RECEIPT"; }
command -v python3.14 >/dev/null || { echo "FAIL python3.14 missing" | log; exit 1; }

VENV="$(mktemp -d "${TMPDIR:-/tmp}/hold-gate.XXXX")"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/hold-gate-work.XXXX")"
RULE_PRE=""; RULE_SAVED="$WORK/preexisting.rules"
if sudo test -f "$RULE"; then
  RULE_PRE="$(sudo sha256sum "$RULE" | awk '{print $1}')"
  sudo cat "$RULE" > "$RULE_SAVED"
fi
cleanup() {
  if [[ -n "$RULE_PRE" ]]; then
    sudo install -m 644 "$RULE_SAVED" "$RULE"   # restore exactly what we found
  else
    sudo rm -f "$RULE" 2>/dev/null || true       # only what THIS run created
  fi
  sudo udevadm control --reload 2>/dev/null || true
  sudo udevadm trigger --subsystem-match=misc --sysname-match=cpu_dma_latency 2>/dev/null || true
  rm -rf "$VENV" "$WORK"
}
trap cleanup EXIT

# gpu-turn ticket when present; flock of the host lock otherwise. Never both.
runner() {
  if command -v gpu-turn >/dev/null 2>&1; then
    gpu-turn -m 5 timeout -k 30 300 "$@"
  else
    timeout -k 30 300 flock "$GPU_LOCK" "$@"
  fi
}

echo "== wheel + vendored mlx-lm + rule ==" | log
python3.14 -m venv "$VENV"
VW="$WORK/vendor-wheels"
mkdir -p "$VW"
tar -xf "$VENDOR_TAR" -C "$VW" --strip-components=1
# The exact packaged set: release wheel first (provides mlx), then every
# vendored wheel with --no-deps (nothing pulls upstream mlx).
"$VENV/bin/python" -m pip install --quiet --no-index --no-deps "$WHEEL"
"$VENV/bin/python" -m pip install --quiet --no-index --no-deps "$VW"/*.whl
# The vendor set must NOT carry an upstream mlx wheel that could shadow the
# release wheel, and the installed mlx must be the release wheel's bytes
# (dist version + libmlx sha against the wheel's RECORD).
if compgen -G "$VW/mlx-*.whl" >/dev/null; then
  echo "FAIL: vendor tar carries an upstream mlx wheel:" | log
  ls "$VW"/mlx-*.whl | log
  exit 1
fi
ls "$VW"/*.whl | log
EXPECTED_MLX_VERSION="${WHEEL##*/mlx_omarchy-}"; EXPECTED_MLX_VERSION="${EXPECTED_MLX_VERSION%%-cp314*}"
export EXPECTED_MLX_VERSION
"$VENV/bin/python" - <<'EOF' | log
import base64, hashlib, importlib.metadata as md, os, pathlib, sys
import mlx
expected = os.environ["EXPECTED_MLX_VERSION"]
version = md.version("mlx_omarchy")
# mlx is a namespace package in this wheel: __file__ is None, __path__[0]
# is .../site-packages/mlx.
site = pathlib.Path(list(mlx.__path__)[0]).resolve().parent
recs = list(site.glob("mlx_omarchy-*.dist-info/RECORD"))
lib = site / "mlx" / "lib" / "libmlx.so"
actual = hashlib.sha256(lib.read_bytes()).hexdigest()
pinned = None
if recs:
    for line in recs[0].read_text().splitlines():
        if line.startswith("mlx/lib/libmlx.so,"):
            b64 = line.split(",")[1].split("=", 1)[1]
            pinned = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()
ok_v = version == expected
ok_h = pinned is not None and pinned == actual and lib.exists()
print(f"MLX_PIN version={version} expected={expected} ok={ok_v}")
print(f"MLX_PIN libmlx_sha={actual} record_sha={pinned} ok={ok_h}")
sys.exit(0 if (ok_v and ok_h) else 1)
EOF
"$VENV/bin/python" -c 'import mlx, mlx_lm; print("mlx_lm", mlx_lm.__version__)' | log
# ICD receipt lines (w7P rule): which ICD this run used, the sha256 of the
# libvulkan_asahi.so actually loaded (from VK_DRIVER_FILES' json
# library_path, else the packaged honeykrisp ICD), and the driverInfo line.
ICD_JSON="${VK_DRIVER_FILES:-/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json}"
LIBV=$(python3 -c "import json;print(json.load(open('$ICD_JSON'))['ICD']['library_path'])")
LIBV="${LIBV/\$DEST/$DEST}"
echo "ICD_LINE icd_json=$ICD_JSON" | log
echo "ICD_LINE libvulkan_sha256=$(sha256sum "${LIBV/#\~/$HOME}" 2>/dev/null | awk '{print $1}')" | log
"$VENV/bin/python" - <<'EOF' | tee -a "$RECEIPT"
import mlx.core as mx
info = mx.device_info()
print("ICD_LINE driver:", info.get("driver", ""), "|", info.get("driver_info", ""))
EOF
sudo install -m 644 "$RULE_SRC" "$RULE"
sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=misc --sysname-match=cpu_dma_latency
sudo udevadm settle
stat_out="$(stat -c '%U %G %a' /dev/cpu_dma_latency)"
[[ "$stat_out" == "root video 660" ]] && echo "HOLD_UDEV $stat_out" | log \
  || { echo "FAIL udev: $stat_out" | log; exit 1; }

echo "== live hold tests (H2: must really run, under the same ticket/lock) ==" | log
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/test_cpu_pd_hold.py" ]]; then
  cp "$SCRIPT_DIR/test_cpu_pd_hold.py" "$WORK/"
elif [[ -f "$SCRIPT_DIR/../../tests/test_cpu_pd_hold.py" ]]; then
  cp "$SCRIPT_DIR/../../tests/test_cpu_pd_hold.py" "$WORK/"
else
  echo "FAIL: stage tests/test_cpu_pd_hold.py next to this script" | log; exit 1
fi
# H3: the live tests run under the SAME ticket/lock as the A/B leg. The rc is
# captured, not piped under set -e: the strict pass/fail comes from the log
# rules below (a skipped run is a FAIL, not a crash).
set +e
runner sudo "$VENV/bin/python" "$WORK/test_cpu_pd_hold.py" -v 2>&1 | tee "$WORK/tests.log" | tee -a "$RECEIPT"
set -e
grep -q "Ran 2 tests" "$WORK/tests.log" || { echo "FAIL: tests did not run (Ran 2 tests missing)" | log; exit 1; }
grep -q "skipped" "$WORK/tests.log" && { echo "FAIL: tests were skipped" | log; exit 1; }
grep -qE "^OK$" "$WORK/tests.log" && echo "HOLD_TESTS 2 run, 0 skipped, OK" | log \
  || { echo "FAIL: no bare OK line" | log; exit 1; }

echo "== greedy A/B (H1/H4: token ids from generate_step, 4 digests) ==" | log
cat >"$WORK/ab.py" <<'EOF'
import hashlib, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

model, tokenizer = load(sys.argv[1])
prompt_ids = tokenizer.encode("Summarize the plot of Hamlet in one paragraph.")
prompt = mx.array(prompt_ids)
ids = [int(t) for t, _ in generate_step(prompt, model, max_tokens=64)]
digest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
print("AB_DIGEST", digest, "ntok", len(ids), sep=" ")
EOF
digests=()
arm=0
for envset in "" "MLX_OMARCHY_CPU_PD_HOLD=0"; do
  arm=$((arm+1))
  for rep in 1 2; do
    out="$(runner env $envset "$VENV/bin/python" "$WORK/ab.py" "$MODEL" 2>&1 | tee -a "$RECEIPT")"
    d="$(echo "$out" | grep '^AB_DIGEST ' | awk '{print $2}')"
    [[ -n "$d" ]] || { echo "FAIL arm $arm rep $rep: no digest ($out)" | log; exit 1; }
    digests+=("$d")
  done
done
echo "HOLD_AB 4 digests: ${digests[*]}" | log
uniq_count="$(printf '%s\n' "${digests[@]}" | sort -u | wc -l)"
[[ "$uniq_count" == 1 && "${#digests[@]}" == 4 ]] \
  && echo "HOLD_AB all 4 digests equal and non-empty" | log \
  || { echo "FAIL A/B digest mismatch: ${digests[*]}" | log; exit 1; }

echo "HOLD_PASS receipt=$RECEIPT" | log
