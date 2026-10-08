#!/usr/bin/env bash
# One-command M3/M4 tester kit for omarchy-mlx.
#
# Linux (aurora 12.3, M3):
#   1. aurora's own M3 bring-up report (--m3-report: read-only,
#      sanitized by aurora before saving),
#   2. the omarchy-mlx collector archive (capability, correctness,
#      benchmark, ANE state) plus its paste-ready submission note.
# macOS (M3/M4; M4 is macOS-only today):
#   1. the same collector, native: MLX/Metal benchmark numbers, ANE
#      ioreg/DeviceTree dump, ANE firmware hashes,
#   2. optional powermetrics one-shot (printed for you to run; sudo),
#   3. optional H16G compiler proof with MKIT_HWX=1.
#
# Nothing is uploaded. The default flow needs no root; the opt-in ANE
# stage needs root (it runs insmod). Review the printed manifest, then
# send with the printed --submit command.
# Why each step exists: scripts/m3m4_kit.md.
#
# Usage:
#   bash scripts/m3m4_kit.sh                # writes ./m3m4-kit-<UTC>/
#   AURORA_URL=... bash scripts/m3m4_kit.sh
#   MKIT_HWX=1 bash scripts/m3m4_kit.sh     # macOS only: H16G compile
#   MKIT_ANE_STAGE=t8122:1 bash scripts/m3m4_kit.sh   # Linux: one opt-in
#     ANE bring-up stage (root); parameters and refusals mirror the
#     omarchy-ane module (ane/h15, ane/h16). Bring-up evidence only:
#     no M3/M4 hardware support is claimed or implied.
#   MKIT_DRY_RUN=1 MKIT_ANE_STAGE=t8122:1 ...  # print every refusal
#     check result and the exact insmod command; run nothing
set -euo pipefail

AURORA_URL="${AURORA_URL:-https://github.com/iconidentify/aurora-linux/releases/latest/download/install-aurora-sep.sh}"
HWX_REPO="${HWX_REPO:-https://github.com/joshuaswarren/mil-hwx-compiler}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPTS_DIR")"
OUT="${OUT:-m3m4-kit-$(date -u +%Y%m%dT%H%M%SZ)}"
OS="$(uname -s)"
# MKIT_ANE_ROOT re-roots /proc, /etc and /sys for tests and chroot-style
# runs; unset or "/" (the default) means the real filesystem, as before.
MKIT_ANE_ROOT="${MKIT_ANE_ROOT:-}"
MKIT_ANE_ROOT="${MKIT_ANE_ROOT%/}"

detect_chip() {
  if [ "$OS" = "Linux" ]; then
    tr '\0' '\n' < "$MKIT_ANE_ROOT/proc/device-tree/compatible" 2>/dev/null |
      grep -E '^apple,t[0-9]+$' | head -1 || true
  else
    # macOS: the marketing chip name from the brand string.
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
  fi
}

CHIP_RAW="$(detect_chip)"
chip=""
case "${CHIP_RAW}" in
  apple,t8122 | Apple\ M3) chip="Apple M3" ;;
  apple,t6030 | Apple\ M3\ Pro) chip="Apple M3 Pro" ;;
  apple,t6031 | apple,t6034 | Apple\ M3\ Max) chip="Apple M3 Max" ;;
  apple,t8132 | Apple\ M4) chip="Apple M4" ;;
  apple,t6040 | Apple\ M4\ Pro) chip="Apple M4 Pro" ;;
  apple,t6041 | Apple\ M4\ Max) chip="Apple M4 Max" ;;
esac
if [ -z "$chip" ]; then
  echo "This is not an M3/M4 Mac (detected: ${CHIP_RAW:-unknown})."
  echo "The M3/M4 kit is for t8122/t6030/t6031/t6034 and t8132/t6040/t6041."
  echo "For M1/M2 data, run: python3 scripts/collect_deep.py --submit"
  exit 1
fi

# --- Optional ANE bring-up stage (opt-in, Linux only, needs root) ---
# MKIT_ANE_STAGE=<chip>:<stage> maps the chip to the omarchy-ane bring-up
# module and runs exactly one insmod stage. Module names, parameter names,
# stage names/numbers and the RESULT grammar are taken verbatim from
# omarchy-ane ane/h15 and ane/h16 (README-bringup.md + ane_h15_main.c /
# ane_h16_main.c at omarchy-ane main): h15 has optin/stage/ps_wait_ms/
# confirm_boot/fw_path and emits
#   ane_h15 RESULT stage=%u soc=%s verdict=%s reason=%s
# h16 has optin/stage/ps_wait_ms/boot_wait_ms/hello_wait_ms and emits the
# same line with ane_h16; stage 2 is not an h16 module stage (the ladder's
# fw-pin step runs in userspace), and the h15 wrapper stage (2) is refused
# by the module itself. The kit may only NARROW the module's refusals.
# Sources of truth: ane/h15/README-bringup.md, ane/h16/README-bringup.md.
ane_module_for_chip() {
  case "$1" in
    t8122 | t6030 | t6031 | t6034) echo "ane_h15" ;;
    t8132 | t6040 | t6041) echo "ane_h16" ;;
    *) echo "" ;;
  esac
}

# ane_want prints 0 when the condition holds, 1 when it fails (set -e safe).
ane_want() {
  if "$@" >/dev/null 2>&1; then echo 0; else echo 1; fi
}

ane_stage() {
  [ -n "${MKIT_ANE_STAGE:-}" ] || return 0
  echo "ANE stage requested (MKIT_ANE_STAGE=$MKIT_ANE_STAGE)."
  local root="$MKIT_ANE_ROOT"
  local modinfo="${MKIT_MODINFO:-modinfo}"
  local fwfetch="${MKIT_FW_FETCH:-omarchy-ane-firmware-fetch}"
  local dry=0
  [ "${MKIT_DRY_RUN:-0}" = "1" ] && dry=1
  local fails=0 soc="" module="" key="" ver="" loaded="" rc=0
  local chip_tok="${MKIT_ANE_STAGE%%:*}"
  local stage_tok="${MKIT_ANE_STAGE#*:}"

  # <name> <0=ok|other> <reason>; the first refusal fails closed unless
  # MKIT_DRY_RUN=1, which records every check result and stops.
  ane_check() {
    if [ "$2" = "0" ]; then
      echo "ANE check $1: PASS"
    else
      echo "ANE check $1: REFUSED ($3)"
      fails=$((fails + 1))
      if [ "$dry" != "1" ]; then exit 1; fi
    fi
  }

  if [ "$OS" != "Linux" ]; then
    echo "ANE stage: Linux only (insmod); run the kit on the Linux side of the box." >&2
    exit 1
  fi
  [ "$chip_tok" = "$MKIT_ANE_STAGE" ] && { echo "ANE stage: MKIT_ANE_STAGE must be <chip>:<stage> (got '$MKIT_ANE_STAGE')" >&2; exit 1; }

  module="$(ane_module_for_chip "$chip_tok")"
  ane_check chip-map "$(ane_want test -n "$module")" "unknown chip '$chip_tok' for ANE staging (t8122/t6030/t6031/t6034 -> ane_h15.ko, t8132/t6040/t6041 -> ane_h16.ko)"

  soc="$(tr '\0' '\n' < "$root/proc/device-tree/compatible" 2>/dev/null | grep -E '^apple,t[0-9]+$' | head -1 || true)"
  soc="${soc#apple,}"
  ane_check soc-match "$(ane_want [ "$soc" = "$chip_tok" ])" "MKIT_ANE_STAGE names '$chip_tok' but this machine reports '${soc:-unknown}'"

  case "$module:$stage_tok" in
    ane_h15:dt | ane_h15:0 | ane_h15:status | ane_h15:1 | ane_h15:boot | ane_h15:3 | \
      ane_h16:dt | ane_h16:0 | ane_h16:status | ane_h16:1 | ane_h16:boot | ane_h16:3) rc=0 ;;
    *) rc=1 ;;
  esac
  ane_check stage-valid $rc "stage '$stage_tok' is not a runnable $module stage (h15: dt/0, status/1, boot/3 — wrapper/2 exists but the module refuses it; h16: dt/0, status/1, boot/3 — 2 is not an h16 module stage)"

  if [ -n "${MKIT_ANE_KO:-}" ] && [ -r "$MKIT_ANE_KO" ] && [ "$(basename "$MKIT_ANE_KO")" = "$module.ko" ]; then rc=0; else rc=1; fi
  ane_check ko-file $rc "set MKIT_ANE_KO to the built $module.ko (the bring-up modules are not packaged and never autoload)"

  rc=1
  if [ -n "${MKIT_ANE_KO:-}" ] && [ -r "$MKIT_ANE_KO" ]; then
    ver="$("$modinfo" -F vermagic "$MKIT_ANE_KO" 2>/dev/null | head -1 || true)"
    if [ -n "$ver" ] && [ "${ver%% *}" = "$(uname -r)" ]; then rc=0; fi
  fi
  ane_check vermagic $rc "module vermagic '${ver:-unreadable}' does not match the running kernel '$(uname -r)'; rebuild the module against the running kernel headers"

  if [ "$module" = "ane_h16" ]; then key="ane-h16-experimental"; else key="ane-h15-experimental"; fi
  ane_check opt-in-key "$(ane_want grep -qx "$key" "$root/etc/omarchy-mac-boot/dtb-overlays.opt-in")" "add '$key' to $root/etc/omarchy-mac-boot/dtb-overlays.opt-in and reboot before staging"

  rc=0
  for loaded in ane ane_t6021 ane_h15 ane_h16; do
    if [ -d "$root/sys/module/$loaded" ]; then rc=1; break; fi
  done
  ane_check no-ane-module-loaded $rc "ANE module '${loaded:-?}' is already loaded; some stages refuse rmmod — reboot before staging"

  rc=0
  if [ "$module" = "ane_h16" ]; then
    if ! "$fwfetch" --check --root "$root" >/dev/null 2>&1; then rc=1; fi
  fi
  ane_check pinned-firmware $rc "omarchy-ane-firmware-fetch --check failed: the h16 module refuses any payload but the pinned leto/aether bytes (install via omarchy-ane packaging/omarchy-ane-firmware-fetch; h15 has no pin yet, which is why its boot stage needs MKIT_ANE_FW_PATH and refuses on unfilled facts)"

  rc=0
  case "$module:$stage_tok" in
    ane_h15:boot | ane_h15:3 | ane_h16:boot | ane_h16:3)
      [ "${MKIT_ANE_CONFIRM_BOOT:-0}" = "1" ] || rc=1 ;;
  esac
  ane_check boot-class-confirm $rc "'$stage_tok' is a boot-class stage (register writes; the CPU-release latch needs a reboot); rerun with MKIT_ANE_CONFIRM_BOOT=1"

  rc=0
  case "$module:$stage_tok" in
    ane_h15:boot | ane_h15:3)
      if [ -z "${MKIT_ANE_FW_PATH:-}" ] || [ ! -r "${MKIT_ANE_FW_PATH}" ]; then rc=1; fi ;;
  esac
  ane_check h15-fw-path $rc "ane_h15 stage=boot refuses without fw_path=<file under /lib/firmware>: no H15 pin exists; set MKIT_ANE_FW_PATH"

  local params=(optin="$soc" stage="$stage_tok")
  case "$module:$stage_tok" in
    ane_h15:boot | ane_h15:3) params+=(confirm_boot=1 fw_path="$MKIT_ANE_FW_PATH") ;;
    ane_h16:boot | ane_h16:3)
      if [ -n "${MKIT_ANE_HELLO_WAIT_MS:-}" ]; then params+=("hello_wait_ms=$MKIT_ANE_HELLO_WAIT_MS"); fi ;;
  esac
  echo "ANE insmod command: insmod $MKIT_ANE_KO ${params[*]}"

  if [ "$dry" = "1" ]; then
    if [ "$fails" -gt 0 ]; then
      echo "MKIT_DRY_RUN=1: $fails refusal(s) recorded above; nothing was executed." >&2
      exit 1
    fi
    echo "MKIT_DRY_RUN=1: every check passed; nothing was executed (no insmod, no dmesg, no hardware access)."
    exit 0
  fi

  echo "ANE stage: $module stage=$stage_tok on $soc (one shot per insmod; stop at the first unexpected result and keep the log)."
  mkdir -p "$OUT"
  local ane_out
  ane_out="$(cd "$OUT" && pwd)"
  if ! insmod "$MKIT_ANE_KO" "${params[@]}"; then
    dmesg 2>/dev/null | grep -E 'ane_h1[56]|RESULT' > "$ane_out/ane-stage.log" || true
    echo "insmod refused or failed; the module's RESULT/REFUSED lines are in $ane_out/ane-stage.log" >&2
    exit 1
  fi
  sleep 1
  if dmesg 2>/dev/null | grep -E 'ane_h1[56]|RESULT' > "$ane_out/ane-stage.log"; then
    echo "ANE RESULT line(s):"
    grep 'RESULT' "$ane_out/ane-stage.log" || true
  else
    : > "$ane_out/ane-stage.log"
    echo "no ane_h1[56]/RESULT lines found in dmesg (empty $ane_out/ane-stage.log; the absence is itself a result)" >&2
  fi
  echo "Send $ane_out/ane-stage.log (with the kit OUT/) to the ANE thread; nothing is uploaded."
  echo "Submission is a separate explicit step: python3 scripts/collect_deep.py --out $ane_out/mlx-omarchy-m3m4.tar --submit"
  return 0
}
ane_stage || exit 1

if [ "$OS" = "Linux" ] && [ "${chip#Apple M4}" != "$chip" ]; then
  echo "M4 on Linux: aurora does not boot M4 Macs yet; run this kit from"
  echo "macOS on the same box so we get the reference numbers."
  exit 1
fi
echo "M3/M4 kit: $chip (${CHIP_RAW}) on ${OS}; writing to $OUT/"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

collector_stage() {
  if (cd "$REPO_DIR" &&
      python3 scripts/collect_deep.py --out "$OUT/mlx-omarchy-m3m4.tar"); then
    echo "collector: OK ($OUT/mlx-omarchy-m3m4.tar)"
  else
    echo "collector did not complete (recorded above); continuing." >&2
  fi
}

if [ "$OS" = "Linux" ]; then
  # aurora's M3 bring-up report (their tool, their sanitizer).
  if (cd "$OUT" &&
      curl -fsSL "$AURORA_URL" -o install-aurora-sep.sh &&
      bash install-aurora-sep.sh --m3-report); then
    echo "aurora --m3-report: OK (tgz in $OUT/)"
  else
    echo "aurora --m3-report did not complete (recorded above); continuing." >&2
  fi
  collector_stage
  echo "(Optional, needs sudo, ~5 min, on power: aurora's power survey —"
  echo "  bash $OUT/install-aurora-sep.sh --m3-power-survey)"
else
  collector_stage
  echo "(Optional, needs sudo: one powermetrics sample for the GPU/ANE"
  echo " power channels:"
  echo "  sudo powermetrics --samplers gpu_power,ane_power -n 1 -o $OUT/powermetrics.txt)"
  if [ "${MKIT_HWX:-0}" = "1" ]; then
    echo "H16G compiler proof (mil-hwx-compiler):"
    if (git clone --depth 1 "$HWX_REPO" "$OUT/mil-hwx-compiler" &&
        make -C "$OUT/mil-hwx-compiler" mil-hwxc &&
        "$OUT/mil-hwx-compiler/build/mil-hwxc" \
          --mil "$OUT/mil-hwx-compiler/tests/fixtures/conv_relu.mil" \
          --model-root "$OUT/mil-hwx-compiler/tests/models/conv_relu" \
          --output "$OUT/hwx-conv-relu"); then
      shasum -a 256 "$OUT"/hwx-conv-relu/* 2>/dev/null |
        tee "$OUT/hwx-conv-relu/SHA256SUMS" || true
      echo "hwx: OK (program-0.hwx hashed in $OUT/hwx-conv-relu/)"
    else
      echo "hwx stage did not complete (recorded above); continuing." >&2
    fi
  else
    echo "(Optional H16G compiler proof: rerun with MKIT_HWX=1)"
  fi
fi

cat <<EOF

Next steps (nothing has been sent):
  1. Review $OUT/ contents and the printed redaction manifest.
  2. Send the collector archive:
       python3 scripts/collect_deep.py --out $OUT/mlx-omarchy-m3m4.tar --submit
     (the endpoint replies with a public receipt URL; identical content
     deduplicates by SHA-256).
  3. On Linux, also attach the aurora report tgz from $OUT/.
EOF
