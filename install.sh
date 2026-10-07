#!/usr/bin/env bash
# mlx-omarchy installer for Omarchy on Apple M1 (Asahi Linux, Honeykrisp Vulkan).
#
#   curl -fsSL https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/main/install.sh | bash
#   bash install.sh --ane
#   bash install.sh --voice
#   bash install.sh --uninstall
#
# The default install writes only under $HOME, except for runtime packages
# installed through pacman. --ane also provisions host-global ANE ownership.
# --voice adds the optional local speech dependencies (mlx-audio: speech
# input and read-aloud; text chat never needs them).

set -euo pipefail

# Name table of record for every install path: serve/mlx_omarchy_paths.py
# (packaging/paths.sh is generated from it and sourced by --system).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO=joshuaswarren/omarchy-mlx
PREFIX="${MLX_OMARCHY_HOME:-$HOME/.local/share/mlx-omarchy}"
VENV="$PREFIX/venv"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
MLX_LM_VERSION=0.31.3
TRANSFORMERS_VERSION=5.16.1
MLX_AUDIO_VERSION=0.5.6
ANE=0
VOICE=0
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# System-layout install (packaging lane). A PKGBUILD runs this in build()
# against vendored, hash-locked wheels and stages the /usr tree under
# --dest-root; package() copies the staged tree. Nothing here touches the
# network, pacman, systemd state, or the running machine's $HOME.
install_system() {
  local dest_root="" vendor="" lock="" serve_src="" python_bin="" import_smoke=0
  while (($#)); do
    case "$1" in
      --dest-root) dest_root="$2"; shift 2 ;;
      --vendor) vendor="$2"; shift 2 ;;
      --lock) lock="$2"; shift 2 ;;
      --serve-src) serve_src="$2"; shift 2 ;;
      --python) python_bin="$2"; shift 2 ;;
      --import-smoke) import_smoke=1; shift ;;
      *) die "unknown --system option: $1 (supported: --dest-root, --vendor, --lock, --serve-src, --python, --import-smoke)" ;;
    esac
  done
  if [[ -z $vendor || -z $lock ]]; then
    die "--system requires --vendor DIR and --lock FILE (vendored, hash-locked wheels)"
  fi
  if [[ -z $serve_src ]]; then
    serve_src="$ROOT/serve"
  fi
  [[ -d $vendor ]] || die "vendor directory not found: $vendor"
  [[ -f $lock ]] || die "lock file not found: $lock"
  [[ -d $serve_src ]] || die "serve sources not found: $serve_src"
  [[ -f "$ROOT/packaging/paths.sh" ]] || die "--system runs from a checkout; packaging/paths.sh is missing"
  command -v patch >/dev/null || die "patch is missing; the mlx-lm serve patches need it (add patch to makedepends)."

  # Every staged path comes from the generated name table.
  # shellcheck disable=SC1091
  source "$ROOT/packaging/paths.sh"

  local system_root="${dest_root%/}$SYSTEM_PREFIX"
  local venv="$system_root/$VENV_DIR"
  local bindir="${dest_root%/}/usr/bin"
  # Generated launcher/unit/desktop contents carry the FINAL system paths:
  # a PKGBUILD stages under --dest-root and copies the tree to / in
  # package(), so an embedded staging path would break at runtime.
  local final_venv="$SYSTEM_PREFIX/$VENV_DIR"
  local final_bindir="/usr/bin"
  say "Verifying vendored wheels against $lock"
  bash "$ROOT/packaging/verify-vendor.sh" "$vendor" "$lock"

  local build_args=(--vendor "$vendor" --lock "$lock" --venv "$venv")
  if [[ -n $python_bin ]]; then
    build_args+=(--python "$python_bin")
  fi
  say "Building $venv offline from vendored wheels"
  bash "$ROOT/packaging/build-venv.sh" "${build_args[@]}"

  # The venv was created at the staging absolute path, so its own bin
  # scripts (pip, activate, entry points) embed that path. A PKGBUILD
  # copies the tree to / in package(), which would orphan every one of
  # them; point them at the final venv before staging completes.
  local bin_script total nul
  while IFS= read -r -d '' bin_script; do
    [[ -f $bin_script && ! -L $bin_script ]] || continue
    total=$(head -c 4096 -- "$bin_script" | wc -c)
    nul=$(head -c 4096 -- "$bin_script" | LC_ALL=C tr -d '\0' | wc -c)
    [[ $total == "$nul" ]] || continue  # NUL bytes: leave binaries alone
    grep -qF -- "$venv" "$bin_script" || continue
    sed -i "s|$venv|$final_venv|g" -- "$bin_script"
  done < <(find "$venv/bin" -maxdepth 1 -type f -print0)

  local site_dir
  site_dir="$(printf '%s\n' "$venv"/lib/python3.*/site-packages)"
  [[ -d $site_dir ]] || die "no site-packages under $venv"

  # The staged share tree becomes the package. A recipe (or a human) that
  # swaps ANE bytes into it after the wheel install — a locally rebuilt
  # libane-strict.so or worker — must fail HERE, not ship a package whose
  # worker seal refuses its own pin at first transcribe.
  say "Verifying staged runtime assets against the wheel pin"
  python3 "$ROOT/scripts/verify_runtime_assets.py" \
    "$site_dir/mlx/share/mlx-omarchy/parakeet-1"
  local pkg
  for pkg in $SERVE_PACKAGES; do
    [[ -d "$serve_src/$pkg" ]] || die "serve package missing: $serve_src/$pkg"
    mkdir -p "$site_dir/$pkg"
    cp -a "$serve_src/$pkg/." "$site_dir/$pkg/"
  done
  cp "$ROOT/serve/mlx_omarchy_paths.py" "$site_dir/mlx_omarchy_paths.py"

  if [[ -d "$site_dir/mlx_lm" ]]; then
    say "Applying mlx-lm serve patches (conv-ring off unless MLX_OMARCHY_CONV_RING=1)"
    : "${MLX_OMARCHY_CONV_RING:=0}"
    bash "$ROOT/scripts/apply-mlx-lm-patches.sh" "$venv"
  else
    echo "note: mlx-lm is not in this venv; skipped the mlx-lm serve patches (every real vendor lock carries mlx-lm==0.31.3)"
  fi

  say "Installing launchers into $bindir"
  mkdir -p "$bindir" "$system_root"
  cp "$ROOT/demo/chat.py" "$system_root/chat.py"
  cat >"$bindir/mlx-omarchy" <<EOF
#!/usr/bin/env bash
exec "$final_venv/bin/python" "\$@"
EOF
  cat >"$bindir/mlx-omarchy-demo" <<EOF
#!/usr/bin/env bash
exec "$final_venv/bin/python" "$SYSTEM_PREFIX/chat.py" "\$@"
EOF
  cat >"$bindir/mlx-omarchy-chat" <<EOF
#!/usr/bin/env bash
exec "$final_venv/bin/python" -m mlx_omarchy_assistant "\$@"
EOF
  cat >"$bindir/mlx-omarchy-serve" <<EOF
#!/usr/bin/env bash
exec "$final_venv/bin/python" -m mlx_omarchy_serve "\$@"
EOF
  cat >"$bindir/omarchy-mlx-serve" <<EOF
#!/usr/bin/env bash
# omarchy:group=mlx
# omarchy:name=serve
# omarchy:summary=Serve a vetted local model on the Apple GPU (memory-checked, approve-first)
# omarchy:args=[target] [--context N] [--server mlx-lm|omlx] [--host H] [--port P] [--yes]
# omarchy:examples=omarchy mlx serve | omarchy mlx serve recommend | omarchy mlx serve plan <model> | omarchy mlx serve catalog list
exec "$final_venv/bin/python" -m mlx_omarchy_serve "\$@"
EOF
  chmod +x "$bindir/mlx-omarchy" "$bindir/mlx-omarchy-demo" "$bindir/mlx-omarchy-chat" \
    "$bindir/mlx-omarchy-serve" "$bindir/omarchy-mlx-serve"
  local info info_final
  if info="$("$venv/bin/python" -I -c 'import os, mlx
print(next(p for root in mlx.__path__
           if os.access(p := os.path.join(root, "bin", "mlx-omarchy-info"), os.X_OK)))' 2>/dev/null)"; then
    info_final="${info#"$dest_root"}"
    printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$info_final" >"$bindir/mlx-omarchy-info"
    chmod +x "$bindir/mlx-omarchy-info"
  else
    die "the installed wheel provides no mlx/bin/mlx-omarchy-info; cannot stage the info launcher"
  fi
  # The parakeet CLI is a data file with an `env python3` shebang; run it
  # through this venv's interpreter so the vendored numpy/protobuf are
  # the ones that load. aarch64 wheels only — absent elsewhere by design.
  if para="$("$venv/bin/python" -I -c 'import os, mlx
print(next(p for root in mlx.__path__
           if os.access(p := os.path.join(root, "bin", "mlx-omarchy-parakeet"), os.X_OK)))' 2>/dev/null)"; then
    para_final="${para#"$dest_root"}"
    printf '#!/usr/bin/env bash\nexec "%s/bin/python" %q "$@"\n' \
      "$final_venv" "$para_final" >"$bindir/mlx-omarchy-parakeet"
    chmod +x "$bindir/mlx-omarchy-parakeet"
  else
    say "note: no mlx/bin/mlx-omarchy-parakeet in this wheel (aarch64-only runtime); launcher skipped"
  fi
  install -m 755 "$ROOT/packaging/mlx-omarchy-retire-legacy" "$bindir/$RETIRE_CMD"
  local sharedir="${dest_root%/}$SYSTEM_SHARE_PREFIX"
  install -d -m 755 "$sharedir"
  install -m 644 "$ROOT/packaging/paths.sh" "$sharedir/paths.sh"

  # PM-QoS access for the CPU deep-idle hold (overlay/.../cpu_pd_hold.h).
  local udevdir="${dest_root%/}/usr/lib/udev/rules.d"
  install -d -m 755 "$udevdir"
  install -m 644 "$ROOT/packaging/udev/70-omarchy-mlx-cpu-dma-latency.rules" \
    "$udevdir/70-omarchy-mlx-cpu-dma-latency.rules"

  local unitdir="${dest_root%/}/usr/lib/systemd/user"
  mkdir -p "$unitdir"
  cat >"$unitdir/$UNIT_NAME" <<EOF
[Unit]
Description=MLX Chat resident pair
After=default.target

[Service]
ExecStart=$final_bindir/mlx-omarchy-chat --resume --no-browser
Restart=on-failure
RestartSec=15

[Install]
WantedBy=default.target
EOF

  local appsdir="${dest_root%/}/usr/share/applications"
  mkdir -p "$appsdir"
  cat >"$appsdir/mlx-omarchy-chat.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MLX Chat (Apple GPU)
Comment=Local chat and typed decisions on the Apple GPU via mlx-omarchy
Exec=$final_bindir/mlx-omarchy-chat
Icon=applications-internet
Categories=Development;Utility;
EOF

  local docdir="${dest_root%/}/usr/share/doc/$HOME_NAME"
  mkdir -p "$docdir"
  cp "$ROOT/README.md" "$docdir/README.md"

  if (( import_smoke )); then
    say "Import smoke"
    "$venv/bin/python" -c 'import mlx.core, mlx_omarchy_paths, mlx_omarchy_serve'
  fi

  say "System tree staged under ${dest_root:-/}"
  echo "  venv:      $venv"
  echo "  launchers: $bindir/{mlx-omarchy,mlx-omarchy-demo,mlx-omarchy-chat,mlx-omarchy-serve,mlx-omarchy-info,mlx-omarchy-parakeet,$RETIRE_CMD,omarchy-mlx-serve}"
  echo "  unit:      $unitdir/$UNIT_NAME (staged only; never enabled at build time)"
  echo "  desktop:   $appsdir/mlx-omarchy-chat.desktop"
  echo "  packaging: the recipe declares pacman depends=(python=$PYTHON_VERSION openblas lapack blas)"
}

case "${1:-}" in
  --ane) ANE=1 ;;
  --voice) VOICE=1 ;;
  --system)
    install_system "${@:2}"
    exit 0
    ;;
  --uninstall)
    if command -v systemctl >/dev/null 2>&1; then
      systemctl --user disable --now mlx-omarchy-chat.service >/dev/null 2>&1 || true
    fi
    rm -rf "$PREFIX" "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo" "$BIN/mlx-omarchy-chat" "$BIN/mlx-omarchy-info" \
      "$BIN/mlx-omarchy-serve" "$BIN/omarchy-mlx-serve" \
      "$APPS/mlx-omarchy-chat.desktop" "$APPS/mlx-omarchy-demo.desktop" \
      "$HOME/.config/systemd/user/mlx-omarchy-chat.service"
    if command -v omarchy >/dev/null 2>&1; then
      omarchy_target="$(dirname "$(command -v omarchy)")/omarchy-mlx-serve"
      if [[ -w "$(dirname "$omarchy_target")" ]] && grep -qs 'mlx_omarchy_serve' "$omarchy_target" 2>/dev/null; then
        rm -f "$omarchy_target"
      fi
    fi
    say "mlx-omarchy removed. Model downloads stay in ~/.cache/huggingface; delete them yourself if you want the space back."
    exit 0
    ;;
  "") ;;
  *) die "unknown option: $1 (supported: --ane, --voice, --system, --uninstall)" ;;
esac

# 1. Hardware and interpreter checks. The release wheel is cp314 linux_aarch64
#    and is supported on the M1 family (t8103, t6000, t6001, t6002 — same GPU
#    generation and driver path; per-chip measurements live in the README)
#    and M2 Max (t6021).
#    The ANE gate runs BEFORE any network access: a missing device is a
#    local fact and must refuse the install without depending on the
#    GitHub API (rate-limited runners otherwise see the release-resolution
#    error instead of the device refusal).
if (( ANE )); then
  [[ -c /dev/accel/accel0 ]] || die "ANE installation requires /dev/accel/accel0."
  command -v sudo >/dev/null || die "ANE installation requires sudo."
  command -v systemd-tmpfiles >/dev/null || die "ANE installation requires systemd-tmpfiles."
fi

# MLX_OMARCHY_VERSION pins a release explicitly; otherwise the latest published
# release is resolved from the GitHub API (unauthenticated limit: 60 req/h/IP).
if [[ -n "${MLX_OMARCHY_VERSION:-}" ]]; then
  VERSION=$MLX_OMARCHY_VERSION
  say "Installing mlx-omarchy $VERSION (pinned by MLX_OMARCHY_VERSION)"
else
  say "Resolving the latest mlx-omarchy release"
  VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])' 2>/dev/null) ||
    die "could not resolve the latest release from api.github.com (offline, or the unauthenticated 60 req/h limit is exhausted); install a known version with MLX_OMARCHY_VERSION=v0.7.1"
  say "Installing mlx-omarchy $VERSION"
fi
[[ "$(uname -m)" == aarch64 ]] || die "mlx-omarchy runs on Apple Silicon (aarch64); this machine is $(uname -m)."
if [[ -r /proc/device-tree/compatible ]] &&
   ! tr '\0' ' ' </proc/device-tree/compatible | grep -qE 'apple,t(8103|6000|6001|6002|6021)'; then
  echo "warning: this SoC is outside mlx-omarchy's supported list (M1 family t8103/t6000/t6001/t6002, M2 Max t6021); it is untested here." >&2
fi
command -v python3 >/dev/null || die "python3 is missing."
command -v patch >/dev/null || die "patch is missing; the mlx-lm serve patches need it (e.g. pacman -S patch)."
python3 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))' \
  || die "Python 3.14 is required (found $(python3 --version)); the wheel is built for cp314."

# 2. Runtime packages. omarchy-pkg-add is Omarchy's own helper; plain pacman
#    is the fallback on any other Asahi Arch install.
# openblas provides libopenblas.so.0, which the wheel's BLAS calls resolve
# against at import time. Without it `import mlx.core` fails on a fresh install.
say "Installing runtime packages (lapack, blas, openblas)"
if command -v omarchy-pkg-add >/dev/null; then
  omarchy-pkg-add lapack blas openblas
else
  sudo pacman -S --needed --noconfirm lapack blas openblas
fi
# Voice playback (--voice) needs the PortAudio shared library: sounddevice
# ships no Linux binary and loads the system libportaudio.so.2. Text chat
# never needs it.
if (( VOICE )); then
  say "Installing voice runtime package (portaudio)"
  if command -v omarchy-pkg-add >/dev/null; then
    omarchy-pkg-add portaudio
  else
    sudo pacman -S --needed --noconfirm portaudio
  fi
fi

# 3. Download the release wheel and verify it against the SHA256SUMS asset.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
base="${MLX_OMARCHY_RELEASE_BASE:-https://github.com/$REPO/releases/download/$VERSION}"
say "Fetching $VERSION checksums"
curl -fsSL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS"
wheel="$(grep -o 'mlx_omarchy-[^ ]*cp314-cp314-linux_aarch64\.whl' "$tmp/SHA256SUMS" | head -n 1)"
[[ -n "$wheel" ]] || die "no aarch64 wheel listed in $base/SHA256SUMS"
say "Fetching $wheel"
curl -fsSL "$base/$wheel" -o "$tmp/$wheel"
(cd "$tmp" && sha256sum -c --ignore-missing --quiet SHA256SUMS) || die "checksum mismatch for $wheel"

if (( ANE )); then
  ane_conf="$tmp/mlx-omarchy-ane.conf"
  python3 - "$tmp/$wheel" "$ane_conf" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel, destination = sys.argv[1:]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    configs = [name for name in names if name.endswith("lib/tmpfiles.d/mlx-omarchy-ane.conf")]
    workers = [name for name in names if name.endswith("bin/mlx-omarchy-ane-worker")]
    if len(configs) != 1 or len(workers) != 1:
        raise SystemExit("wheel does not contain exactly one ANE worker and tmpfiles policy")
    Path(destination).write_bytes(archive.read(configs[0]))
PY
  say "Provisioning host-global ANE ownership"
  sudo install -D -m0644 "$ane_conf" /usr/lib/tmpfiles.d/mlx-omarchy-ane.conf
  sudo systemd-tmpfiles --create /usr/lib/tmpfiles.d/mlx-omarchy-ane.conf

  sudo python3 - <<'PY'
import os
import stat

device = os.stat("/dev/accel/accel0", follow_symlinks=False)
if not stat.S_ISCHR(device.st_mode):
    raise SystemExit("/dev/accel/accel0 is not a character device")
expected = (
    ("/run/lock/mlx-omarchy-ane", stat.S_ISDIR, 0o1777, False),
    ("/run/lock/mlx-omarchy-ane/device.lock", stat.S_ISREG, 0o666, True),
    ("/run/lock/mlx-omarchy-ane/quarantine", stat.S_ISREG, 0o666, True),
)
for path, type_check, mode, single_link in expected:
    status = os.stat(path, follow_symlinks=False)
    valid = (
        type_check(status.st_mode)
        and status.st_uid == 0
        and stat.S_IMODE(status.st_mode) == mode
        and (not single_link or status.st_nlink == 1)
    )
    if not valid:
        raise SystemExit(f"invalid ANE ownership path: {path}")
    print(f"  {status.st_uid}:{status.st_gid} {mode:o} {path}")
PY
fi

# 4. Private venv. Nothing is installed into the system Python.
say "Creating $VENV"
mkdir -p "$PREFIX" "$BIN" "$APPS"
python3 -m venv --clear "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$tmp/$wheel"
# mlx-lm declares a dependency on upstream mlx, which provides the same module
# and would conflict, so it is installed without dependencies and its real
# runtime dependencies are pinned explicitly.
"$VENV/bin/pip" install --quiet --no-deps "mlx-lm==$MLX_LM_VERSION"
# soundfile: the Parakeet CLI decodes its FLAC fixture with it; without it the CLI
# spawns ffprobe/ffmpeg three times (measured on an M1: audio_load 181 ms vs 4.7 ms,
# total_pipeline_ms 567 vs 363 ms, transcript sha unchanged).
"$VENV/bin/pip" install --quiet "transformers[sentencepiece]==$TRANSFORMERS_VERSION" numpy protobuf pyyaml jinja2 huggingface_hub soundfile

# 4a. Optional voice dependencies (--voice), used by the assistant's local
#     speech input and synthesis. mlx-audio declares an mlx requirement (>= 0.31.1)
#     that the custom wheel installed above satisfies, so it goes in with
#     --no-deps: letting pip resolve it would pull upstream mlx over the
#     vendored mlx-omarchy build. The remaining lines are the verified
#     mlx-audio $MLX_AUDIO_VERSION runtime floors, extras excluded;
#     transformers, numpy, and huggingface_hub are usually already
#     satisfied by the base pins above.
if (( VOICE )); then
  say "Installing voice dependencies (mlx-audio $MLX_AUDIO_VERSION, no deps)"
  "$VENV/bin/pip" install --quiet --no-deps "mlx-audio==$MLX_AUDIO_VERSION"
  # The default speech engine is Kokoro-82M (pins mirror KOKORO_PACK's
  # runtime.requires/constraints in serve/mlx_omarchy_assistant/synthesis.py;
  # tests/test_install_sh_contract.py enforces the mirror). misaki's English
  # G2P runs through the espeakng-loader user-space wheel (libespeak-ng +
  # data, no root); en_core_web_sm ships as a GitHub release wheel because
  # it has no PyPI package.
  "$VENV/bin/pip" install --quiet "huggingface_hub>=1.0" "miniaudio>=1.61" "numpy>=1.26.4" \
    "scipy>=1.10.0" "sounddevice>=0.5.3" "tqdm>=4.67.1" "transformers>=5.14.0" \
    "misaki==0.7.4" "num2words==0.5.14" "spacy>=3.8.0" "phonemizer>=3.2.1" \
    "espeakng-loader==0.2.4"
  "$VENV/bin/pip" install --quiet \
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
fi

# 4b. Vendored mlx-lm serve patches. The apply script decides which patches
#     run (conv-ring stays OFF unless MLX_OMARCHY_CONV_RING=1). The installer
#     downloads exactly the patches that script names, so a patch added to the
#     script can never be missing here. Served models pick them up from the venv.
say "Applying mlx-lm serve patches (conv-ring off unless MLX_OMARCHY_CONV_RING=1)"
curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/scripts/apply-mlx-lm-patches.sh" -o "$PREFIX/apply-mlx-lm-patches.sh"
mkdir -p "$PREFIX/patches"
for p in $(grep -oE 'mlx-lm-[a-z0-9-]+\.patch' "$PREFIX/apply-mlx-lm-patches.sh" | sort -u); do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/patches/$p" -o "$PREFIX/patches/$p"
done
# The apply script also runs helper scripts from the repo (scripts/*.py).
# Download exactly those, by the same derive-from-the-script contract: a
# helper the script names can never be missing here.
mkdir -p "$PREFIX/scripts"
for s in $(grep -oE 'scripts/[a-z0-9_-]+\.py' "$PREFIX/apply-mlx-lm-patches.sh" | sort -u); do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/$s" -o "$PREFIX/$s"
done
MLX_OMARCHY_CONV_RING="${MLX_OMARCHY_CONV_RING:-0}" bash "$PREFIX/apply-mlx-lm-patches.sh" "$VENV"

# 5. Demo and launchers.
say "Installing launchers into $BIN"
curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/demo/chat.py" -o "$PREFIX/chat.py"
cat >"$BIN/mlx-omarchy" <<EOF
#!/usr/bin/env bash
# Python interpreter with mlx-omarchy and mlx-lm installed.
exec "$VENV/bin/python" "\$@"
EOF
cat >"$BIN/mlx-omarchy-demo" <<EOF
#!/usr/bin/env bash
# Terminal face of MLX Chat: attaches to the shared coordinator (see 5d).
export PYTHONPATH="$PREFIX\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$VENV/bin/python" "$PREFIX/chat.py" "\$@"
EOF
cat >"$BIN/mlx-omarchy-chat" <<EOF
#!/usr/bin/env bash
# MLX Chat web application: loopback coordinator + local UI, opens the browser.
export PYTHONPATH="$PREFIX\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$VENV/bin/python" -m mlx_omarchy_assistant "\$@"
EOF
INFO=$("$VENV/bin/python" -I -c 'import os, mlx
print(next(p for root in mlx.__path__
           if os.access(p := os.path.join(root, "bin", "mlx-omarchy-info"), os.X_OK)))')
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$INFO" >"$BIN/mlx-omarchy-info"
chmod +x "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo" "$BIN/mlx-omarchy-chat" "$BIN/mlx-omarchy-info"

# Parakeet CLI through this venv's interpreter: the wheel ships it as a
# data file with an `env python3` shebang, which would otherwise bind to
# the system python (no numpy/protobuf). aarch64 wheels only.
if PARA=$("$VENV/bin/python" -I -c 'import os, mlx
print(next(p for root in mlx.__path__
           if os.access(p := os.path.join(root, "bin", "mlx-omarchy-parakeet"), os.X_OK)))' 2>/dev/null); then
  printf '#!/usr/bin/env bash\nexec %q %q "$@"\n' "$VENV/bin/python" "$PARA" >"$BIN/mlx-omarchy-parakeet"
  chmod +x "$BIN/mlx-omarchy-parakeet"
else
  say "note: no mlx/bin/mlx-omarchy-parakeet in this wheel (aarch64-only runtime); launcher skipped"
fi

# 5b. Serve CLI: catalog-driven serving with memory admission and an
#     approve-first download gate. The package ships from the same release
#     tag as the wheel, so an install is internally consistent.
say "Installing serve CLI"
# Shared name module first: venv discovery and the install name table,
# imported by the serve CLI, the assistant, and the retire tooling.
curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_paths.py" \
  -o "$PREFIX/mlx_omarchy_paths.py"
SERVE_PKG="$PREFIX/mlx_omarchy_serve"
mkdir -p "$SERVE_PKG"
for serve_file in __init__.py catalog.py budget.py perf_placement.py export_fit_table.py __main__.py _mlxlm_server.py catalog.json; do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_serve/$serve_file" \
    -o "$SERVE_PKG/$serve_file"
done
# Laya typed-decisions server (serve/mlx_omarchy_laya/), same release tag;
# its catalog entries route to it via serve.backend == "module".
LAYA_PKG="$PREFIX/mlx_omarchy_laya"
mkdir -p "$LAYA_PKG"
for laya_file in __init__.py model.py sequence.py api.py server.py convert.py qualify.py; do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_laya/$laya_file" \
    -o "$LAYA_PKG/$laya_file"
done
# Bonsai2 packed-runtime server (serve/mlx_omarchy_bonsai2/), same release
# tag; catalog entries route to it via serve.backend == "module".
BONSAI2_PKG="$PREFIX/mlx_omarchy_bonsai2"
mkdir -p "$BONSAI2_PKG"
for bonsai2_file in __init__.py packed.py loader.py server.py; do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_bonsai2/$bonsai2_file" \
    -o "$BONSAI2_PKG/$bonsai2_file"
done
cat >"$BIN/mlx-omarchy-serve" <<EOF
#!/usr/bin/env bash
# Serve CLI: recommend/plan/serve/catalog; memory-admitted, approve-first.
export PYTHONPATH="$PREFIX\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$VENV/bin/python" -m mlx_omarchy_serve "\$@"
EOF
chmod +x "$BIN/mlx-omarchy-serve"

# 5c. Omarchy command-center integration: our own self-describing binary in
#     the command center's conventions (omarchy:group metadata header). This
#     never modifies an Omarchy-owned file; it adds ours next to them.
#     An existing non-mlx-omarchy binary is never overwritten.
write_omarchy_serve_launcher() {
  local dest="$1"
  cat >"$dest" <<OMLX
#!/usr/bin/env bash
# omarchy:group=mlx
# omarchy:name=serve
# omarchy:summary=Serve a vetted local model on the Apple GPU (memory-checked, approve-first)
# omarchy:args=[target] [--context N] [--server mlx-lm|omlx] [--host H] [--port P] [--yes]
# omarchy:examples=omarchy mlx serve | omarchy mlx serve recommend | omarchy mlx serve plan <model> | omarchy mlx serve catalog list
export PYTHONPATH="$PREFIX\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$VENV/bin/python" -m mlx_omarchy_serve "\$@"
OMLX
  chmod +x "$dest"
}
if command -v omarchy >/dev/null 2>&1; then
  omarchy_bin_dir="$(dirname "$(command -v omarchy)")"
  omarchy_target="$omarchy_bin_dir/omarchy-mlx-serve"
  if [[ -e "$omarchy_target" ]] &&
     ! grep -qs 'mlx_omarchy_serve' "$omarchy_target"; then
    echo "note: $omarchy_target exists and is not mlx-omarchy's; left untouched."
    write_omarchy_serve_launcher "$BIN/omarchy-mlx-serve"
    echo "note: 'omarchy mlx serve' routes only from $omarchy_bin_dir; until then use: mlx-omarchy-serve"
  elif [[ -w "$omarchy_bin_dir" ]]; then
    write_omarchy_serve_launcher "$omarchy_target"
  else
    write_omarchy_serve_launcher "$BIN/omarchy-mlx-serve"
    echo "note: to register 'omarchy mlx serve' in the system command center, run:"
    echo "  sudo install -m 755 $BIN/omarchy-mlx-serve $omarchy_target"
    echo "note: mlx-omarchy-serve works right now without that step."
  fi
fi

# 5d. MLX Chat assistant: the local web application — static UI plus the
#     shared conversation coordinator — that both the desktop entry and the
#     terminal demo attach to. Same release tag as the wheel, so an install
#     is internally consistent. Voice assets are NOT shipped: they download
#     approve-first at runtime from the pinned revision recorded in the
#     synthesis module's manifest.
say "Installing MLX Chat"
ASSISTANT_PKG="$PREFIX/mlx_omarchy_assistant"
mkdir -p "$ASSISTANT_PKG/static/css" "$ASSISTANT_PKG/static/js/worklet"
for assistant_file in __init__.py __main__.py card_promotion.py routing.py coordinator.py history.py server.py pairs.py managed.py transfer.py components.py theme.py recognition.py synthesis.py kokoro_stream.py kokoro_gen_stats.npz speech_yield.py gpu_stt.py gpu_stt_worker.py; do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_assistant/$assistant_file" \
    -o "$ASSISTANT_PKG/$assistant_file"
done
for assistant_static in index.html css/app.css js/api.js js/app.js js/chat.js js/composer.js js/dom.js js/genui.js js/markdown.js js/setup.js js/theme.js js/transfer.js js/util.js js/voice.js js/worklet/capture-worklet.js; do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/serve/mlx_omarchy_assistant/static/$assistant_static" \
    -o "$ASSISTANT_PKG/static/$assistant_static"
done

# Desktop entry: MLX Chat opens the same loopback web application the
# launcher starts — no terminal wrapper, no model dialog. The coordinator
# mints a one-use session URL per launch. Upgrades from the old terminal
# entry must not leave it behind in the launcher.
rm -f "$APPS/mlx-omarchy-demo.desktop"
cat >"$APPS/mlx-omarchy-chat.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MLX Chat (Apple GPU)
Comment=Local chat and typed decisions on the Apple GPU via mlx-omarchy
Exec=$BIN/mlx-omarchy-chat
Icon=applications-internet
Categories=Development;Utility;
EOF

# Login service: after reboot, load the saved pair before the first message.
# A reboot clears GPU memory. This starts the saved pair once at login.
# No saved pair exits immediately. Enable only for a real user home so a
# test install cannot enable a service on the machine running the test.
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
cat >"$UNIT_DIR/mlx-omarchy-chat.service" <<EOF
[Unit]
Description=MLX Chat resident pair
After=default.target

[Service]
ExecStart=$BIN/mlx-omarchy-chat --resume --no-browser
Restart=on-failure
RestartSec=15

[Install]
WantedBy=default.target
EOF
REAL_HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
if [ "$HOME" = "$REAL_HOME" ] && command -v systemctl >/dev/null 2>&1; then
  systemctl --user daemon-reload >/dev/null 2>&1 || true
  systemctl --user enable mlx-omarchy-chat.service >/dev/null 2>&1 || true
fi
if command -v omarchy-menu >/dev/null 2>&1; then
  # The Omarchy shell scans desktop entries at startup; ask it to rescan so
  # the entry shows up in the launcher (Super+Space) without a re-login.
  omarchy-menu refresh >/dev/null 2>&1 || true
fi

# 6. Smoke test on the real GPU: import, device, one matmul. Always runs.
say "Smoke test"
"$VENV/bin/python" - <<'EOF'
import mlx.core as mx
info = mx.device_info()
a = mx.random.normal((256, 256))
b = mx.random.normal((256, 256))
c = (a @ b).sum()
mx.eval(c)
assert mx.isfinite(c).item(), "matmul produced a non-finite result"
print(f"  device: {info.get('device_name', info)}")
print(f"  mlx-omarchy {mx.__version__}: matmul OK")
EOF

# 7. ANE smoke only when the accelerator node exists. Missing accel0 is a
#    GPU-only success. Present accel0 without an FDT ANE node or loaded
#    ane module refuses the install. This never installs kmod-ane.
if [[ -c "${MLX_OMARCHY_ACCEL_DEV:-/dev/accel/accel0}" ]]; then
  say "ANE smoke"
  "$VENV/bin/python" - <<'ANE_SMOKE' || die "ANE smoke failed"
import os
import stat
import sys

accel = os.environ.get("MLX_OMARCHY_ACCEL_DEV", "/dev/accel/accel0")
sysroot = os.environ.get("MLX_OMARCHY_SYSROOT", "/")
try:
    status = os.stat(accel, follow_symlinks=False)
except OSError as exc:
    raise SystemExit(f"ANE smoke: cannot stat {accel}: {exc}") from exc
if not stat.S_ISCHR(status.st_mode):
    raise SystemExit("ANE smoke: accel0 is not a character device")
dt = (
    "/sys/firmware/devicetree/base"
    if sysroot in ("", "/")
    else os.path.join(sysroot, "sys/firmware/devicetree/base")
)
# The loaded ANE driver registers as `ane` on the M1 family and as
# `ane_t6021` on the M2 Max (omarchy-ane installs a per-chip module there).
module = next(
    (
        m
        for m in (
            os.path.join(sysroot, "sys/module/ane"),
            os.path.join(sysroot, "sys/module/ane_t6021"),
        )
        if os.path.isdir(m)
    ),
    None,
)

def ane_fdt(base):
    matches = []
    if not os.path.isdir(base):
        return False, None
    for dirpath, _dirs, _files in os.walk(base):
        name = os.path.basename(dirpath)
        named = name == "ane" or name.startswith("ane@")
        tokens = []
        try:
            with open(os.path.join(dirpath, "compatible"), "rb") as fh:
                tokens = [t.decode("utf-8", "replace") for t in fh.read().split(b"\0") if t]
        except OSError:
            pass
        hit = [t for t in tokens if t == "apple,ane" or t.endswith("-ane")]
        if named or hit:
            matches.extend(hit or tokens or [name])
    if not matches:
        return False, None
    return True, sorted(set(matches))[:8]

fdt_node, compatible = ane_fdt(dt)
module_present = module is not None
version = None
if module_present:
    try:
        with open(os.path.join(module, "version"), encoding="utf-8") as fh:
            version = fh.read().strip() or None
    except OSError:
        version = None
print(
    "  ANE fdt: "
    + ("yes" if fdt_node else "no")
    + " compatible="
    + (",".join(compatible or []) or "none")
)
print("  ANE module: " + ("yes" if module_present else "no") + (f" ({version})" if version else ""))
print("  ANE accel0: yes")
if not (fdt_node and module_present):
    raise SystemExit("ANE smoke failed: missing FDT node or ane module")
print("  ANE smoke OK")
ANE_SMOKE
else
  echo "  ANE: unavailable (no /dev/accel/accel0); GPU-only install"
fi

say "Done."
echo "  Open MLX Chat:       mlx-omarchy-chat      (also in the Omarchy app launcher as 'MLX Chat')"
echo "  Terminal chat:       mlx-omarchy-demo      (the same local coordinator, no browser)"
echo "  Use in your scripts: mlx-omarchy your_script.py   (import mlx.core as mx)"
echo "  Serve a model:       mlx-omarchy-serve     (also 'omarchy mlx serve' when registered)"
echo "  Voice:               bash install.sh --voice   (adds local speech input and read-aloud deps)"
echo "  Remove everything:   bash install.sh --uninstall"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "  note: $BIN is not on your PATH in this shell; open a new terminal or add it." ;; esac
