#!/usr/bin/env bash
# mlx-omarchy installer for Omarchy on Apple M1 (Asahi Linux, Honeykrisp Vulkan).
#
#   curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
#   bash install.sh --ane
#   bash install.sh --uninstall
#
# The default install writes only under $HOME, except for runtime packages
# installed through pacman. --ane also provisions host-global ANE ownership.
set -euo pipefail

REPO=joshuaswarren/mlx-omarchy
PREFIX="${MLX_OMARCHY_HOME:-$HOME/.local/share/mlx-omarchy}"
VENV="$PREFIX/venv"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
MLX_LM_VERSION=0.31.3
TRANSFORMERS_VERSION=5.16.1
ANE=0
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  --ane) ANE=1 ;;
  --uninstall)
    rm -rf "$PREFIX" "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo" "$BIN/mlx-omarchy-info" \
      "$BIN/mlx-omarchy-serve" "$BIN/omarchy-mlx-serve" "$APPS/mlx-omarchy-demo.desktop"
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
  *) die "unknown option: $1 (supported: --ane, --uninstall)" ;;
esac

# 1. Hardware and interpreter checks. The release wheel is cp314 linux_aarch64
#    and is verified on the M1 family (t8103, t6000, t6001, t6002) and
#    M2 Max (t6021).
#    The ANE gate runs BEFORE any network access: a missing device is a
#    local fact and must refuse the install without depending on the
#    GitHub API (rate-limited runners otherwise see the release-resolution
#    error instead of the device refusal).
if (( ANE )); then
  [[ -c /dev/accel/accel0 ]] || die "ANE installation requires /dev/accel/accel0."
  command -v sudo >/dev/null || die "ANE installation requires sudo."
  command -v systemd-tmpfiles >/dev/null || die "ANE installation requires systemd-tmpfiles."
  getent group render >/dev/null || die "ANE installation requires the render group."
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
  echo "warning: this SoC is not one mlx-omarchy is verified on (M1 family t8103/t6000/t6001/t6002, M2 Max t6021); it is untested here." >&2
fi
command -v python3 >/dev/null || die "python3 is missing."
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

  install_user="${SUDO_USER:-$(id -un)}"
  render_gid="$(getent group render | cut -d: -f3)"
  if [[ " $(id -G "$install_user") " != *" $render_gid "* ]]; then
    sudo usermod -aG render "$install_user"
    echo "note: $install_user was added to render; open a new login session before using ANE."
  fi

  sudo python3 - <<'PY'
import os
import stat

device = os.stat("/dev/accel/accel0", follow_symlinks=False)
if not stat.S_ISCHR(device.st_mode):
    raise SystemExit("/dev/accel/accel0 is not a character device")
expected = (
    ("/run/lock/mlx-omarchy-ane", stat.S_ISDIR, 0o750, False),
    ("/run/lock/mlx-omarchy-ane/device.lock", stat.S_ISREG, 0o660, True),
    ("/run/lock/mlx-omarchy-ane/quarantine", stat.S_ISREG, 0o660, True),
)
for path, type_check, mode, single_link in expected:
    status = os.stat(path, follow_symlinks=False)
    valid = (
        type_check(status.st_mode)
        and status.st_uid == 0
        and status.st_gid == device.st_gid
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
"$VENV/bin/pip" install --quiet "transformers[sentencepiece]==$TRANSFORMERS_VERSION" numpy protobuf pyyaml jinja2 huggingface_hub

# 4b. Vendored mlx-lm serve patches. The GDN fast route is applied by
#     default (gated-delta updates go to mx.fast.gated_delta_update in
#     this wheel); the conv-ring patch stays OFF unless
#     MLX_OMARCHY_CONV_RING=1. Served models pick both up from the venv,
#     so no manual venv patching is needed after install.
say "Applying mlx-lm serve patches (GDN fast route on; conv-ring off unless MLX_OMARCHY_CONV_RING=1)"
curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/scripts/apply-mlx-lm-patches.sh" -o "$PREFIX/apply-mlx-lm-patches.sh"
mkdir -p "$PREFIX/patches"
for p in mlx-lm-gated-delta-fast-route.patch mlx-lm-gated-delta-raw.patch mlx-lm-convring.patch; do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$VERSION/patches/$p" -o "$PREFIX/patches/$p"
done
MLX_OMARCHY_CONV_RING="${MLX_OMARCHY_CONV_RING:-0}" "$PREFIX/apply-mlx-lm-patches.sh" "$VENV"

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
exec "$VENV/bin/python" "$PREFIX/chat.py" "\$@"
EOF
INFO=$("$VENV/bin/python" -I -c 'import os, mlx
print(next(p for root in mlx.__path__
           if os.access(p := os.path.join(root, "bin", "mlx-omarchy-info"), os.X_OK)))')
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$INFO" >"$BIN/mlx-omarchy-info"
chmod +x "$BIN/mlx-omarchy" "$BIN/mlx-omarchy-demo" "$BIN/mlx-omarchy-info"

# 5b. Serve CLI: catalog-driven serving with memory admission and an
#     approve-first download gate. The package ships from the same release
#     tag as the wheel, so an install is internally consistent.
say "Installing serve CLI"
SERVE_PKG="$PREFIX/mlx_omarchy_serve"
mkdir -p "$SERVE_PKG"
for serve_file in __init__.py catalog.py budget.py __main__.py _mlxlm_server.py catalog.json; do
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
  cat >"$dest" <<'OMLX'
#!/usr/bin/env bash
# omarchy:group=mlx
# omarchy:name=serve
# omarchy:summary=Serve a vetted local model on the Apple GPU (memory-checked, approve-first)
# omarchy:args=[target] [--context N] [--server mlx-lm|omlx] [--host H] [--port P] [--yes]
# omarchy:examples=omarchy mlx serve | omarchy mlx serve recommend | omarchy mlx serve plan <model> | omarchy mlx serve catalog list
export PYTHONPATH="$HOME/.local/share/mlx-omarchy${PYTHONPATH:+:$PYTHONPATH}"
exec "$HOME/.local/share/mlx-omarchy/venv/bin/python" -m mlx_omarchy_serve "$@"
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
if command -v omarchy-launch-floating-terminal-with-presentation >/dev/null; then
  cat >"$APPS/mlx-omarchy-demo.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MLX Chat (Apple GPU)
Comment=Chat with a local LLM on the Apple GPU via mlx-omarchy
Exec=omarchy-launch-floating-terminal-with-presentation $BIN/mlx-omarchy-demo
Icon=utilities-terminal
Categories=Development;Utility;
EOF
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
module = (
    "/sys/module/ane"
    if sysroot in ("", "/")
    else os.path.join(sysroot, "sys/module/ane")
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
module_present = os.path.isdir(module)
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
echo "  Run the demo:        mlx-omarchy-demo      (also in the Omarchy app launcher as 'MLX Chat')"
echo "  Use in your scripts: mlx-omarchy your_script.py   (import mlx.core as mx)"
echo "  Serve a model:       mlx-omarchy-serve     (also 'omarchy mlx serve' when registered)"
echo "  Remove everything:   bash install.sh --uninstall"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "  note: $BIN is not on your PATH in this shell; open a new terminal or add it." ;; esac
