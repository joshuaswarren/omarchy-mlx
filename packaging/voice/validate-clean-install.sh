#!/usr/bin/env bash
# VXP-4 — clean-install validator for the offline voice assistant.
#
# Dry-run of the V19 package layout inside a throwaway container (user +
# mount + pid + net namespaces, overlay /usr). Nothing touches the host:
# the artifact is merged over a read-only host /usr, the unit's ExecStart
# is launched against a stub venv python, and the network namespace has no
# route out, proving the install/start path needs no network.
#
#   validate-clean-install.sh [--skeleton-dir DIR | --artifact-usr DIR]
#
#   --skeleton-dir  flat skeleton: omarchy-voice, omarchy-voice-setup,
#                   omarchy-voice.service (default: this script's dir).
#   --artifact-usr  installed-layout usr/ tree from an extracted package.
#
# The omarchy-mlx package's /usr/lib/omarchy-mlx/venv is a separate
# artifact; here it is a stub that records its argv and idles, so the
# unit->wrapper->venv chain is proven without the real runtime.
#
# Output is a fixed-order PASS/FAIL list on stdout; two runs on the same
# artifact must produce identical output. Exit 0 iff every check passed.
# no set -e: a failed check records FAIL and the run continues, so one report lists every failure
set -uo pipefail
PATH="/usr/sbin:/sbin:$PATH"

self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="skeleton"
skeleton_dir="$self_dir"
artifact_usr=""
while (($#)); do
  case "$1" in
    --skeleton-dir) skeleton_dir="${2:?}"; shift 2 ;;
    --artifact-usr) artifact_usr="${2:?}"; mode="artifact"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "error: unknown arg $1" >&2; exit 2 ;;
  esac
done

for tool in unshare systemd-analyze chroot; do
  command -v "$tool" >/dev/null || { echo "error: $tool is required" >&2; exit 1; }
done
unshare -rmpfn true 2>/dev/null || {
  echo "error: this kernel refuses unprivileged user/mount/pid/net namespaces" >&2; exit 1; }

pass_n=0; fail_n=0
pass() { printf 'PASS %s\n' "$1"; pass_n=$((pass_n+1)); }
fail() { printf 'FAIL %s: %s\n' "$1" "$2"; fail_n=$((fail_n+1)); }

# The V19 contract: payload files and their installed paths and modes.
declare -A WANT_MODE=(
  [usr/bin/omarchy-voice]=755
  [usr/bin/omarchy-voice-setup]=755
  [usr/lib/systemd/user/omarchy-voice.service]=644
)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
stage_usr="$work/stage/usr"
mkdir -p "$stage_usr/bin" "$stage_usr/lib/systemd/user"

# --- stage the artifact into installed layout -----------------------------
if [[ $mode == skeleton ]]; then
  for f in omarchy-voice omarchy-voice-setup omarchy-voice.service; do
    [[ -f "$skeleton_dir/$f" ]] || { fail "artifact.skeleton-files" "$skeleton_dir/$f missing"; }
  done
  (( fail_n == 0 )) || { printf 'VALIDATOR FAILURES=%d\n' "$fail_n"; exit 1; }
  install -m 755 "$skeleton_dir/omarchy-voice" "$stage_usr/bin/omarchy-voice"
  install -m 755 "$skeleton_dir/omarchy-voice-setup" "$stage_usr/bin/omarchy-voice-setup"
  install -m 644 "$skeleton_dir/omarchy-voice.service" "$stage_usr/lib/systemd/user/omarchy-voice.service"
  pass "artifact.skeleton-files"
else
  [[ -d $artifact_usr ]] || { echo "error: --artifact-usr $artifact_usr is not a directory" >&2; exit 1; }
  cp -a "$artifact_usr/." "$stage_usr/"
  pass "artifact.installed-tree"
fi

# --- artifact file inventory (fixed sort: no unexpected payloads) ---------
mapfile -t staged < <(cd "$work/stage" && find . -type f | sed 's#^\./##' | LC_ALL=C sort)
unexpected=()
for f in "${staged[@]}"; do
  [[ -n ${WANT_MODE[$f]:-} ]] || unexpected+=("$f")
done
if (( ${#unexpected[@]} )); then
  fail "files.only-expected" "unexpected: ${unexpected[*]}"
else
  pass "files.only-expected"
fi

# --- stub the omarchy-mlx package's venv python ---------------------------
stub_usr="$work/stub/usr"
stub_py="$stub_usr/lib/omarchy-mlx/venv/bin"
mkdir -p "$stub_py"
cat > "$stub_py/python" <<'STUB'
#!/usr/bin/env bash
# Stand-in for the omarchy-mlx package's venv python. Records argv and
# idles so the validator can prove the unit's ExecStart chain launches.
printf '%s\n' "$@" > /home/voice/.voice-stub.argv
sleep 30
STUB
chmod 755 "$stub_py/python"

# --- guest check script, run inside the container -------------------------
guest="$work/guest.sh"
cat > "$guest" <<'GUEST'
set -uo pipefail
pass() { printf 'PASS %s\n' "$1"; }
fail() { printf 'FAIL %s: %s\n' "$1" "$2"; FAILED=$((FAILED+1)); }
FAILED=0

for spec in "usr/bin/omarchy-voice 755" \
            "usr/bin/omarchy-voice-setup 755" \
            "usr/lib/systemd/user/omarchy-voice.service 644"; do
  p="${spec% *}"; want="${spec##* }"
  if [[ -f "/$p" ]]; then
    got="$(stat -c %a "/$p")"
    if [[ $got == "$want" ]]; then pass "files./$p"; else fail "files./$p" "mode $got want $want"; fi
  else
    fail "files./$p" "missing"
  fi
done

if bash -n /usr/bin/omarchy-voice 2>/dev/null && bash -n /usr/bin/omarchy-voice-setup 2>/dev/null; then
  pass "wrappers.bash-syntax"
else
  fail "wrappers.bash-syntax" "a wrapper failed bash -n"
fi

if [[ -x /usr/lib/omarchy-mlx/venv/bin/python ]]; then
  pass "deps.venv-python-stubbed"
else
  fail "deps.venv-python-stubbed" "venv python stub not staged"
fi

unit=/usr/lib/systemd/user/omarchy-voice.service
# Verify the unit on its own, not the whole host graph: a minimal load path
# with stub stand-ins for the boot targets the unit orders against.
unitpath=/run/voice-unit-path
mkdir -p "$unitpath"
cp "$unit" "$unitpath/"
for t in sysinit.target basic.target sound.target default.target \
         local-fs.target sockets.target swap.target slices.target shutdown.target; do
  printf '[Unit]\nDescription=vxp4 stub\n' > "$unitpath/$t"
done
# LogNamespace derives a journal socket unit. That template must live in
# this path, or verify follows the host and two hosts can disagree.
for name in systemd-journald@.socket systemd-journald-varlink@.socket; do
  if [[ -f /usr/lib/systemd/user/$name ]]; then
    cp "/usr/lib/systemd/user/$name" "$unitpath/"
  elif [[ -f /usr/lib/systemd/system/$name ]]; then
    cp "/usr/lib/systemd/system/$name" "$unitpath/"
  else
    printf '%s\n' '[Unit]' 'Description=stub journal socket' '[Socket]' \
      'ListenStream=/run/systemd/journal.%i/io.systemd.journal' > "$unitpath/$name"
  fi
done
if SYSTEMD_UNIT_PATH="$unitpath" timeout 15 systemd-analyze verify "$unitpath/omarchy-voice.service" >/dev/null 2>&1; then
  pass "unit.verify"
else
  fail "unit.verify" "systemd-analyze verify rejected the unit"
fi

exec_line="$(sed -n 's/^ExecStart=//p' "$unit" | head -1)"
exec_bin="${exec_line%% *}"
if [[ $exec_bin == /usr/bin/omarchy-voice && -x $exec_bin ]]; then
  pass "unit.execstart-resolves"
else
  fail "unit.execstart-resolves" "ExecStart '$exec_line' does not resolve to /usr/bin/omarchy-voice"
fi

# "unit starts": run the ExecStart line exactly as systemd would (simple
# type, no shell), prove the stub python recorded the assistant argv and
# the process stayed up, then stop it.
rm -f /home/voice/.voice-stub.argv
# shellcheck disable=SC2086 # intentional split: ExecStart words -> argv
setsid $exec_line >/dev/null 2>&1 &
svc=$!
for _ in 1 2 3 4 5 6; do [[ -s /home/voice/.voice-stub.argv ]] && break; sleep 0.5; done
argv="$(head -3 /home/voice/.voice-stub.argv 2>/dev/null | tr '\n' ' ')"
if [[ -s /home/voice/.voice-stub.argv && $argv == *mlx_omarchy_assistant* && $argv == *--wake-word* ]] \
   && kill -0 "$svc" 2>/dev/null; then
  pass "unit.starts"
else
  fail "unit.starts" "ExecStart did not reach the venv python (argv: ${argv:-none})"
fi
kill "$svc" 2>/dev/null; wait "$svc" 2>/dev/null

# Model prefetch path: the user data home ($MLX_OMARCHY_HOME, default
# ~/.local/share/mlx-omarchy) and the HF weight cache the setup step fills.
for pair in "data-home $HOME/.local/share/mlx-omarchy" \
            "hf-cache $HOME/.cache/huggingface"; do
  name="prefetch.${pair%% *}-writable"; dir="${pair##* }"
  probe="$dir/.vxp4-write-test"
  if mkdir -p "$dir" 2>/dev/null && touch "$probe" 2>/dev/null && rm -f "$probe"; then
    pass "$name"
  else
    fail "$name" "cannot create/write $dir"
  fi
done

# The container was started with an empty network namespace; prove the
# whole sequence above needed no route out.
if timeout 3 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null; then
  fail "container.offline" "the dry-run container can reach the network"
else
  pass "container.offline"
fi

if (( FAILED == 0 )); then printf 'VALIDATOR GREEN\n'; else printf 'VALIDATOR FAILURES=%d\n' "$FAILED"; exit 1; fi
GUEST

# --- assemble the container root ------------------------------------------
rootfs="$work/rootfs"
setup="$work/container-setup.sh"
cat > "$setup" <<SETUP
set -euo pipefail
rootfs="$rootfs"
mkdir -p "\$rootfs"
mount -t tmpfs tmpfs "\$rootfs"
mkdir -p "\$rootfs"/{usr,etc,dev,home/voice,tmp,run}
mount -t tmpfs tmpfs "\$rootfs/tmp"; chmod 1777 "\$rootfs/tmp"
mount -t tmpfs tmpfs "\$rootfs/dev"
for d in null zero full random urandom; do
  mknod "\$rootfs/dev/\$d" c \$(case \$d in null) echo 1 3;; zero) echo 1 5;; full) echo 1 7;; random) echo 1 8;; urandom) echo 1 9;; esac) 2>/dev/null || :
done
mount -t overlay overlay -o "lowerdir=$stub_usr:$stage_usr:/usr" "\$rootfs/usr"
mount --bind /etc "\$rootfs/etc"
mount -o remount,ro,bind "\$rootfs/etc"
for l in bin sbin lib lib64; do
  if [[ -L /\$l ]]; then ln -s "\$(readlink /\$l)" "\$rootfs/\$l"; else mkdir -p "\$rootfs/\$l"; mount --bind "/\$l" "\$rootfs/\$l"; mount -o remount,ro,bind "\$rootfs/\$l"; fi
done
cp "$guest" "\$rootfs/voice-validate-guest.sh"
cd "\$rootfs"   # chroot keeps the cwd; it must exist inside the container
"$(command -v chroot)" "\$rootfs" /usr/bin/env -i HOME=/home/voice PATH=/usr/bin TERM=dumb \
  /usr/bin/bash ${VOICE_VALIDATE_DEBUG:+-x} /voice-validate-guest.sh
SETUP

unshare -rmpfn bash "$setup"
rc=$?
(( rc == 0 )) || fail_n=$((fail_n+1))
exit "$fail_n"
