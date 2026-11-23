#!/bin/sh
# Independent tests for privacy_check.py + the REAL pre-push hook.
# All sensitive strings are SYNTHETIC and built via concatenation at
# runtime, so this file contains no literals the generic guard would
# flag. mktemp + trap cleanup.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CHECKER="$HERE/privacy_check.py"
HOOK="$HERE/hooks/pre-push"
ID="816217+joshuaswarren@users.noreply.github.com"
FAILS=0
T=$(mktemp -d) || exit 2
trap 'rm -rf "$T"' EXIT
SITE="$T/site-patterns"
# synthetic tokens via concatenation:
H1="synth"$(printf '')"host1"; SEC="synth"$(printf '')"secret"
UPATH="/"$(printf '')"home/synth"$(printf '')"user/"
printf '%s|%s|%s\n' "$H1" "$SEC" "$UPATH" > "$SITE"
NL=$(printf '\n')

newrepo(){
  R="$1"; mkdir -p "$R"
  git init -q -b main "$R"
  git -C "$R" config user.email "$ID"; git -C "$R" config user.name "t"
  printf '.private-patterns\n' > "$R/.gitignore"
  echo base > "$R/base.txt"; git -C "$R" add .; git -C "$R" commit -qm base; git -C "$R" tag vbase
  printf '%s|%s|%s\n' "$H1" "$SEC" "$UPATH" > "$R/.private-patterns"
}
g(){ PRIVACY_REPO="$R" PRIVACY_PATTERNS_NAME=".private-patterns" python3 "$CHECKER" "$@"; }

# Real-hook clones: the hook also runs the compute-kernel wiring gate and
# the kernel-inventory doc gate before privacy_check. Both resolve outside
# a bare test clone ($HERE/../check_compute_kernel_wiring.py is missing,
# and $HERE/../../tools/ escapes the clone to $T). Stub both so only the
# privacy path is under test.
stubgates(){
  printf 'pass\n' > "$1/hooks/check_compute_kernel_wiring.py"
  mkdir -p "$T/tools/kernel_recheck"
  printf 'pass\n' > "$T/tools/kernel_recheck/check_doc_table.py"
}

# 1 clean actual commit
R="$T/c1"; newrepo "$R"; echo harmless > "$R/x.txt"; git -C "$R" add .; git -C "$R" commit -qm clean
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL1 clean rc=$rc"; FAILS=$((FAILS+1)); }

# 2 add-then-remove secret on merged side branch
git -C "$R" checkout -q -b feat
printf 'key=%s\n' "$SEC" > "$R/cfg.txt"; git -C "$R" add .; git -C "$R" commit -qm add
git -C "$R" rm -qf cfg.txt; git -C "$R" commit -qm remove
git -C "$R" checkout -q main; git -C "$R" merge -q --no-ff -m merge feat
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL2 add-remove-merged rc=$rc"; FAILS=$((FAILS+1)); }

# 3 root commit secret (single-rev full history)
R="$T/c3"; mkdir -p "$R"; git init -q -b main "$R"
git -C "$R" config user.email "$ID"; git -C "$R" config user.name "t"
printf 'host=%s\n' "$H1" > "$R/s.txt"; git -C "$R" add .; git -C "$R" commit -qm root
printf '%s|%s|%s\n' "$H1" "$SEC" "$UPATH" > "$R/.private-patterns"
g main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL3 root rc=$rc"; FAILS=$((FAILS+1)); }

# 4 private filename (synthetic token)
R="$T/c4"; newrepo "$R"; printf x > "$R/${H1}-probe.sh"; git -C "$R" add .; git -C "$R" commit -qm probe
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL4 filename rc=$rc"; FAILS=$((FAILS+1)); }

# 5 message token
R="$T/c5"; newrepo "$R"; printf y > "$R/d.md"; git -C "$R" add .; git -C "$R" commit -qm "measured on $H1"
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL5 message rc=$rc"; FAILS=$((FAILS+1)); }

# 6 bad range fail-closed
R="$T/c5"; g nosuch..alsonot >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL6 badrange rc=$rc"; FAILS=$((FAILS+1)); }

# 7 content line beginning with ++ and containing secret detected
R="$T/c7"; newrepo "$R"; printf '++%s\n' "$SEC" > "$R/plus.txt"; git -C "$R" add .; git -C "$R" commit -qm plus
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL7 plus-secret rc=$rc"; FAILS=$((FAILS+1)); }

# 8 binary file with secret bytes (octal printf, NUL asserted)
R="$T/c8"; newrepo "$R"
printf 'BIN\000%s\001\377' "$SEC" > "$R/blob.bin"
od -An -c "$R/blob.bin" | grep -q '\\0' || { echo "FAIL8-setup no NUL"; FAILS=$((FAILS+1)); }
git -C "$R" add .; git -C "$R" commit -qm bin
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL8 binary rc=$rc"; FAILS=$((FAILS+1)); }

# 9 rename to token filename with space
R="$T/c9"; newrepo "$R"; printf x > "$R/keep.txt"; git -C "$R" add .; git -C "$R" commit -qm base2
git -C "$R" mv keep.txt "$H1 probe file"; git -C "$R" add .; git -C "$R" commit -qm rename-space
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL9 rename-space rc=$rc"; FAILS=$((FAILS+1)); }

# 10 malformed stdin -> fail-closed
R="$T/m1"; newrepo "$R"
printf 'refs/heads/main %s\n' "$(git -C "$R" rev-parse HEAD)" | PRIVACY_REPO="$R" python3 "$CHECKER" --hook >/dev/null 2>&1
rc=$?; [ $rc -eq 1 ] || { echo "FAIL10 malformed rc=$rc"; FAILS=$((FAILS+1)); }

# 11 content line '++ secret' (plus space) detected
R="$T/c11"; newrepo "$R"; printf '++ %s\n' "$SEC" > "$R/plus2.txt"; git -C "$R" add .; git -C "$R" commit -qm plus2
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL11 plus-space rc=$rc"; FAILS=$((FAILS+1)); }

# 12 binary secret duplicate-named file (binary + renamed binary)
R="$T/c12"; newrepo "$R"
printf 'BIN\000%s\002\376' "$SEC" > "$R/blob2.bin"; git -C "$R" add .; git -C "$R" commit -qm bin2
git -C "$R" mv blob2.bin "$UPATH"$(printf '')"renamed.bin" 2>/dev/null || mv "$R/blob2.bin" "$R/renamed.bin"
git -C "$R" add .; git -C "$R" commit -qm rename-bin 2>/dev/null
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL12 rename-binary rc=$rc"; FAILS=$((FAILS+1)); }

# 13 REAL HOOK new-ref: secret commit blocked (full history default)
BR="$T/remote.git"; mkdir -p "$BR"; git init -q --bare "$BR"
C="$T/clone"; git clone -q "$BR" "$C" 2>/dev/null; git -C "$C" checkout -q -b main
mkdir -p "$C/hooks"; cp "$CHECKER" "$C/hooks/privacy_check.py"; cp "$HOOK" "$C/hooks/pre-push"
chmod +x "$C/hooks/pre-push" "$C/hooks/privacy_check.py"; stubgates "$C"
printf '%s|%s|%s\n' "$H1" "$SEC" "$UPATH" > "$C/.private-patterns"
git -C "$C" config user.email "$ID"; git -C "$C" config user.name "t"; git -C "$C" config core.hooksPath hooks
printf 'token=%s\n' "$SEC" > "$C/leak.txt"; git -C "$C" add .; git -C "$C" commit -qm leak
git -C "$C" push origin HEAD:main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL13 hook-block rc=$rc"; FAILS=$((FAILS+1)); }

# 14 REAL HOOK existing ref: clean passes, then secret blocked
BR2="$T/remote2.git"; mkdir -p "$BR2"; git init -q --bare "$BR2"
C2="$T/clone2"; git clone -q "$BR2" "$C2" 2>/dev/null; git -C "$C2" checkout -q -b main
mkdir -p "$C2/hooks"; cp "$CHECKER" "$C2/hooks/privacy_check.py"; cp "$HOOK" "$C2/hooks/pre-push"
chmod +x "$C2/hooks/pre-push" "$C2/hooks/privacy_check.py"; stubgates "$C2"
git -C "$C2" config user.email "$ID"; git -C "$C2" config user.name "t"; git -C "$C2" config core.hooksPath hooks
printf '.private-patterns\n' > "$C2/.gitignore"
echo ok > "$C2/f.txt"; git -C "$C2" add .; git -C "$C2" commit -qm ok
printf '%s|%s|%s\n' "$H1" "$SEC" "$UPATH" > "$C2/.private-patterns"
git -C "$C2" push origin HEAD:main >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL14 initial rc=$rc"; FAILS=$((FAILS+1)); }
printf 'token=%s\n' "$SEC" > "$C2/leak.txt"; git -C "$C2" add .; git -C "$C2" commit -qm leak
git -C "$C2" push origin HEAD:main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL14 existing-ref rc=$rc"; FAILS=$((FAILS+1)); }

# 15 newline filename with token detected (-z path preservation)
R="$T/c15"; newrepo "$R"
F="$R/token$NL$H1.txt"; printf 'x\n' > "$F"; git -C "$R" add .; git -C "$R" commit -qm nlname
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL15 newline-filename rc=$rc"; FAILS=$((FAILS+1)); }

# 16 no self-trigger: guard commits its own sources cleanly
R="$T/self"; newrepo "$R"; mkdir -p "$R/scripts/hooks"
cp "$CHECKER" "$R/scripts/privacy_check.py"; cp "$HOOK" "$R/scripts/hooks/pre-push"
cp "$0" "$R/scripts/privacy_check-tests.sh"
git -C "$R" add .; git -C "$R" commit -qm guard
g vbase..main >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL16 self-trigger rc=$rc"; FAILS=$((FAILS+1)); }

# 17 .privacy-allow exact-blob pin: pinned fixture passes a new-ref
# (full-history) scan; a new blob at the pinned path blocks again
R="$T/c17"; newrepo "$R"
printf 'path=%s\n' "$UPATH" > "$R/fixture.txt"; git -C "$R" add .; git -C "$R" commit -qm fixture
g main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL17 unpinned rc=$rc"; FAILS=$((FAILS+1)); }
printf '%s fixture.txt\n' "$(git -C "$R" rev-parse HEAD:fixture.txt)" >> "$R/.privacy-allow"
g main >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL17 pinned rc=$rc"; FAILS=$((FAILS+1)); }
printf 'path=%s\n' "$UPATH" >> "$R/fixture.txt"; git -C "$R" add .; git -C "$R" commit -qm fixture2
g main >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL17 edited rc=$rc"; FAILS=$((FAILS+1)); }

# 18 REAL HOOK existing ref rebased over a published flagged blob pushes
# clean; a NEW commit adding a flagged pattern is still refused.
BR3="$T/remote3.git"; mkdir -p "$BR3"; git init -q --bare "$BR3"
C3="$T/clone3"; git clone -q "$BR3" "$C3" 2>/dev/null; git -C "$C3" checkout -q -b main
mkdir -p "$C3/hooks"; cp "$CHECKER" "$C3/hooks/privacy_check.py"; cp "$HOOK" "$C3/hooks/pre-push"
chmod +x "$C3/hooks/pre-push" "$C3/hooks/privacy_check.py"; stubgates "$C3"
git -C "$C3" config user.email "$ID"; git -C "$C3" config user.name "t"; git -C "$C3" config core.hooksPath hooks
printf '.private-patterns\n' > "$C3/.gitignore"
echo ok > "$C3/f.txt"; git -C "$C3" add .; git -C "$C3" commit -qm ok
printf '%s|%s|%s\n' "$H1" "$SEC" "$UPATH" > "$C3/.private-patterns"
git -C "$C3" push -q origin main >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL18 main-push rc=$rc"; FAILS=$((FAILS+1)); }
git -C "$C3" checkout -q -b feat
echo work > "$C3/w.txt"; git -C "$C3" add .; git -C "$C3" commit -qm featwork
git -C "$C3" push -q origin feat >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL18 feat-push rc=$rc"; FAILS=$((FAILS+1)); }
# publisher clone WITHOUT the hook lands a flagged blob on main
CP3="$T/pub3"; git clone -q "$BR3" "$CP3" 2>/dev/null
git -C "$CP3" config user.email "$ID"; git -C "$CP3" config user.name "t"
git -C "$CP3" checkout -q main
printf 'dir=%s\n' "$UPATH" > "$CP3/pub.sh"; git -C "$CP3" add .; git -C "$CP3" commit -qm published
git -C "$CP3" push -q origin main >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL18 publish rc=$rc"; FAILS=$((FAILS+1)); }
# guarded clone fetches, rebases feat past the flagged main commit,
# force-pushes: published history is not re-blocked
git -C "$C3" fetch -q origin
git -C "$C3" rebase -q origin/main feat
git -C "$C3" push -q --force-with-lease origin feat >/dev/null 2>&1; rc=$?; [ $rc -eq 0 ] || { echo "FAIL18 rebase-push rc=$rc"; FAILS=$((FAILS+1)); }
# a NEW commit introducing a flagged pattern is still refused
printf 'dir=%s\n' "$UPATH" > "$C3/new.sh"; git -C "$C3" add .; git -C "$C3" commit -qm leak2
git -C "$C3" push -q origin feat >/dev/null 2>&1; rc=$?; [ $rc -eq 1 ] || { echo "FAIL18 new-flag rc=$rc"; FAILS=$((FAILS+1)); }

rm -rf "$T" "$SITE"
if [ $FAILS -eq 0 ]; then echo ALL_GUARD_V5_TESTS_PASS; else echo "FAILURES=$FAILS"; exit 1; fi
