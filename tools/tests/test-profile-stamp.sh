#!/usr/bin/env bash
# d-0102 p1: the diag profiling harness must refuse to run when the venv's
# mlx_lm carries no apply-mlx-lm-patches stamp (SC-11/SC-12/GAP-E profiled an
# unpatched mlx-lm and reported 800 launches/token where prod runs 205).
# no set -e: the suite asserts intentional failing exit codes (rc 1/2/3) from
# the tool; -e would abort on the first deliberate failure
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
src=$(cd "$here/../.." && pwd)
fail=0
ok() { echo "PASS $1"; }
bad() { echo "FAIL $1"; fail=1; }

t=$(mktemp -d /tmp/profile-stamp-XXXXXX)
# a stub venv: bin/python execs the real python3 with PYTHONPATH pointed at a
# stub site-packages that provides the mlx_lm surface profile_generate uses
mkdir -p "$t/venv/bin" "$t/site/mlx_lm"
cat > "$t/venv/bin/python" <<'PY'
#!/usr/bin/env bash
exec python3 -c 'import sys; sys.path.insert(0, os.environ["STUB_SITE"]); import os' 2>/dev/null
PY
# a real wrapper: exec python3 with the stub site-packages first
cat > "$t/venv/bin/python" <<PY
#!/usr/bin/env bash
exec env PYTHONPATH="$t/site\${PYTHONPATH:+:\$PYTHONPATH}" python3 -u "\$@"
PY
chmod +x "$t/venv/bin/python"
printf 'def load(path, **k):\n    raise RuntimeError("stub")\n' > "$t/site/mlx_lm/__init__.py"
mkdir -p "$t/site/mlx_lm/utils" "$t/site/mlx_lm/generate" "$t/site/mlx_lm/sample_utils"
printf 'def load(*a, **k):\n    raise RuntimeError("stub")\n' > "$t/site/mlx_lm/utils/__init__.py"
printf 'def stream_generate(*a, **k):\n    raise RuntimeError("stub")\n' > "$t/site/mlx_lm/generate/__init__.py"
printf 'def make_sampler(*a, **k):\n    return None\n' > "$t/site/mlx_lm/sample_utils/__init__.py"

run_gen() { # run_gen <extra python args...>: profile_generate under the stub venv
  "$t/venv/bin/python" "$src/scripts/profile_generate.py" \
    --model /nonexistent/model --prompt hi --max-tokens 4 "$@"
}

# 1. without the stamp the harness refuses before touching mlx_lm
out=$(run_gen 2>&1); rc=$?
if [ "$rc" = 3 ] && grep -q "omarchy-patches-stamp" <<<"$out"; then
  ok "no stamp: harness refuses with exit 3 and names the stamp"
else
  bad "no stamp: rc=$rc out=$(head -c 200 <<<"$out")"
fi

# 2. with the stamp the harness proceeds past the check (the stub mlx_lm
#    raises its own load error afterwards, which is the stub's doing)
printf 'series=patches\nwritten=2026-11-23T12:00:00+00:00\n' > "$t/site/mlx_lm/.omarchy-patches-stamp"
out=$(run_gen 2>&1); rc=$?
if [ "$rc" != 3 ] && ! grep -q "omarchy-patches-stamp" <<<"$out"; then
  ok "with stamp: the stamp refusal is gone"
else
  bad "with stamp: rc=$rc out=$(head -c 200 <<<"$out")"
fi

# 3. the stamp writer lands the file next to mlx_lm
rm -f "$t/site/mlx_lm/.omarchy-patches-stamp"
python3 "$src/scripts/mlx_patch_stamp.py" write "$t/site/mlx_lm/__init__.py" patches >/dev/null 2>&1 &&
  grep -q "series=patches" "$t/site/mlx_lm/.omarchy-patches-stamp" &&
  ok "the stamp writer lands the file" || bad "the stamp writer failed"

# 4. trace_dispatch carries the same refusal inside the code it runs in the
#    venv python (the chat phase imports mlx_lm there)
grep -q "require_stamp" "$src/scripts/trace_dispatch.py" &&
  ok "trace_dispatch's venv code carries the stamp refusal" ||
  bad "trace_dispatch's venv code has no stamp refusal"

if [ $fail = 0 ]; then echo "ALL PASS"; fi
exit $fail
