#!/bin/bash
# GDN coopmat A/B window on m1max-host (jw16) — 2026-09-22, GdnCoopmat lane.
# Owner-approved bounded window. Discipline: stop llm-inference BEFORE
# flock on /tmp/m1-gpu.lock; release the lock (exit) before restart.
set -u
W=/var/tmp/gdncoop
LOG=$W/window.log
mkdir -p "$W"
exec >>"$LOG" 2>&1
WINDOW_DONE=0
trap 'if [ "$WINDOW_DONE" = 0 ]; then echo "=== gdncoop window end ABNORMAL $(date -Is) ==="; fi' EXIT
echo "=== gdncoop window start $(date -Is) ==="
uptime

sudo systemctl stop llm-inference
echo "llm-inference: $(systemctl is-active llm-inference)"
exec 9>/tmp/m1-gpu.lock
flock -w 600 9 || { echo "LOCK-TIMEOUT" >&2; sudo systemctl start llm-inference; exit 9; }

SNAP=$(ls -d ~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/* | head -1)
echo "model snapshot: $SNAP"
PROMPTS=$W/qwen38-2b-prompts.jsonl
[ -f "$PROMPTS" ] || cp /var/tmp/ppa/qwen38-2b-prompts.jsonl "$PROMPTS" 2>/dev/null || true
[ -f "$PROMPTS" ] || { echo "NO-PROMPTS"; }
python3 - "$PROMPTS" <<'EOF'
import json, sys
text = " ".join(json.loads(l)["text"] for l in open(sys.argv[1]) if l.strip())
open("/var/tmp/gdncoop/prompt.txt", "w").write(text)
EOF

# ---------- 1. wheels ----------
cd /var/tmp/compose-wt
echo "compose-wt at: $(git log --oneline -1) status:$(git status --porcelain | wc -l)"
git fetch /var/tmp/gdncoop/gdn-coopmat-full.bundle \
    "refs/heads/agent/gdn-coopmat:refs/heads/agent/gdn-coopmat" || true
git worktree add /var/tmp/gdncoop-wt agent/gdn-coopmat 2>/dev/null || true
cd /var/tmp/gdncoop-wt
git log --oneline -1
mkdir -p "$W/wheels"
DIAG_WHL=$(ls "$W/wheels"/*diag*.whl 2>/dev/null | head -1)
if [ -z "$DIAG_WHL" ]; then
  scripts/build-wheel.sh --diagnostics >"$W/build-diag.log" 2>&1 \
    && echo DIAG-BUILD-OK || { echo DIAG-BUILD-FAIL; tail -20 "$W/build-diag.log"; }
  cp dist/*.whl "$W/wheels"/ 2>/dev/null
  DIAG_WHL=$(ls "$W/wheels"/*diag*.whl 2>/dev/null | head -1)
else
  echo DIAG-BUILD-SKIPPED "$DIAG_WHL"
fi
REL_WHL=$(ls "$W/wheels"/mlx_omarchy-*.whl 2>/dev/null | grep -v diag | head -1)
if [ -z "$REL_WHL" ]; then
  scripts/build-wheel.sh >"$W/build-rel.log" 2>&1 \
    && echo REL-BUILD-OK || { echo REL-BUILD-FAIL; tail -20 "$W/build-rel.log"; }
  cp dist/*.whl "$W/wheels"/ 2>/dev/null
  REL_WHL=$(ls "$W/wheels"/mlx_omarchy-*.whl 2>/dev/null | grep -v diag | head -1)
else
  echo REL-BUILD-SKIPPED "$REL_WHL"
fi
ls -la "$W/wheels"/ 2>/dev/null
sha256sum "$W/wheels"/*.whl 2>/dev/null | tee "$W/wheel-shas.txt"
echo "DIAG_WHL=$DIAG_WHL"; echo "REL_WHL=$REL_WHL"
[ -n "$DIAG_WHL" ] && [ -n "$REL_WHL" ] || { echo WHEELS-MISSING; sudo systemctl start llm-inference; exit 1; }

# Test venvs: clones of the patched production venv (mlx-lm 0.31.3 with the
# gated-delta fast-route patch already applied), candidate wheel forced in.
rm -rf "$W/venv-diag" "$W/venv-rel"
cp -a /var/tmp/v072-venv-fused "$W/venv-diag"
cp -a /var/tmp/v072-venv-fused "$W/venv-rel"
"$W/venv-diag/bin/python" -m pip install -q --ignore-installed --no-deps "$DIAG_WHL" \
  || { echo DIAG-VENV-INSTALL-FAIL; sudo systemctl start llm-inference; exit 1; }
"$W/venv-rel/bin/python" -m pip install -q --ignore-installed --no-deps "$REL_WHL" \
  || { echo REL-VENV-INSTALL-FAIL; sudo systemctl start llm-inference; exit 1; }
"$W/venv-diag/bin/python" -c "import mlx.core as mx; v = mx.__version__; print('diag wheel:', v); assert 'diag' in v, 'NOT the diag wheel'"
"$W/venv-rel/bin/python" -c "import mlx.core as mx; print('rel wheel:', mx.__version__)"

# ---------- 2. shaderdb stats ----------
export AGX_MESA_DEBUG=shaderdb,internal MESA_SHADER_CACHE_DISABLE=true
"$W/venv-diag/bin/python" "$W/gdn_probe.py" 512 >"$W/db-coop.log" 2>&1
MLX_OMARCHY_NO_COOPMAT_GDN=1 "$W/venv-diag/bin/python" "$W/gdn_probe.py" 512 >"$W/db-scan.log" 2>&1
unset AGX_MESA_DEBUG MESA_SHADER_CACHE_DISABLE
echo "--- shaderdb coopmat kernel ---"; grep -iE "CS shader|scratch|spill" "$W/db-coop.log" | tail -5
echo "--- shaderdb scan kernel ---";    grep -iE "CS shader|scratch|spill" "$W/db-scan.log" | tail -5

# ---------- 3. offline numeric check ----------
"$W/venv-diag/bin/python" "$W/gdn-coopmat-check.py" "$W/venv-diag/bin/python" \
    "$W/numeric-check.json" >"$W/numeric-check.out" 2>&1 \
  && echo NUMERIC-SCRIPT-OK || { echo NUMERIC-SCRIPT-FAIL; tail -10 "$W/numeric-check.out"; }
grep -E "DETERMINISM|quanta|exact_frac|max_abs" "$W/numeric-check.out" | head -20

# ---------- 4. per-kernel profile (diag wheel) ----------
export MLX_OMARCHY_GPU_PROFILE_LABEL=gdncoop
export MLX_OMARCHY_GPU_PROFILE=$W/prof-coop.ndjson
"$W/venv-diag/bin/python" /var/tmp/ppa/profile_prefill.py "$W/prompt.txt" >"$W/pref-coop.log" 2>&1
export MLX_OMARCHY_NO_COOPMAT_GDN=1
export MLX_OMARCHY_GPU_PROFILE=$W/prof-scan.ndjson
"$W/venv-diag/bin/python" /var/tmp/ppa/profile_prefill.py "$W/prompt.txt" >"$W/pref-scan.log" 2>&1
unset MLX_OMARCHY_NO_COOPMAT_GDN MLX_OMARCHY_GPU_PROFILE MLX_OMARCHY_GPU_PROFILE_LABEL
echo "--- prefill wall (diag wheel) ---"
grep -h "PREFILL" "$W/pref-coop.log" "$W/pref-scan.log"
python3 - "$W/prof-coop.ndjson" "$W/prof-scan.ndjson" <<'EOF'
import json, sys, collections
for tag, path in (("coop", sys.argv[1]), ("scan", sys.argv[2])):
    agg = collections.defaultdict(lambda: [0, 0.0])
    try:
        for line in open(path):
            e = json.loads(line)
            k = e.get("kernel") or e.get("name") or "?"
            agg[k][0] += 1
            agg[k][1] += float(e.get("ms", e.get("duration_ms", 0)))
    except FileNotFoundError:
        print(tag, "no profile")
        continue
    print(f"--- {tag} kernel totals (ms) ---")
    for k, (n, ms) in sorted(agg.items(), key=lambda x: -x[1][1]):
        if ms > 1.0:
            print(f"{k:44s} n={n:5d} total={ms:9.2f} mean={ms/max(n,1):8.4f}")
EOF

# ---------- 5. contract arms (release wheel, same wheel both paths) ----------
BENCH=$W/qwen38-mlx-bench.py
"$W/venv-rel/bin/python" "$BENCH" --model "$SNAP" --prompts "$PROMPTS" \
    --limit 10 --new-tokens 32 --prefill-tokens 512 --warmup 3 --passes 10 \
    --label gdncoop-coop --out "$W/contract-coop.json" >"$W/contract-coop.log" 2>&1 \
  && echo CONTRACT-COOP-OK || echo CONTRACT-COOP-FAIL
MLX_OMARCHY_NO_COOPMAT_GDN=1 "$W/venv-rel/bin/python" "$BENCH" --model "$SNAP" \
    --prompts "$PROMPTS" --limit 10 --new-tokens 32 --prefill-tokens 512 \
    --warmup 3 --passes 10 --label gdncoop-scan --out "$W/contract-scan.json" \
    >"$W/contract-scan.log" 2>&1 && echo CONTRACT-SCAN-OK || echo CONTRACT-SCAN-FAIL
python3 - "$W/contract-scan.json" "$W/contract-coop.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
c = json.load(open(sys.argv[2]))
for tag, d in (("scan ", s), ("coop ", c)):
    print(f"{tag} ttft={d['ttft_tok_rate']['median']:7.2f} "
          f"decode={d['decode_tok_rate']['median']:6.2f} "
          f"prefill512={d['pure_prefill']['pure_prefill_tok_rate']:7.2f} "
          f"digest={d['ordered_records_sha256'][:16]}")
EOF

# ---------- 6. teacher-forced logits ----------
GL=$W/logits_gl.py
export MLX_OMARCHY_NO_COOPMAT_GDN=1
"$W/venv-rel/bin/python" "$GL" "$PROMPTS" "$W/logits-scan.json" 32 >"$W/logits-scan.log" 2>&1 \
  || { echo LOGITS-SCAN-FAIL; tail -5 "$W/logits-scan.log"; }
unset MLX_OMARCHY_NO_COOPMAT_GDN
"$W/venv-rel/bin/python" "$GL" "$PROMPTS" "$W/logits-coop.json" 32 >"$W/logits-coop.log" 2>&1 \
  || { echo LOGITS-COOP-FAIL; tail -5 "$W/logits-coop.log"; }
"$W/venv-rel/bin/python" "$W/logits_cmp.py" "$W/logits-scan.json" "$W/logits-coop.json" | tee "$W/logits-verdict.txt"

# ---------- 7. gate + install ----------
python3 - "$W" <<'EOF'
import json, sys
W = sys.argv[1]
s = json.load(open(f"{W}/contract-scan.json"))
c = json.load(open(f"{W}/contract-coop.json"))
pf = c["pure_prefill"]["pure_prefill_tok_rate"]
pf0 = s["pure_prefill"]["pure_prefill_tok_rate"]
dc = c["decode_tok_rate"]["median"]; dc0 = s["decode_tok_rate"]["median"]
tt = c["ttft_tok_rate"]["median"]; tt0 = s["ttft_tok_rate"]["median"]
num = json.load(open(f"{W}/numeric-check.json"))
det = all(num[t]["determinism"] for t in ("g_f32", "g_bf16"))
logits = open(f"{W}/logits-verdict.txt").read()
flips = int([l for l in logits.splitlines() if l.startswith("steps=")][0].split("flips=")[1].split()[0])
maxd = float([l for l in logits.splitlines() if l.startswith("steps=")][0].split("max|d_top1|=")[1].split()[0])
gate = {
    "prefill_ge_324_or_scan_plus_10pct": pf >= max(324.0, pf0 * 1.10),
    "prefill_scan_tok_s": pf0, "prefill_coop_tok_s": pf,
    "decode_no_regress": dc >= dc0 * 0.98, "decode_scan": dc0, "decode_coop": dc,
    "ttft_no_regress": tt <= tt0 * 1.05, "ttft_scan": tt0, "ttft_coop": tt,
    "determinism": det,
    "logits_0_flips": flips == 0, "logits_maxd_le_0.5": maxd <= 0.5,
    "flips": flips, "maxd": maxd,
}
win = all(v for k, v in gate.items() if isinstance(v, bool))
gate["ALL-GATES"] = win
print(json.dumps(gate, indent=1))
open(f"{W}/gate.json", "w").write(json.dumps(gate, indent=1))
EOF

if python3 -c "import json,sys; sys.exit(0 if json.load(open('$W/gate.json'))['ALL-GATES'] else 1)"; then
  echo "=== INSTALL: gates pass; installing release wheel into production venv ==="
  echo "pre-install state:" | tee "$W/install.log"
  /var/tmp/v072-venv-fused/bin/pip freeze 2>/dev/null | grep -i "mlx" | tee -a "$W/install.log"
  rm -rf /var/tmp/rollback-gdncoop && mkdir -p /var/tmp/rollback-gdncoop
  cp -a /var/tmp/v072-venv-fused /var/tmp/rollback-gdncoop/venv-pre-gdncoop \
      && echo "rollback venv snapshot: /var/tmp/rollback-gdncoop/venv-pre-gdncoop" \
      || echo "WARN: rollback venv snapshot failed"
  /var/tmp/v072-venv-fused/bin/pip install -q --force-reinstall --no-deps "$REL_WHL" \
      >>"$W/install.log" 2>&1 && echo INSTALL-OK || { echo INSTALL-FAIL; tail -10 "$W/install.log"; }
  /var/tmp/v072-venv-fused/bin/python -c "import mlx.core as mx; print('installed:', mx.__version__)"
  sha256sum "$REL_WHL" | tee -a "$W/install.log"
else
  echo "=== NO-INSTALL: gates failed; production venv untouched ==="
fi

# ---------- 8. restore ----------
sudo systemctl start llm-inference
for i in $(seq 1 24); do
  sleep 5
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8002/health)
  [ "$code" = "200" ] && { echo "health 200 at poll $i"; break; }
done
KEY=$(sudo cat /etc/llm-inference/api-key)
curl -s --max-time 60 http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}' \
  > "$W/completion.json"
grep -oE '"(content|finish_reason)"[^,}]*' "$W/completion.json" | head -4
WINDOW_DONE=1
echo "=== gdncoop window end $(date -Is) ==="
