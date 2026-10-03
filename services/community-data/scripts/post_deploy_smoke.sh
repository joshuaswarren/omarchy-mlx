#!/usr/bin/env bash
# Post-deploy smoke for the community-data worker (2026-10-03 incident
# follow-up): deploy migrations BEFORE code, deploy, then run this.
# Any failure auto-rolls the worker back to the previous deployment
# and exits nonzero, so a bad deploy cannot sit on the promotion job's
# read path.
#
# Usage: scripts/post_deploy_smoke.sh [base-url]
# Env: CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID must be set (or
# sourced from ~/.config/cloudflare/thewarrens-co.env) for rollback.
set -euo pipefail

BASE="${1:-https://mlx-omarchy-community-data.joshua-s-warren.workers.dev}"
UA="omarchy-mlx-community-data-smoke/1.0"

fail() {
  echo "[smoke] FAIL: $*" >&2
  echo "[smoke] rolling back worker..." >&2
  npx wrangler rollback --yes >&2 || echo "[smoke] ROLLBACK FAILED — manual action required" >&2
  exit 1
}

get_status() {
  curl -sS -o /dev/null -w '%{http_code}' -A "$UA" --max-time 30 "$1"
}

echo "[smoke] 1/3 GET /v1/schema"
code="$(get_status "$BASE/v1/schema")"
[ "$code" = "200" ] || fail "/v1/schema -> $code"

echo "[smoke] 2/3 GET /v1/results parses as JSON"
body_file="$(mktemp)"
curl -sS -A "$UA" --max-time 60 -o "$body_file" "$BASE/v1/results"
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$body_file" \
  || fail "/v1/results body is not valid JSON"
code="$(get_status "$BASE/v1/results")"
[ "$code" = "200" ] || fail "/v1/results -> $code"

echo "[smoke] 3/3 three random GET /v1/results/<sha>"
shas="$(python3 -c 'import json,random,sys
d = json.load(open(sys.argv[1]))
rows = d.get("results") or []
shas = [r["content_sha256"] for r in rows if r.get("content_sha256")]
print("\n".join(random.sample(shas, min(3, len(shas)))))' "$body_file")" \
  || fail "could not extract shas from /v1/results"
rm -f "$body_file"
[ -n "$shas" ] || fail "no rows in /v1/results to probe"
for sha in $shas; do
  code="$(get_status "$BASE/v1/results/$sha")"
  echo "[smoke]   $sha -> $code"
  [ "$code" = "200" ] || fail "/v1/results/$sha -> $code"
done

echo "[smoke] OK: schema, index, and 3 result reads all 200"
