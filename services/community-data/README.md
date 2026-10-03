# mlx-omarchy community data

A self-contained Cloudflare Worker that accepts contributor hardware and
benchmark submissions from the `scripts/collect_*` collectors and serves
them as a public, agent-readable dataset. Zero dollars: Workers Free and
D1 Free only. No R2 (not enabled on the account), no KV, no paid
add-ons. Large archives are stored as chunks across multiple D1 rows.

## Layout

```
migrations/0001_init.sql   versioned D1 schema
schema/payload-v1.schema.json  canonical strict JSON Schema for summaries
src/                       worker (no framework: fetch + scheduled)
public/index.html          static human-readable index (no build step)
test/unit/                 pure unit tests (bun test)
test/smoke.ts              local end-to-end smoke over wrangler dev
```

## Wire contract

All requests and responses are JSON unless stated otherwise. Errors are
always `{"error": <machine code>, "detail": ...}`.

| Route | Behavior |
|---|---|
| `POST /v1/submit` | Initiate. Body: `{schema_version, kind, content_sha256, payload, archive, pow}`. Verifies proof of work, validates the summary against the pinned schema, scans it for PII, enforces caps. On PII hits the worker STRIPS matched values from the payload and stores the cleaned copy (it does not refuse; the 422 `pii_detected` path is removed). Returns `{status: "awaiting_chunks"\|"stored"\|"duplicate", content_sha256, missing_chunks, receipt_url, pii_redacted?: {kind: n}}` — `pii_redacted` rides the response only when a strip happened. |
| `POST /v1/submit/<sha>/chunk/<idx>` | One raw octet-stream chunk. Idempotent; verifies the received bytes against the hash declared at initiate. Returns `{status, idx, missing_chunks}`. |
| `POST /v1/submit/<sha>/complete` | Publishes when every chunk is present and every stored hash matches the declared hash. `409 {error:"incomplete", missing_chunks}` otherwise. |
| `GET /v1/submit/<sha>` | Dedup probe: `200 {status:"duplicate", receipt_url}` or `404`. |
| `GET /v1/results` | Cached index of published summaries. Each entry carries an added `content_sha256` field so consumers can address records. |
| `GET /v1/results/<sha>` | One full published record (live, not cached). |
| `GET /v1/results/<sha>/archive` | The reassembled archive bytes, streamed in chunk order. Immutable, content-addressed. |
| `GET /v1/dataset/latest.jsonl` | One JSON object per line, for bulk agent consumption. Cached. |
| `GET /v1/schema` | Schema identity (`schema_version`, `fields_sha256`, `schema_sha256`) baked at deploy time. Use `scripts/check_schema_identity.py` to catch a stale deploy before submitters start 422-ing. |

Unpublished (incomplete) submissions never appear on any read route.

## Caps and limits

| Cap | Value | Error code |
|---|---|---|
| Archive total | 8 MB (11 chunks max) | `archive_too_large` / `too_many_chunks` |
| Single chunk | 768 KB | `chunk_too_large` |
| JSON summary payload | 256 KB | `payload_too_large` |
| Proof of work | >= 18 leading zero bits of sha256(`<sha>:<nonce>`) | `pow_missing` / `pow_invalid` |

The server never decompresses, parses, or executes uploaded archive
bytes; chunks are hashed and stored opaquely, which keeps every request
well inside the 10 ms CPU budget of the free plan. The PII scan runs on
the JSON summary only: MAC addresses, IPv4/IPv6, UUIDs, serial numbers,
home paths, credential shapes, and mDNS-style hostnames are refused
(`422 pii_detected`), never stored. Archive defense rests on the
collector's local redaction (see `scripts/collect_common.py`).

Incomplete submissions are garbage-collected by the hourly cron
(`17 * * * *`) after **7 days** without a completed upload; the same
cron rebuilds the cached index/dataset responses. Bulk routes self-heal
by building the cache on first request if the cron has not run yet.

## One-time deploy

The account is shared with unrelated projects, so every name is
namespaced: worker `mlx-omarchy-community-data`, D1 database
`mlx-omarchy-community`. `wrangler.jsonc` contains no `account_id`
(wrangler reads `CLOUDFLARE_ACCOUNT_ID` from the environment) and no
secrets. The ONE value to fill in by hand is `database_id`, printed by
`wrangler d1 create` and pasted into `wrangler.jsonc`; it is not a
secret.

```sh
cd services/community-data
npm install

# Load the canonical account token (never commit or print it).
set -a; . ~/.config/cloudflare/thewarrens-co.env; set +a
export CLOUDFLARE_API_TOKEN=$CF_API_TOKEN CLOUDFLARE_ACCOUNT_ID=$CF_ACCOUNT_ID

# 1. Create the database, then paste the printed database_id into
#    wrangler.jsonc (replacing the OWNER_FILL_ME placeholder).
npx wrangler d1 create mlx-omarchy-community

# 2. Apply migrations to the remote database.
npx wrangler d1 migrations apply mlx-omarchy-community --remote

# 3. Deploy the worker.
npx wrangler deploy
```

workers.dev fronting 403s default non-browser user agents, which is why
the collector sends `User-Agent: mlx-omarchy-collector/1`.

## Redeploying after a schema change (schema v2, 2026-10)

The live worker validates submissions against the schema baked at
deploy time. A schema change (new payload fields, a new
`schema_version`) is only real once the worker is redeployed; until
then every current-collector submit 422s with `schema_invalid` (the
2026-09-17 `ane_port` incident). There is no CI deploy step: deploys
are manual and owned by **Joshua** (the Cloudflare account token lives
only on esper at `~/.config/cloudflare/thewarrens-co.env`; an agent
session may run the commands from esper with that env, but the run is
Joshua's call).

```sh
# From a checkout at the commit that changed the schema (esper):
cd services/community-data
npm install
bun test test/unit                       # must be green first
set -a; . ~/.config/cloudflare/thewarrens-co.env; set +a
export CLOUDFLARE_API_TOKEN=$CF_API_TOKEN CLOUDFLARE_ACCOUNT_ID=$CF_ACCOUNT_ID

# 1. Migrations BEFORE the code deploy. The deployed worker reads the
#    columns its code expects; deploying code that selects a column
#    the remote D1 lacks 500s every row read (2026-10-03 incident:
#    28563f55 selected pii_redacted pre-migration and every
#    /v1/results/<sha> returned 500 for ~25 min).
npx wrangler d1 migrations list mlx-omarchy-community --remote
npx wrangler d1 migrations apply mlx-omarchy-community --remote

# 2. Deploy the worker. RECORD THE PREVIOUS VERSION ID first — it is
#    the rollback target:
#      prev=$(npx wrangler deployments list | head -5)  # note the id
#      echo "$(date -u +%FT%TZ) deploy <new-id> prev=<prev-id>" >> deploy.log
#    Rollback is: npx wrangler rollback            # to the previous deployment
#    (or `npx wrangler rollback <version-id>` for a specific one).
npx wrangler deploy

# 3. Post-deploy smoke: schema, index, and three row reads; any
#    failure auto-rolls the worker back.
scripts/post_deploy_smoke.sh

npm run check:schema                     # compares /v1/schema to the repo files
```

`check:schema` (scripts/check_schema_identity.py) must print OK; if it
reports a stale deploy, the worker answered with the OLD identity —
re-run `wrangler deploy` and check again. Then confirm from any
machine: `curl -A omarchy-check
https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/schema`
must report `schema_versions: [1, 2]` and the `fields_sha256` from
`scripts/compute_schema_identity.py` at the deployed commit. The
scheduled `.github/workflows/community-data.yml` also runs the identity
check against the live worker on every mirror run and fails loudly on
drift.

## Local development and tests (no credentials needed)

```sh
npm install
npm test          # pure unit tests: validation, PII, PoW, hashing, caps
npm run smoke     # end-to-end over wrangler dev with local D1
```

The smoke run covers: full multi-chunk submission, resumed upload that
sends only the missing chunks, replay dedupe, chunk hash mismatch,
oversize archive, oversize chunk, weak and absent proof of work, PII
refusal, completing with a missing chunk, byte-identical archive
round-trip, cron cache rebuild, and GC of stale incomplete submissions.

## Repository mirror

The GitHub side consumes ONLY the public read routes above
(`/v1/results`, `/v1/results/<sha>`, `/v1/results/<sha>/archive`,
`/v1/dataset/latest.jsonl`). Submitters never open issues, PRs, or
discussions; their data reaches the repo through the mirror.
