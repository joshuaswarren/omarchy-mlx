#!/usr/bin/env python3
"""Compare the live worker's schema identity to the repo file.

Usage:
  scripts/check_schema_identity.py [--url URL]

URL defaults to https://mlx-omarchy-community-data.joshua-s-warren.workers.dev.

Exits 2 if the live worker reports a different schema_versions,
fields_sha256 or schema_sha256 than the bundled schema, and 1 if the
worker cannot be reached or its reply is not a JSON object. Exit 2
catches the failure mode behind the 2026-09-17 ane_port 422: the worker
was deployed from a revision that pre-dated the schema field, so the
JSON validator silently rejected every current-collector submit. Wire it
into pre-deploy and cron health checks.
"""

import argparse
import hashlib
import http.client
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "schema"
DEFAULT_URL = (
    "https://mlx-omarchy-community-data.joshua-s-warren.workers.dev"
)

USER_AGENT = "mlx-omarchy-schema-check/1"


def expected() -> dict:
    raws = [p.read_bytes() for p in sorted(SCHEMA_DIR.glob("payload-v1*.schema.json"))]
    # raws is sorted, so raws[0] is the e2e sibling; read fields from
    # both but the version list from the MAIN payload schema.
    main = next(json.loads(raw) for raw in raws
                if json.loads(raw)["properties"]["kind"].get("enum")
                == ["quick", "deep"])
    fields = sorted(set().union(
        *(json.loads(raw)["properties"].keys() for raw in raws)))
    version_prop = main["properties"]["schema_version"]
    versions = version_prop["enum"] if "enum" in version_prop \
        else [version_prop["const"]]
    return {
        "schema_versions": versions,
        "fields_sha256": hashlib.sha256(
            "\n".join(fields).encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(b"".join(raws)).hexdigest(),
    }


def live(url: str) -> dict:
    req = urllib.request.Request(
        f"{url}/v1/schema", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        got = json.loads(resp.read())
    if not isinstance(got, dict):
        raise ValueError(f"reply is {type(got).__name__}, not a JSON object")
    return got


def same_versions(want: list, have: object) -> bool:
    # bool is an int subclass and 1.0 == 1, so [True, 2.0] would sort
    # equal to [1, 2]; the worker sends plain integers.
    return (isinstance(have, list)
            and all(type(v) is int for v in have)
            and sorted(want) == sorted(have))


def shown(value: object) -> str:
    # Python 3.14 can decode a reply nested deeper than it can repr.
    try:
        return str(value)
    except RecursionError:
        return f"<{type(value).__name__} nested too deeply to print>"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL,
                    help="worker URL (default: %(default)s)")
    args = ap.parse_args(argv)
    url = args.url.rstrip("/")
    exp = expected()
    try:
        got = live(url)
    except (OSError, http.client.HTTPException, ValueError,
            OverflowError, RecursionError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()  # urlopen raised before the with block owned it
        print(f"[schema-identity] cannot read {url}/v1/schema: {exc}",
              file=sys.stderr)
        return 1
    mismatches = []
    for key in ("schema_versions", "fields_sha256", "schema_sha256"):
        want, have = exp[key], got.get(key)
        same = same_versions(want, have) \
            if key == "schema_versions" else want == have
        if not same:
            mismatches.append(f"  {key}: expected={want} live={shown(have)}")
    if mismatches:
        print(
            f"[schema-identity] STALE DEPLOY at {url}\n"
            + "\n".join(mismatches),
            file=sys.stderr,
        )
        return 2
    print(f"[schema-identity] OK live={url} fields_sha256={got['fields_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
