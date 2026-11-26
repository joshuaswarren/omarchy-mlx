#!/usr/bin/env python3
"""Build the community-data snapshot from the public read API.

Runs in the scheduled GitHub Actions mirror workflow. Downloads the two
summaries the Contract exposes (latest.jsonl and the results index),
caps their size, derives a small Markdown table, and writes:

  latest.jsonl   one JSON record per line, byte-identical to the API
  index.json     generated_at, schema_version, count, bytes, sha256
  SUMMARY.md     human table linking records and archives

Never downloads archive blobs; SUMMARY.md links to them instead.

Endpoint problems (absent, 404, unreachable, oversized) exit 0 without
writing anything, so a bad endpoint day never fails the repo's status.
Every tracked byte is API-derived, so unchanged data means no commit.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import query_community_data as q  # noqa: E402

MAX_JSONL_BYTES = 32 * 1024 * 1024
MAX_INDEX_BYTES = 4 * 1024 * 1024
FETCH_TIMEOUT = 60


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": q.USER_AGENT})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        return response.read()


def fetch_jsonl_capped(url, max_bytes):
    """latest.jsonl trimmed to whole lines within max_bytes.

    Returns (data, truncated). A partial trailing line is dropped, so
    the file always holds complete records.
    """
    request = urllib.request.Request(url, headers={"User-Agent": q.USER_AGENT})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        data = b""
        while len(data) <= max_bytes:
            block = response.read(1 << 16)
            if not block:
                return data, False
            data += block
    data = data[:max_bytes]
    cut = data.rfind(b"\n")
    if cut == -1:
        return b"", True
    return data[:cut + 1], True


def fetch_index(base):
    """({generated_at, schema_version, count}, or None on any trouble)."""
    url = f"{base}/v1/results"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": q.USER_AGENT})
        with urllib.request.urlopen(request,
                                    timeout=FETCH_TIMEOUT) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_INDEX_BYTES:
                return None
            body = response.read(MAX_INDEX_BYTES + 1)
        if len(body) > MAX_INDEX_BYTES:
            return None
        index = json.loads(body.decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    if not isinstance(index, dict):
        return None
    return {"generated_at": index.get("generated_at"),
            "schema_version": index.get("schema_version"),
            "count": index.get("count")}


def fmt(value):
    return "-" if value is None else str(value)


def write_summary(path, base, records, index):
    lines = [
        "# Community hardware data snapshot",
        "",
        f"Records: {len(records)}"
        f" | dataset generated_at: {fmt(index and index.get('generated_at'))}"
        f" | source: {base}",
        "",
        "Mirrored by `.github/workflows/community-data.yml` from the public",
        "read API. Summaries only; archive blobs stay on the endpoint.",
        "",
        "| hash | kind | chip | kernel | mesa | mlx-omarchy | bench |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        sha = q.record_sha(record) or "unknown"
        short = sha[:12]
        lines.append(
            f"| [{short}]({base}/v1/results/{sha}) "
            f"| {q.record_kind(record)} | {fmt(q.record_chip(record))} "
            f"| {fmt(q.record_kernel(record))} "
            f"| {fmt(q.record_mesa(record))} "
            f"| {fmt(q.record_mlx_version(record))} "
            f"| [{len(q.record_benchmarks(record))}]"
            f"({base}/v1/results/{sha}/archive) |")
    lines += [
        "",
        "Query this snapshot:",
        "",
        "```",
        "python3 scripts/query_community_data.py list",
        "python3 scripts/query_community_data.py --json --kind deep list",
        "python3 scripts/query_community_data.py show <sha256-prefix>",
        "python3 scripts/query_community_data.py compare --metric tflops",
        "```",
        "",
        "Redacted, allowlisted fields only. Submissions never touch the",
        "contributor graph: this branch is written only by"
        " github-actions[bot].",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_snapshot(base, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl, truncated = fetch_jsonl_capped(
        f"{base}/v1/dataset/latest.jsonl", MAX_JSONL_BYTES)
    records, _skipped = q.parse_jsonl(jsonl.decode("utf-8", "replace"))
    index = fetch_index(base)
    payload = {
        "generated_at": index and index.get("generated_at"),
        "schema_version": index and index.get("schema_version"),
        "count": len(records),
        "bytes": len(jsonl),
        "sha256": hashlib.sha256(jsonl).hexdigest(),
        "truncated": truncated,
        "source": base,
    }
    (out_dir / "latest.jsonl").write_bytes(jsonl)
    (out_dir / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_summary(out_dir / "SUMMARY.md", base, records, index)
    print(json.dumps({"status": "mirrored", **payload}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.environ.get(
        "MLX_OMARCHY_COMMUNITY_URL", q.DEFAULT_BASE_URL))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")
    try:
        build_snapshot(base, Path(args.out_dir))
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            ValueError) as exc:
        print(f"community-data mirror: endpoint unavailable, skipping "
              f"({exc})", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
