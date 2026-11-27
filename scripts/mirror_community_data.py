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
import contextlib
import hashlib
import http.client
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import query_community_data as q  # noqa: E402

MAX_JSONL_BYTES = q.MAX_JSONL_BYTES
MAX_INDEX_BYTES = 4 * 1024 * 1024
FETCH_TIMEOUT = 60


def fetch_jsonl_capped(url, max_bytes):
    """latest.jsonl trimmed to whole lines within max_bytes.

    Returns (data, truncated). A partial trailing line is dropped, so
    the file always holds complete records. A status other than 200 or
    a body cut short raises q.DatasetError.
    """
    request = urllib.request.Request(url, headers={"User-Agent": q.USER_AGENT})
    with q.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        data = q.read_response(response, max_bytes, truncate=True)
    if len(data) <= max_bytes:
        return data, False
    data = data[:max_bytes]
    cut = data.rfind(b"\n")
    if cut == -1:
        return b"", True
    return data[:cut + 1], True


def _typed(value, types):
    """value when it is an instance of types (never a bool), else None."""
    if isinstance(value, bool) or not isinstance(value, types):
        return None
    return value


def fetch_index(base):
    """({generated_at, schema_version, count}, or None on any trouble).

    A field of the wrong type reads as None: a nested list or object
    would bloat index.json and SUMMARY.md, or overflow the stack.
    """
    url = f"{base}/v1/results"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": q.USER_AGENT})
        with q.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            body = q.read_response(response, MAX_INDEX_BYTES)
        index = json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        exc.close()
        return None
    except (urllib.error.URLError, OSError, ValueError, RecursionError,
            http.client.HTTPException, q.DatasetError):
        return None
    if not isinstance(index, dict):
        return None
    return {"generated_at": _typed(index.get("generated_at"), str),
            "schema_version": _typed(index.get("schema_version"), (str, int)),
            "count": _typed(index.get("count"), int)}


def fmt(value):
    return "-" if value is None else str(value)


# "&" too: an entity such as &#64; would otherwise decode to "@".
_CELL_ESCAPES = str.maketrans({c: "\\" + c for c in "\\|[]<&"})
# Where GitHub would autolink a bare URL, www. host or email address.
_AUTOLINK = re.compile(
    r"(?<=:)(?=//)|(?<=www)(?=\.)|(?<=[\w.+-])(?=@[\w.-])", re.IGNORECASE)


def cell(value):
    """value as one line of table text: no new row, cell, link or tag."""
    text = " ".join(fmt(value).split()).translate(_CELL_ESCAPES)
    return _AUTOLINK.sub("\u200b", text)


def read_field(accessor, record):
    """accessor(record), or None when a value is too deep or too large."""
    try:
        return accessor(record)
    except (RecursionError, OverflowError):
        return None


def has_archive(record):
    """True when the record is expected to have a downloadable archive.

    Honors an explicit `archive` field when present; otherwise treats
    `quick` rows as archive-less.
    """
    if "archive" in record:
        return bool(record["archive"])
    return q.record_kind(record) not in ("quick", "mlx-omarchy-quick")


def write_files(out_dir, files):
    """Write each {name: bytes} into out_dir, renaming in dict order.

    Every file is written in full to a fresh hidden file before any is
    renamed over its name, so a failed write changes none of them, and a
    symlink at a name is replaced, never followed. A failed rename (a
    name that is now a directory, a disk gone read-only) leaves the
    names before it replaced, so callers put the file that marks a
    snapshot complete last.
    """
    staged = []
    try:
        for name, data in files.items():
            tmp = out_dir / f".{name}.{secrets.token_hex(8)}.part"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666)
            staged.append(tmp)
            with open(fd, "wb") as out:
                out.write(data)
        for tmp, name in zip(staged, files):
            os.replace(tmp, out_dir / name)
    except BaseException:
        for tmp in staged:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        raise


def summary_bytes(base, records, index):
    lines = [
        "# Community hardware data snapshot",
        "",
        f"Records: {len(records)}"
        f" | dataset generated_at: {cell(index and index.get('generated_at'))}"
        f" | source: {base}",
        "",
        "Mirrored by `.github/workflows/community-data.yml` from the public",
        "read API. Summaries only; archive blobs stay on the endpoint.",
        "",
        "| hash | kind | chip | kernel | mesa | mlx-omarchy | bench |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in records:
        sha = q.record_sha(record)
        if sha in (None, ".", ".."):  # "." and ".." are URL dot segments
            sha = "unknown"
        url = (f"{base}/v1/results/"
               f"{urllib.parse.quote(sha, safe='', errors='replace')}")
        benchmarks = read_field(q.record_benchmarks, record)
        bench_count = "-" if benchmarks is None else len(benchmarks)
        bench = (
            f"[{bench_count}]({url}/archive)"
            if has_archive(record) else str(bench_count)
        )
        lines.append(
            f"| [{cell(sha[:12]) or 'unknown'}]({url}) "
            f"| {cell(q.record_kind(record))} "
            f"| {cell(read_field(q.record_chip, record))} "
            f"| {cell(read_field(q.record_kernel, record))} "
            f"| {cell(read_field(q.record_mesa, record))} "
            f"| {cell(read_field(q.record_mlx_version, record))} "
            f"| {bench} |")
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
    # errors="replace": a JSON string may hold a lone surrogate.
    return "\n".join(lines).encode("utf-8", "replace")


def write_summary(path, base, records, index):
    write_files(path.parent, {path.name: summary_bytes(base, records, index)})


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
    # latest.jsonl last: the workflow commits only when it is non-empty.
    write_files(out_dir, {
        "index.json": (json.dumps(payload, indent=2, sort_keys=True)
                       + "\n").encode("utf-8"),
        "SUMMARY.md": summary_bytes(base, records, index),
        "latest.jsonl": jsonl,
    })
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
    except (urllib.error.URLError, OSError, ValueError,
            http.client.HTTPException, q.DatasetError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
        print(f"community-data mirror: endpoint unavailable, skipping "
              f"({exc})", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
