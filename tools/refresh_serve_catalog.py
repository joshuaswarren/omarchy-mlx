#!/usr/bin/env python3
"""Maintainer-side availability refresh for the bundled serve catalog.

This tool runs on a maintainer machine against the Hugging Face metadata
API. User machines never call Hugging Face: their periodic refresh fetches
the committed catalog from this repository only.

Contract (schema owner: serve/mlx_omarchy_serve/catalog.py):

- For every entry the refresher fetches the repo head sha and the pinned
  revision's safetensors sizes, then rewrites ONLY `availability.size_bytes`
  and `availability.refreshed_at` (plus the top-level `generated_at` on a
  fully successful run). Every other byte of catalog content passes through
  untouched and is verified to do so before the write.
- The vetted revision is retained until requalification. Upstream drift
  (head != pinned revision) is reported and exits 3; it never rewrites the
  revision or qualification. Repinning to a new upstream revision is a
  manual commit that must reset qualification to untested.
- Any fetch or validation failure aborts with no write: last-known-good
  wins. Exit 0 clean, 1 fetch/error, 2 invalid catalog, 3 drift (written).
- The refresher never promotes a model to recommended, never edits
  qualification or priority, and never evaluates content fetched from the
  network: API payloads are mined for a sha and integer sizes only.

Usage:
    python3 tools/refresh_serve_catalog.py [--check] [--catalog PATH]
        [--api-base URL] [--timeout S]
"""

import argparse
import copy
import json
import os
import sys
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = REPO_ROOT / "serve" / "mlx_omarchy_serve" / "catalog.json"
DEFAULT_API_BASE = "https://huggingface.co/api/models"
USER_AGENT = "mlx-omarchy-catalog-refresh/1 (maintainer metadata tool; no code execution)"

EXIT_OK = 0
EXIT_FETCH_ERROR = 1
EXIT_INVALID = 2
EXIT_DRIFT = 3


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_validator():
    """Import the single strict-schema validator. No fallback exists here."""
    sys.path.insert(0, str(REPO_ROOT / "serve"))
    from mlx_omarchy_serve.catalog import validate_catalog  # noqa: PLC0415

    return validate_catalog


def fetch_json(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def safetensors_bytes(payload, only=None):
    """Sum *.safetensors blob sizes; None when no matching blob exists.

    `only` restricts the sum to exact sibling paths (multi-variant repos
    where the served artifact is one file, e.g. laya's root variant)."""
    total = 0
    seen = False
    for sibling in payload.get("siblings") or []:
        name = sibling.get("rfilename") or ""
        size = sibling.get("size")
        if only is not None and name not in only:
            continue
        if name.endswith(".safetensors") and isinstance(size, int):
            total += size
            seen = True
    return total if seen else None


def refresh(catalog_path, fetch, validate, api_base=DEFAULT_API_BASE, timeout=20,
            check=False, now=None):
    """Refresh availability fields in place; returns (exit_code, report).

    `fetch(url)` and `validate(obj)` are injected so the network-free core
    is testable; production main() wires the real fetch and the schema
    owner's validator.
    """
    now = now or utc_now()
    report = []

    try:
        raw = Path(catalog_path).read_text(encoding="utf-8")
        catalog = json.loads(raw)
    except (OSError, ValueError) as exc:
        return EXIT_INVALID, [f"cannot read catalog: {exc}"]
    try:
        validate(catalog)
    except Exception as exc:
        return EXIT_INVALID, [f"catalog rejected by schema validator: {exc}"]

    updated = copy.deepcopy(catalog)
    drifted = []
    changed = False
    for entry in updated["models"]:
        repo = entry["repo"]
        revision = entry["revision"]
        try:
            head = fetch(f"{api_base}/{repo}")
            head_sha = head.get("sha")
            pinned = fetch(f"{api_base}/{repo}?blobs=true&revision={revision}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                report.append(
                    f"{entry['id']}: pinned revision {revision} is GONE upstream "
                    f"(HTTP 404); availability not updated, catalog not written"
                )
            else:
                report.append(f"{entry['id']}: fetch failed: HTTP {exc.code}")
            return EXIT_FETCH_ERROR, report
        except Exception as exc:  # network, timeout, bad JSON
            report.append(f"{entry['id']}: fetch failed: {exc}")
            return EXIT_FETCH_ERROR, report

        size = safetensors_bytes(
            pinned, (entry.get("extension") or {}).get("availability_files"))
        if size is None:
            report.append(f"{entry['id']}: no safetensors sizes in API payload; abort")
            return EXIT_FETCH_ERROR, report

        if head_sha is not None and head_sha != revision:
            drifted.append(entry["id"])
            report.append(
                f"{entry['id']}: DRIFT upstream head {head_sha} != pinned "
                f"{revision}; revision and qualification retained, maintainer "
                f"must repin (which resets qualification) or requalify"
            )

        if entry.get("availability", {}).get("size_bytes") == size:
            report.append(f"{entry['id']}: size unchanged ({size})")
            continue
        changed = True
        entry.setdefault("availability", {})["size_bytes"] = size
        entry["availability"]["refreshed_at"] = now
        report.append(f"{entry['id']}: availability.size_bytes={size} at pinned revision")

    # Zero-noise: a pure verification pass (every size already current)
    # writes nothing -- refreshed_at/generated_at tick only on real
    # metadata change, so the CI workflow never opens a cosmetic PR.
    if not changed:
        report.append(
            "all availability sizes already current; catalog not written"
        )
        return (EXIT_DRIFT if drifted else EXIT_OK), report

    # Defense in depth: prove only allowed fields moved.
    for old, new in zip(catalog["models"], updated["models"]):
        old_rest = {k: v for k, v in old.items() if k != "availability"}
        new_rest = {k: v for k, v in new.items() if k != "availability"}
        if old_rest != new_rest:
            raise AssertionError(f"refresher mutated vetted fields of {old.get('id')}")
    if {k: v for k, v in catalog.items() if k not in ("generated_at", "models")} != {
        k: v for k, v in updated.items() if k not in ("generated_at", "models")
    }:
        raise AssertionError("refresher mutated catalog outside models/availability")

    updated["generated_at"] = now

    if check:
        report.append("check mode: no write")
        return (EXIT_DRIFT if drifted else EXIT_OK), report

    # Write back with the escaping the file was read with: a catalog stored
    # ASCII-escaped stays ASCII-escaped, so a refresh diffs as metadata only
    # (PR 475 review at 9aeae2bd: em-dash escapes were rewritten as literal
    # characters, a content change the metadata-only claim excludes).
    ensure_ascii = not any(ch > "\x7f" for ch in raw)
    text = json.dumps(updated, indent=2, ensure_ascii=ensure_ascii) + "\n"
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(Path(catalog_path).parent),
        prefix=".catalog-refresh-", delete=False,
    )
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, str(catalog_path))
    except BaseException:
        os.unlink(handle.name)
        raise
    report.append(f"wrote {catalog_path} (generated_at={now})")
    return (EXIT_DRIFT if drifted else EXIT_OK), report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--check", action="store_true", help="report only; no write")
    args = parser.parse_args(argv)

    try:
        validate = load_validator()
    except ImportError as exc:
        print(f"schema validator unavailable ({exc}); refusing to run without "
              "the single strict validator", file=sys.stderr)
        return EXIT_INVALID

    code, report = refresh(
        args.catalog, lambda url: fetch_json(url, args.timeout), validate,
        api_base=args.api_base.rstrip("/"), timeout=args.timeout, check=args.check,
    )
    for line in report:
        print(line, file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
