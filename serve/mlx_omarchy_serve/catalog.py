"""Serve catalog: strict v1 schema, loader, and GitHub-only refresh.

The catalog is data only. It names vetted models (repo, pinned revision,
kind, quantization, memory facts, qualification state, curated priority).
Validation is closed-schema with hard size bounds; anything not in the
schema is an error, never a pass-through.

Refresh policy (user machines): fetch ONLY the public GitHub raw URL of the
catalog for this project. Conditional GET with ETag, short timeout, atomic
replace, last-known-good retained on any failure. No telemetry, no
background timer: the check runs when a serve/catalog command runs, at most
once per TTL. Offline/opt-out is honored via MLX_OMARCHY_OFFLINE=1 or
--offline. Availability/size columns are the only fields a refresh pipeline
may author; qualification, priority, and recommended change only through
manual commits of the vetted file (enforced socially here, enforced by the
maintainer-side refresher contract).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
CATALOG_MAX_BYTES = 256 * 1024
MAX_MODELS = 64
MAX_CONTEXT_TOKENS = 1_048_576
MIN_CONTEXT_TOKENS = 128

DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/"
    "serve/mlx_omarchy_serve/catalog.json"
)
DEFAULT_TTL_HOURS = 24.0
DEFAULT_TIMEOUT_SECONDS = 5.0

APPROVED_CATALOG_HOSTS = frozenset({"raw.githubusercontent.com"})
APPROVED_CATALOG_PREFIX = "/joshuaswarren/mlx-omarchy/"
DEV_URL_OVERRIDE_ENV = "MLX_OMARCHY_CATALOG_ALLOW_ANY_URL"


REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

KINDS = ("chat", "base", "decisions", "embed", "other")
QUANT_BITS = (2, 3, 4, 5, 6, 8)
QUANT_MODES = ("affine", "mxfp4", "nvfp4", "mxfp8")
FP_MODE_SHAPES = {  # mode -> (bits, group_size)
    "mxfp4": (4, 32),
    "nvfp4": (4, 16),
    "mxfp8": (8, 32),
}
ARCHES = ("t8103", "t6000", "t6001", "t6002", "t6021")
BACKENDS = ("mlx-lm", "omlx", "module")

QUAL_FIELDS = ("status", "receipt", "date")
MEM_FIELDS = ("weights_bytes", "kv_bytes_per_token", "peak_estimate_bytes")


class CatalogError(ValueError):
    """Catalog data violates the v1 schema or its bounds."""


def default_home() -> Path:
    return Path(os.environ.get("MLX_OMARCHY_HOME", Path.home() / ".local/share/mlx-omarchy"))


def env_offline() -> bool:
    return os.environ.get("MLX_OMARCHY_OFFLINE", "") not in ("", "0", "false")


def cache_path(home: Path | None = None) -> Path:
    return (home or default_home()) / "cache" / "recommended-catalog.json"


def bundled_path() -> Path:
    return Path(__file__).with_name("catalog.json")


def _fail(msg: str) -> None:
    raise CatalogError(msg)


def _check_str(value, field: str, cap: int, pattern: re.Pattern | None = None) -> None:
    if not isinstance(value, str) or not value:
        _fail(f"{field}: expected non-empty string")
    if len(value) > cap:
        _fail(f"{field}: longer than {cap} chars")
    if pattern is not None and pattern.match(value) is None:
        _fail(f"{field}: does not match required format")


def _check_int(value, field: str, lo: int, hi: int) -> None:
    # bool is an int subclass; reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        _fail(f"{field}: expected int in [{lo}, {hi}]")


def _check_keys(obj, expected: tuple, where: str, optional: tuple = ()) -> None:
    if not isinstance(obj, dict):
        _fail(f"{where}: expected object")
    extra = sorted(set(obj) - set(expected) - set(optional))
    missing = sorted(set(expected) - set(obj))
    if extra:
        _fail(f"{where}: unknown key(s) {extra}")
    if missing:
        _fail(f"{where}: missing key(s) {missing}")


def _check_qualification(group: str, obj) -> None:
    _check_keys(obj, QUAL_FIELDS, f"qualification.{group}")
    if obj["status"] not in ("qualified", "untested"):
        _fail(f"qualification.{group}.status: unknown status {obj['status']!r}")
    if obj["status"] == "qualified":
        _check_str(obj["receipt"], f"qualification.{group}.receipt", 128)
        _check_str(obj["date"], f"qualification.{group}.date", 10, DATE_PATTERN)
    else:
        for f in ("receipt", "date"):
            if obj[f] is not None:
                _fail(f"qualification.{group}.{f}: must be null unless status is qualified")


def validate_catalog(obj) -> None:
    """Validate a parsed catalog object against schema v1. Raises CatalogError."""
    _check_keys(obj, ("version", "generated_at", "source", "models"), "catalog")
    if isinstance(obj["version"], bool) or obj["version"] != SCHEMA_VERSION:
        _fail(f"catalog.version: expected {SCHEMA_VERSION}, got {obj['version']!r}")
    _check_str(obj["source"], "catalog.source", 256)
    if not obj["source"].startswith("https://"):
        _fail("catalog.source: must be an https URL")
    _check_iso(obj["generated_at"], "catalog.generated_at")

    models = obj["models"]
    if not isinstance(models, list) or not 1 <= len(models) <= MAX_MODELS:
        _fail(f"catalog.models: expected 1..{MAX_MODELS} entries")

    seen_ids: set[str] = set()
    seen_priorities: dict[str, set[int]] = {}
    for i, entry in enumerate(models):
        where = f"models[{i}]"
        _check_keys(
            entry,
            (
                "id", "repo", "revision", "kind", "license", "family", "priority",
                "quant", "memory", "context", "capability", "qualification",
                "recommended", "serve", "availability",
            ),
            where,
            optional=("extension",),
        )
        _check_str(entry["id"], f"{where}.id", 64, ID_PATTERN)
        if entry["id"] in seen_ids:
            _fail(f"{where}.id: duplicate id {entry['id']!r}")
        seen_ids.add(entry["id"])
        _check_str(entry["repo"], f"{where}.repo", 128, REPO_PATTERN)
        _check_str(entry["revision"], f"{where}.revision", 40, REVISION_PATTERN)
        if entry["kind"] not in KINDS:
            _fail(f"{where}.kind: unknown kind {entry['kind']!r}")
        if entry["license"] is not None:
            _check_str(entry["license"], f"{where}.license", 32)
        if entry["family"] is not None:
            _check_str(entry["family"], f"{where}.family", 64)
        _check_int(entry["priority"], f"{where}.priority", 1, 999)
        if entry["priority"] in seen_priorities.setdefault(entry["kind"], set()):
            _fail(f"{where}.priority: duplicate priority {entry['priority']} in kind {entry['kind']!r}")
        seen_priorities[entry["kind"]].add(entry["priority"])

        quant = entry["quant"]
        if quant is not None:  # null = unquantized (full-precision) checkpoint
            _check_keys(quant, ("bits", "group_size", "mode"), f"{where}.quant")
            mode = quant["mode"]
            if mode not in QUANT_MODES:
                _fail(f"{where}.quant.mode: unknown mode {mode!r}")
            if mode == "affine":
                _check_int(quant["bits"], f"{where}.quant.bits", 2, 8)
                if quant["bits"] not in QUANT_BITS:
                    _fail(f"{where}.quant.bits: {quant['bits']} not a supported affine bitwidth")
                if quant["group_size"] is None:
                    _fail(f"{where}.quant.group_size: required for affine")
                _check_int(quant["group_size"], f"{where}.quant.group_size", 16, 128)
                if quant["group_size"] not in (32, 64, 128):
                    _fail(f"{where}.quant.group_size: affine groups are 32/64/128")
            else:
                bits, group = FP_MODE_SHAPES[mode]
                if quant["bits"] != bits or quant["group_size"] != group:
                    _fail(f"{where}.quant: {mode} requires bits={bits}, group_size={group}")

        extension = entry.get("extension")
        if extension is not None and not isinstance(extension, dict):
            _fail(f"{where}.extension: must be an object when present")

        mem = entry["memory"]
        _check_keys(mem, MEM_FIELDS, f"{where}.memory")
        _check_int(mem["weights_bytes"], f"{where}.memory.weights_bytes", 1, 2**48)
        if mem["kv_bytes_per_token"] is not None:
            _check_int(mem["kv_bytes_per_token"], f"{where}.memory.kv_bytes_per_token", 1, 2**24)
        if mem["peak_estimate_bytes"] is not None:
            _check_int(mem["peak_estimate_bytes"], f"{where}.memory.peak_estimate_bytes", 1, 2**48)

        ctx = entry["context"]
        _check_keys(ctx, ("max_tokens",), f"{where}.context")
        if ctx["max_tokens"] is not None:
            _check_int(ctx["max_tokens"], f"{where}.context.max_tokens", MIN_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS)

        cap = entry["capability"]
        _check_keys(cap, ("arch", "min_mem_gib"), f"{where}.capability")
        arch = cap["arch"]
        if arch is not None:
            if not isinstance(arch, list) or not arch or not all(a in ARCHES for a in arch):
                _fail(f"{where}.capability.arch: entries must be from {ARCHES}")
        if cap["min_mem_gib"] is not None:
            _check_int(cap["min_mem_gib"], f"{where}.capability.min_mem_gib", 1, 512)

        qual = entry["qualification"]
        _check_keys(qual, ("generation", "http", "managed"), f"{where}.qualification")
        _check_qualification("generation", qual["generation"])
        _check_qualification("http", qual["http"])
        _check_qualification("managed", qual["managed"])

        if not isinstance(entry["recommended"], bool):
            _fail(f"{where}.recommended: expected bool")
        if entry["recommended"] and qual["generation"]["status"] != "qualified":
            _fail(f"{where}.recommended: requires generation qualification")

        serve = entry["serve"]
        if serve is not None:
            _check_keys(serve, ("backend", "module"), f"{where}.serve")
            if serve["backend"] not in BACKENDS:
                _fail(f"{where}.serve.backend: unknown backend {serve['backend']!r}")
            if serve["backend"] == "module":
                _check_str(serve["module"], f"{where}.serve.module", 128)
            elif serve["module"] is not None:
                _fail(f"{where}.serve.module: must be null unless backend is module")

        avail = entry["availability"]
        _check_keys(avail, ("size_bytes", "refreshed_at"), f"{where}.availability")
        if avail["size_bytes"] is not None:
            _check_int(avail["size_bytes"], f"{where}.availability.size_bytes", 1, 2**48)
        if avail["refreshed_at"] is not None:
            _check_iso(avail["refreshed_at"], f"{where}.availability.refreshed_at")


def _check_iso(value, field: str) -> None:
    _check_str(value, field, 32)
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{field}: not an ISO-8601 timestamp")


def validate_file(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if len(raw) > CATALOG_MAX_BYTES:
        raise CatalogError(f"{path}: {len(raw)} bytes exceeds {CATALOG_MAX_BYTES}")
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CatalogError(f"{path}: invalid JSON ({exc})") from exc
    validate_catalog(obj)
    return obj


def _is_fresh(path: Path, ttl_hours: float, now: float | None) -> bool:
    if not path.is_file():
        return False
    ts = now if now is not None else datetime.now(timezone.utc).timestamp()
    age_hours = (ts - path.stat().st_mtime) / 3600.0
    return age_hours < ttl_hours


def load_catalog(home: Path | None = None, prefer_cache: bool = True) -> dict:
    """Cache first (if valid), bundled fallback last. Never raises for a
    damaged cache: fall back instead, a broken refresh must not break serve."""
    candidates = []
    if prefer_cache:
        candidates.append(cache_path(home))
    candidates.append(bundled_path())
    for path in candidates:
        try:
            return validate_file(path)
        except (OSError, CatalogError):
            continue
    raise CatalogError("no usable catalog: cache missing and bundled catalog invalid")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def validate_catalog_url(url: str) -> None:
    """Only the project's public GitHub raw URLs are approved catalog sources.
    Anything else needs the explicit non-default dev override env."""
    parsed = urllib.parse.urlsplit(url)
    approved = (parsed.scheme == "https"
                and parsed.hostname in APPROVED_CATALOG_HOSTS
                and parsed.path.startswith(APPROVED_CATALOG_PREFIX))
    if approved:
        return
    if os.environ.get(DEV_URL_OVERRIDE_ENV) == "1":
        print(f"warning: {DEV_URL_OVERRIDE_ENV}=1 — using non-default catalog URL {url}",
              file=sys.stderr)
        return
    raise CatalogError(
        f"catalog URL {url!r} is not an approved public GitHub raw URL "
        f"(expected https://raw.githubusercontent.com{APPROVED_CATALOG_PREFIX}...); "
        f"export {DEV_URL_OVERRIDE_ENV}=1 only for explicit development use"
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None  # a catalog fetch must not silently follow a redirect


def refresh(
    home: Path | None = None,
    *,
    url: str | None = None,
    ttl_hours: float | None = None,
    timeout: float | None = None,
    offline: bool | None = None,
    now: float | None = None,
) -> dict:
    """Refresh the cached catalog from the public GitHub raw URL.

    Returns {"status": fresh|updated|not-modified|kept|skipped|fallback,
             "path": cache path, "detail": str}.
    - fresh: cache younger than TTL, no network performed.
    - skipped: offline/opt-out.
    - updated: fetched, validated, atomically installed.
    - not-modified: HTTP 304 against the stored ETag.
    - kept: fetch/validate failed; previous cache (or bundled) still in place.
    - fallback: kept, and no previous cache existed (bundled file serves).
    """
    dest = cache_path(home)
    url = url or os.environ.get("MLX_OMARCHY_CATALOG_URL") or DEFAULT_CATALOG_URL
    validate_catalog_url(url)
    ttl = float(os.environ["MLX_OMARCHY_CATALOG_TTL_HOURS"]) if "MLX_OMARCHY_CATALOG_TTL_HOURS" in os.environ else (DEFAULT_TTL_HOURS if ttl_hours is None else ttl_hours)
    timeout = DEFAULT_TIMEOUT_SECONDS if timeout is None else timeout
    if offline is None:
        offline = env_offline()

    if offline:
        return {"status": "skipped", "path": str(dest), "detail": "offline opt-out honored"}
    if _is_fresh(dest, ttl, now):
        return {"status": "fresh", "path": str(dest), "detail": f"cache younger than {ttl:g}h"}

    etag_path = dest.with_suffix(dest.suffix + ".etag")
    headers = {"User-Agent": "mlx-omarchy-serve-catalog"}
    try:
        if etag_path.is_file():
            stored = etag_path.read_text(encoding="utf-8").strip()
            if stored:
                headers["If-None-Match"] = stored
    except OSError:
        pass

    try:
        request = urllib.request.Request(url, headers=headers)
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise urllib.error.URLError(f"unexpected status {response.status}")
            body = response.read(CATALOG_MAX_BYTES + 1)
            new_etag = response.headers.get("ETag", "")
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            dest.touch()
            return {"status": "not-modified", "path": str(dest), "detail": "etag match"}
        status = "fallback" if not dest.is_file() else "kept"
        return {"status": status, "path": str(dest), "detail": f"fetch failed: {exc}"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        status = "fallback" if not dest.is_file() else "kept"
        return {"status": status, "path": str(dest), "detail": f"fetch failed: {exc}"}

    if len(body) > CATALOG_MAX_BYTES:
        return {"status": "kept" if dest.is_file() else "fallback", "path": str(dest),
                "detail": f"catalog exceeds {CATALOG_MAX_BYTES} bytes"}
    try:
        validate_catalog(json.loads(body))
    except (json.JSONDecodeError, UnicodeDecodeError, CatalogError) as exc:
        return {"status": "kept" if dest.is_file() else "fallback", "path": str(dest),
                "detail": f"fetched catalog rejected: {exc}"}

    _atomic_write(dest, body)
    if new_etag:
        _atomic_write(etag_path, new_etag.encode())
    elif etag_path.exists():
        etag_path.unlink()
    return {"status": "updated", "path": str(dest), "detail": f"{len(body)} bytes from {url}"}
