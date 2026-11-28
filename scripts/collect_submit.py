#!/usr/bin/env python3
"""Upload path for the mlx-omarchy contributor diagnostics archive.

The only network code in the collectors lives here. `collect_quick.py`
never imports this module; `collect_deep.py` imports it lazily, only
after the user has seen the preview and asked to submit explicitly.

Wire protocol (the Cloudflare Worker in services/community-data/):

  POST <base>/v1/submit
       body: {"schema_version", "kind", "content_sha256", "payload",
              "archive": {"total_bytes", "chunk_bytes", "chunk_count",
                          "chunk_sha256": [<hex>...]} | null,
              "pow": {"nonce", "difficulty"}}
       200 -> {"status": "awaiting_chunks"|"stored"|"duplicate",
               "content_sha256", "missing_chunks": [...], "receipt_url"}
  POST <base>/v1/submit/<sha>/chunk/<idx>   body: one raw chunk
       200 -> {"status": "stored"|"duplicate", "idx",
               "missing_chunks": [...]}
  POST <base>/v1/submit/<sha>/complete
       200 -> {"status": "stored"|"duplicate", "receipt_url"}
       409 -> {"error": "incomplete", "missing_chunks": [...]}
  GET  <base>/v1/submit/<sha>               dedup probe
       200 -> {"status": "duplicate"}; 404 -> not present

Only the already-redacted payload and archive are sent. Uploads are
resumable: a re-run repeats the initiate call and uploads only the
chunks the server reports as missing. The proof-of-work nonce is bound
to the archive hash, so a token cannot be reused for other content.

A custom User-Agent is required: workers.dev fronting 403s the default
Python-urllib user agent.

Every failure raises SubmitError. The caller keeps the local archive
and prints the error; a failed submit never destroys local output.
"""

import hashlib
import http.client
import inspect
import json
import os
import re
import reprlib
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_TIMEOUT = 60

# Must mirror services/community-data/src/caps.ts.
CHUNK_BYTES = 768 * 1024
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024
# Client-only cap on a single HTTP response (or error) body from the
# endpoint. All v1 responses are small JSON objects; capping at
# MAX_PAYLOAD_BYTES keeps diagnostic errors intact while refusing
# oversized streams before they fill memory.
MAX_RESPONSE_BYTES = MAX_PAYLOAD_BYTES
_READ_CHUNK_BYTES = 64 * 1024
_DECIMAL_RE = re.compile(r"[0-9]+")
POW_DIFFICULTY = 18

# Client-only: the most a pow_invalid answer may ask for. The server
# accepts up to 64 bits, but solve_pow is pure Python and costs about
# 2**bits hashes: 24 bits is seconds on one core, and each extra bit
# doubles it. A higher ask, from a hostile or broken server, is refused
# instead of hanging the collector.
MAX_POW_DIFFICULTY = 24

USER_AGENT = "mlx-omarchy-collector/1"


class SubmitError(RuntimeError):
    pass


def _header_values(headers, name):
    """All values for header `name`, matched case-insensitively."""
    if headers is None:
        return []
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        return list(get_all(name) or ())
    items = getattr(headers, "items", None)
    if callable(items):
        target = name.lower()
        return [v for k, v in items()
                if isinstance(k, str) and k.lower() == target]
    get = getattr(headers, "get", None)
    if callable(get):
        value = get(name)
        return [] if value is None else [value]
    return []


def _header_get(headers, name):
    values = _header_values(headers, name)
    return values[0] if values else None


def _check_content_length(headers, limit):
    """Refuse responses whose strict-ASCII-digit Content-Length exceeds limit."""
    max_digits = len(str(limit))
    for raw in _header_values(headers, "Content-Length"):
        if not isinstance(raw, str):
            continue
        for part in raw.split(","):
            token = part.strip(" \t")
            if not _DECIMAL_RE.fullmatch(token):
                continue
            digits = token.lstrip("0") or "0"
            if len(digits) > max_digits or int(digits) > limit:
                shown = digits if len(digits) <= 32 else digits[:32] + "..."
                raise SubmitError(
                    f"endpoint response too large: Content-Length {shown} "
                    f"exceeds {limit} bytes")


def _read_accepts_size(read_fn):
    """True unless read_fn explicitly takes no positional arguments."""
    try:
        sig = inspect.signature(read_fn)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL):
            return True
    return False


def _remaining_framed_bytes(resp):
    """Unconsumed Content-Length bytes left on an HTTPResponse, or 0."""
    for obj in (resp, getattr(resp, "fp", None)):
        if obj is None:
            continue
        rem = getattr(obj, "length", None)
        if type(rem) is int:
            return rem if rem > 0 else 0
    return 0


def _bytes_len(chunk):
    return chunk.nbytes if isinstance(chunk, memoryview) else len(chunk)


def _read_bounded(resp, headers, limit=MAX_RESPONSE_BYTES):
    """Read at most `limit` bytes from `resp`, refusing oversized or
    truncated framed bodies with SubmitError."""
    _check_content_length(headers, limit)
    read_fn = resp.read
    if _read_accepts_size(read_fn):
        buf = bytearray()
        remaining = limit + 1
        while remaining > 0:
            chunk = read_fn(min(_READ_CHUNK_BYTES, remaining))
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise SubmitError("endpoint returned non-bytes response body")
            chunk_len = _bytes_len(chunk)
            if chunk_len == 0:
                break
            if len(buf) + chunk_len > limit:
                raise SubmitError(
                    f"endpoint response too large: exceeded {limit} bytes")
            buf += chunk.tobytes() if isinstance(chunk, memoryview) else chunk
            remaining = (limit + 1) - len(buf)
        data = bytes(buf)
    else:
        data = read_fn()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise SubmitError("endpoint returned non-bytes response body")
        if _bytes_len(data) > limit:
            raise SubmitError(
                f"endpoint response too large: exceeded {limit} bytes")
        data = bytes(data)
    short = _remaining_framed_bytes(resp)
    if short > 0:
        raise SubmitError(
            f"endpoint response truncated: {short} bytes remaining")
    return data


def _close_quietly(obj, *, is_cm=False):
    """Close a response or context manager without masking an active error."""
    if obj is None:
        return
    try:
        if is_cm:
            exit_fn = getattr(obj, "__exit__", None)
            if callable(exit_fn):
                exit_fn(None, None, None)
                return
        close_fn = getattr(obj, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass


def _redirect_target_hint(headers):
    """Safe scheme://host summary of a printable http(s) Location, or None."""
    location = _header_get(headers, "Location")
    if (not isinstance(location, str) or not location
            or any(ch.isspace() or not ch.isprintable() for ch in location)):
        return None
    try:
        parts = urllib.parse.urlsplit(location)
        parts.port  # ValueError on a non-numeric or out-of-range port
    except ValueError:
        return None
    host = parts.hostname
    if parts.scheme not in ("http", "https") or not host or len(host) > 255:
        return None
    rendered = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{rendered}"


def _safe_request_id(headers):
    raw = _header_get(headers, "cf-ray")
    if (not isinstance(raw, str) or not raw or len(raw) > 128
            or any(ch.isspace() or not ch.isprintable() for ch in raw)):
        return None
    return raw


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse 3xx redirects without reading the body or forwarding headers."""

    def http_error_302(self, req, fp, code, msg, headers):
        _close_quietly(fp)
        request_id = _safe_request_id(headers)
        target = _redirect_target_hint(headers)
        dest = f" to {target}" if target else ""
        raise SubmitError(
            f"endpoint redirected{dest} (HTTP {code}, request "
            f"{request_id or 'unknown'}); refusing to follow redirects")

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = (
        http_error_302)


def _default_urlopen():
    return urllib.request.build_opener(_NoRedirectHandler).open


def _request(urlopen, req, timeout):
    """(status, body, request_id): the request id is the server's
    cf-ray header, so a contributor report is actionable."""
    try:
        try:
            cm = urlopen(req, timeout=timeout)
            try:
                enter_fn = getattr(cm, "__enter__", None)
                resp = enter_fn() if callable(enter_fn) else cm
                headers = resp.headers
                body = _read_bounded(resp, headers)
                return resp.status, body, _header_get(headers, "cf-ray")
            finally:
                _close_quietly(cm, is_cm=True)
        except urllib.error.HTTPError as exc:
            try:
                headers = exc.headers
                body = _read_bounded(exc, headers)
                return exc.code, body, _header_get(headers, "cf-ray")
            finally:
                _close_quietly(exc)
    # HTTPException: protocol errors, IncompleteRead included. ValueError:
    # a redirect Location urllib cannot parse.
    except (urllib.error.URLError, OSError, TimeoutError,
            http.client.HTTPException, ValueError) as exc:
        raise SubmitError(f"endpoint unreachable: {exc}") from exc


def _server_note(status, body, request_id, cap=2048):
    """Full server response for error messages, plus the request id."""
    text = body.decode("utf-8", "replace").strip()
    if len(text) > cap:
        text = text[:cap] + "…"
    note = f"server said: {text}" if text else "server sent an empty body"
    return f"{note} (HTTP {status}, request {request_id or 'unknown'})"


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def leading_zero_bits(hex_digest):
    bits = 0
    for ch in hex_digest:
        nibble = int(ch, 16)
        if nibble == 0:
            bits += 4
            continue
        if nibble < 2:
            bits += 3
        elif nibble < 4:
            bits += 2
        elif nibble < 8:
            bits += 1
        break
    return bits


def solve_pow(content_sha256, difficulty=POW_DIFFICULTY):
    """Find a nonce so sha256("<sha>:<nonce>") has >= difficulty zero bits."""
    nonce = 0
    while True:
        digest = hashlib.sha256(f"{content_sha256}:{nonce}".encode()).hexdigest()
        if leading_zero_bits(digest) >= difficulty:
            return str(nonce)
        nonce += 1


def chunk_archive(data, chunk_bytes=CHUNK_BYTES):
    """Deterministic split; returns [(idx, bytes, sha256), ...]."""
    chunks = []
    for idx, off in enumerate(range(0, len(data), chunk_bytes)):
        piece = data[off:off + chunk_bytes]
        chunks.append((idx, piece, sha256_hex(piece)))
    return chunks


def _decode(body):
    """The response object, or {} for anything that is not a JSON object."""
    try:
        obj = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _retry_difficulty(status, body, request_id, decoded, difficulty):
    """The higher difficulty a pow_invalid answer asks for, or None.

    SubmitError when the ask is above MAX_POW_DIFFICULTY.
    """
    if status != 403 or decoded.get("error") != "pow_invalid":
        return None
    detail = decoded.get("detail")
    wanted = detail.get("min_difficulty") if isinstance(detail, dict) else None
    if type(wanted) is not int or wanted <= difficulty:
        return None
    if wanted > MAX_POW_DIFFICULTY:
        raise SubmitError(
            f"endpoint asked for proof-of-work difficulty {wanted}, above "
            f"the collector's limit of {MAX_POW_DIFFICULTY}: "
            f"{_server_note(status, body, request_id)}")
    return wanted


def _missing_chunks(decoded, chunk_count):
    """Validated chunk indexes from an initiate answer."""
    missing = decoded.get("missing_chunks", [])
    if not isinstance(missing, list) or not all(
            type(idx) is int and 0 <= idx < chunk_count for idx in missing):
        raise SubmitError(
            f"endpoint returned invalid missing_chunks {reprlib.repr(missing)} "
            f"for {chunk_count} chunks")
    return missing


def _headers(token, extra=None):
    headers = {
        "User-Agent": USER_AGENT,
        "Authorization": f"Bearer {token}",
    } if token else {"User-Agent": USER_AGENT}
    if extra:
        headers.update(extra)
    return headers


# Request-only header carrying the redactor's derived short host names.
# The service scans the summary against them and refuses a payload that
# still carries one; the header itself is never stored or echoed, so
# the alias list does not enter the public record.
ALIAS_HEADER = "X-MLX-Host-Aliases"


def alias_header_value(aliases):
    """One header value from `collect_common.host_aliases` output."""
    clean = [a for a in (aliases or [])
             if isinstance(a, str) and 4 <= len(a) <= 64][:16]
    return ",".join(clean) if clean else None


def _endpoint_base(endpoint):
    """The endpoint without its trailing slash.

    SubmitError unless it is an http or https URL with a host and no
    whitespace, control characters, query or fragment: the collectors
    catch only SubmitError, and urllib raises ValueError on a mistyped
    URL before any request is sent.
    """
    try:
        if any(ch.isspace() or not ch.isprintable() for ch in endpoint):
            raise ValueError("whitespace or control character")
        # Checked on the raw string: urlsplit drops an empty "?" or "#".
        if "?" in endpoint or "#" in endpoint:
            raise ValueError("query and fragment are not allowed")
        parts = urllib.parse.urlsplit(endpoint)
        parts.port  # ValueError on a non-numeric or out-of-range port
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("expected an http:// or https:// URL with a host")
    except ValueError as exc:
        raise SubmitError(
            f"invalid submit endpoint {reprlib.repr(endpoint)}: {exc}") from exc
    return endpoint.rstrip("/")


def submit(endpoint, data, payload, timeout=DEFAULT_TIMEOUT, urlopen=None,
           token=None, aliases=None):
    """Submit one archive; returns {"url", "deduplicated", "status"}.

    `urlopen` is injectable for tests. Resumable: re-invoking after a
    mid-upload failure re-initiates and sends only the missing chunks.
    """
    if len(data) > MAX_ARCHIVE_BYTES:
        raise SubmitError(
            f"archive too large for the public endpoint: {len(data)} > "
            f"{MAX_ARCHIVE_BYTES} bytes; it stays local")
    if urlopen is None:
        urlopen = _default_urlopen()
    base = _endpoint_base(endpoint)
    digest = sha256_hex(data)
    chunks = chunk_archive(data)

    status, body, request_id = _request(
        urlopen,
        urllib.request.Request(
            f"{base}/v1/submit/{digest}", headers=_headers(token)),
        timeout)
    if status == 200:
        return _receipt(_decode(body), deduplicated=True, status=status)
    if status != 404:
        raise SubmitError(
            f"dedup probe failed with HTTP {status}: "
            f"{_server_note(status, body, request_id)}")

    difficulty = POW_DIFFICULTY
    initiate_headers = {"Content-Type": "application/json"}
    alias_value = alias_header_value(aliases)
    if alias_value:
        initiate_headers[ALIAS_HEADER] = alias_value
    for attempt in range(2):
        initiate = {
            "schema_version": payload.get("schema_version", 1),
            "kind": payload.get("kind", "deep"),
            "content_sha256": digest,
            "payload": payload,
            "archive": {
                "total_bytes": len(data),
                "chunk_bytes": CHUNK_BYTES,
                "chunk_count": len(chunks),
                "chunk_sha256": [c[2] for c in chunks],
            },
            "pow": {
                "nonce": solve_pow(digest, difficulty),
                "difficulty": difficulty,
            },
        }
        status, body, request_id = _request(
            urlopen,
            urllib.request.Request(
                f"{base}/v1/submit",
                data=json.dumps(initiate).encode("utf-8"),
                headers=_headers(token, initiate_headers),
                method="POST"),
            timeout)
        decoded = _decode(body)
        wanted = _retry_difficulty(status, body, request_id, decoded,
                                   difficulty)
        if wanted is None:
            break
        difficulty = wanted

    if status != 200:
        raise SubmitError(
            f"initiate failed with HTTP {status}: "
            f"{_server_note(status, body, request_id)}")

    if decoded.get("status") == "duplicate":
        return _receipt(decoded, deduplicated=True, status=status)
    if decoded.get("status") == "stored":
        return _receipt(decoded, deduplicated=False, status=status)

    for idx in _missing_chunks(decoded, len(chunks)):
        piece = chunks[idx][1]
        cstatus, cbody, request_id = _request(
            urlopen,
            urllib.request.Request(
                f"{base}/v1/submit/{digest}/chunk/{idx}",
                data=piece,
                headers=_headers(token,
                                 {"Content-Type": "application/octet-stream"}),
                method="POST"),
            timeout)
        cdecoded = _decode(cbody)
        if cstatus != 200:
            raise SubmitError(
                f"chunk {idx} failed with HTTP {cstatus}: "
                f"{_server_note(cstatus, cbody, request_id)}; "
                f"re-run to resume")

    fstatus, fbody, request_id = _request(
        urlopen,
        urllib.request.Request(
            f"{base}/v1/submit/{digest}/complete",
            data=b"",
            headers=_headers(token),
            method="POST"),
        timeout)
    fdecoded = _decode(fbody)
    if fstatus == 409 and fdecoded.get("error") == "incomplete":
        raise SubmitError(
            f"server still misses chunks "
            f"{reprlib.repr(fdecoded.get('missing_chunks'))}: "
            f"{_server_note(fstatus, fbody, request_id)}; re-run to resume")
    if fstatus != 200:
        raise SubmitError(
            f"complete failed with HTTP {fstatus}: "
            f"{_server_note(fstatus, fbody, request_id)}")
    return _receipt(fdecoded, deduplicated=False, status=fstatus)


def submit_payload(endpoint, payload, timeout=DEFAULT_TIMEOUT, urlopen=None,
                   token=None, aliases=None):
    """Publish a payload-only report: no archive, one round trip.

    The quick report is small and self-describing, so the endpoint takes
    `archive: null` and publishes at initiate. Content is addressed by
    the sha256 of the canonical payload bytes, so re-running an unchanged
    machine deduplicates instead of adding a row.
    """
    if urlopen is None:
        urlopen = _default_urlopen()
    base = _endpoint_base(endpoint)
    body = json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_PAYLOAD_BYTES:
        raise SubmitError(
            f"report too large for the public endpoint: {len(body)} > "
            f"{MAX_PAYLOAD_BYTES} bytes; it stays local")
    digest = sha256_hex(body)

    status, raw, request_id = _request(
        urlopen,
        urllib.request.Request(
            f"{base}/v1/submit/{digest}", headers=_headers(token)),
        timeout)
    if status == 200:
        return _receipt(_decode(raw), deduplicated=True, status=status)
    if status != 404:
        raise SubmitError(
            f"dedup probe failed with HTTP {status}: "
            f"{_server_note(status, raw, request_id)}")

    difficulty = POW_DIFFICULTY
    decoded = {}
    quick_headers = {"Content-Type": "application/json"}
    alias_value = alias_header_value(aliases)
    if alias_value:
        quick_headers[ALIAS_HEADER] = alias_value
    for _attempt in range(2):
        initiate = {
            "schema_version": payload.get("schema_version", 1),
            "kind": payload.get("kind", "quick"),
            "content_sha256": digest,
            "payload": payload,
            "archive": None,
            "pow": {
                "nonce": solve_pow(digest, difficulty),
                "difficulty": difficulty,
            },
        }
        status, raw, request_id = _request(
            urlopen,
            urllib.request.Request(
                f"{base}/v1/submit",
                data=json.dumps(initiate).encode("utf-8"),
                headers=_headers(token, quick_headers),
                method="POST"),
            timeout)
        decoded = _decode(raw)
        wanted = _retry_difficulty(status, raw, request_id, decoded,
                                   difficulty)
        if wanted is None:
            break
        difficulty = wanted

    if status != 200:
        raise SubmitError(
            f"submit failed with HTTP {status}: "
            f"{_server_note(status, raw, request_id)}")
    return _receipt(decoded,
                    deduplicated=decoded.get("status") == "duplicate",
                    status=status)


def _receipt(body, deduplicated, status):
    url = body.get("receipt_url")
    # isprintable: the caller prints the URL, and a surrogate, newline or
    # terminal escape must not crash or rewrite that line.
    if not isinstance(url, str) or not url or not url.isprintable():
        raise SubmitError("endpoint returned no receipt URL")
    return {"url": url, "deduplicated": deduplicated, "status": status}


DEFAULT_ENDPOINT = "https://mlx-omarchy-community-data.joshua-s-warren.workers.dev"


def endpoint_from_args(args):
    """Resolve the endpoint: --submit flag first, then the environment."""
    if getattr(args, "submit", None):
        return args.submit
    return os.environ.get("MLX_OMARCHY_SUBMIT_URL") or None


def token_from_env():
    return os.environ.get("MLX_OMARCHY_SUBMIT_TOKEN") or None


def sys_stdin_isatty():
    return os.environ.get("MLX_OMARCHY_ASSUME_YES") != "1" and \
        sys.stdin.isatty()


def confirm_interactive(endpoint, archive_name, digest):
    """Ask for the exact word SUBMIT on a terminal; anything else declines."""
    if not sys_stdin_isatty():
        return False
    print(f"[submit] endpoint: {endpoint}")
    print(f"[submit] archive: {archive_name} sha256={digest}")
    print("[submit] only redacted report data and any attached archive are sent. "
          "Type SUBMIT to upload, anything else to keep it local:")
    try:
        answer = input().strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == "SUBMIT"
