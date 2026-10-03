#!/usr/bin/env python3
"""Shared helpers for the mlx-omarchy contributor collectors.

`collect_quick.py` and `collect_deep.py` import this module. It owns the
three things both collectors must agree on:

- PII redaction (`Redactor`): usernames, hostnames, home paths, IP
  addresses, MAC addresses, serial numbers, UUIDs, and credential-shaped
  strings never reach an output. Every embedded command output passes
  through it, and its per-kind counts go into the manifest so a reader can
  see what was removed.
- bounded external commands (`run_tool`, `run_python_probe`): a missing or
  hanging tool is recorded data, never a crash, never an unbounded wait.
- deterministic packaging (`build_manifest`, `archive_bytes`): sorted
  members, fixed mtime and owner, gzip mtime 0, so the previewed manifest
  and the written archive agree byte for byte.

Nothing in this module talks to the network. The collectors never upload.
"""

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import getpass

SCHEMA_VERSION = 2
# v1 rows (pre-2026-10) stay readable: the community-data service
# accepts schema_version 1 and 2; v2 adds the ane_macos / ane_linux
# turn-on blocks (see docs/ane-turn-on-data.md) and the mailbox /
# reserved-memory / firmware additions to ane_port_detail.

MAX_STREAM_LINES = 400
MAX_STREAM_CHARS = 200_000
# A probe whose stdout IS the JSON payload needs headroom above the
# generic stream cap: capping mid-JSON makes the payload unparseable
# (the macOS ane-dt dump probe emitted 213,659 chars on a real T6002).
PROBE_STREAM_CHARS = 2_000_000

# Hostname pieces that are generic OS/project words, not machine
# identity. Redacting them would mangle `omarchy-*` paths and every
# `Linux ...` log line on default hosts — the same reason the
# whole-token hostname rule keeps `-` as a boundary.
GENERIC_HOST_PIECES = frozenset(
    ("host", "localhost", "local", "linux", "omarchy", "mac", "lan"))

# Payload fields whose values come from the device tree or IORegistry,
# not from the user: marketing names, chip/board identifiers, machine
# models. Restricted aliases (below) never redact inside these fields —
# `Apple MacBook Air (13-inch, M3, 2024)` must survive a host called
# `m3-air` or `neo` (#27 follow-up). Every other PII kind is still
# redacted inside them.
EXEMPT_NAME_FIELDS = frozenset((
    "model", "chip", "chip_name", "board", "board_id", "board_name",
    "compatible", "product", "product_name", "machine_model",
    "machine_name", "model_identifier", "hw_model", "marketing_name",
))

# Marketing-name words: an alias equal to one of these — or shorter
# than 6 characters, or a tNNNN chip id — is RESTRICTED: whole-word
# matches in free text only (logs, paths, streams), never the name
# fields above. The full hostname and long derived aliases are never
# restricted: they still never leak anywhere.
MODEL_TOKENS = frozenset((
    "m1", "m2", "m3", "m4", "m5", "m6", "pro", "max", "ultra", "air",
    "neo", "mac", "book", "macbook", "studio", "mini", "imac", "apple",
))
_MODEL_CHIP_RE = re.compile(r"t\d{4}")

# Device-tree / IORegistry property NAMES that carry machine identity:
# the serial family (serial-number, mlb-serial-number, board-serial,
# serial-index, IOPlatformSerialNumber), unique-chip/ecid/mlb, and
# UUID/UDID token names. Value blanking is not enough for these: the
# property and its shape survive, and a multi-cell value keeps every
# token after the first (2026-10-03 contributor reports: a serial
# fragment reached the endpoint and the worker refused the row). Such
# properties are REMOVED whole, and each removal is tallied in the
# redaction summary.
IDENTITY_PROP_RE = re.compile(
    r"(?i)(serial|(^|[-_,])mlb([-_,]|$)|ecid|unique-chip|udid|uuid)")


def is_identity_prop(name):
    """True when a property NAME matches the serial / unique-id family."""
    return bool(IDENTITY_PROP_RE.search(name))


def _restricted_alias(alias):
    """True when `alias` may only redact free text, never the name
    fields: short aliases and marketing-name words are exactly the
    tokens those fields are made of."""
    lowered = alias.lower()
    return (len(lowered) < 6 or lowered in MODEL_TOKENS
            or bool(_MODEL_CHIP_RE.fullmatch(lowered)))


def host_aliases(hostname):
    """Short names someone plausibly derives from `hostname`.

    The first dotted label itself, its `-` fragments of length >= 4,
    and digit-bearing truncation prefixes of the label down to length
    4 (`jw16` for `jw16mbp1` is exactly how short fleet aliases are
    born). Prefixes must contain a digit: an alphabetic 4-letter head
    like `dead` (of `deadbeef-live`) is a common word, and redacting
    it standalone would corrupt unrelated text. Generic pieces are
    dropped. Longest first, so a longer alias is never partially eaten
    by a shorter one.
    """
    aliases = set()
    first = (hostname or "").split(".")[0].lower()
    if len(first) < 4 or first in GENERIC_HOST_PIECES:
        return []
    for n in range(len(first), 3, -1):
        piece = first[:n].rstrip("-")
        if (len(piece) >= 4 and piece not in GENERIC_HOST_PIECES
                and any(c.isdigit() for c in piece)):
            aliases.add(piece)
    for part in first.split("-"):
        if len(part) >= 4 and part not in GENERIC_HOST_PIECES:
            aliases.add(part)
    return sorted(aliases, key=len, reverse=True)


def local_hostname():
    """This machine's host name; collectors import no network modules,
    so the only socket use lives here and in `Redactor`."""
    return socket.gethostname()


class Redactor:
    """Replace personally identifying strings with typed placeholders."""

    def __init__(self, hostname=None, username=None, home=None):
        self.hostname = hostname or socket.gethostname()
        try:
            self.username = username or getpass.getuser()
        except Exception:
            self.username = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        self.home = home or os.path.expanduser("~")
        self.counts = {}
        self._rules = self._build_rules()

    def _note(self, kind):
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _replace(self, kind, repl):
        def fn(_match):
            self._note(kind)
            return repl
        return fn

    # A dotted quad is not always an address: Vulkan reports
    # `conformanceVersion = 1.4.0.0`, and Apple reports boot firmware
    # versions like `iBoot-20712.1.2.0.0`, whose trailing quad is a
    # continuation of a dotted version chain. Keep the quad when the
    # text right before it is a version assignment or more dotted
    # octets; redact every other dotted quad.
    _VERSION_CONTEXT = re.compile(r"(?i)(?:version\s*[:=]\s*|\d+\.)$")

    def _ipv4_sub(self, match, field=None):
        if (isinstance(field, str) and field.lower().endswith("version")
                and not match.string[:match.start()].strip()):
            return match.group(0)
        line_start = match.string.rfind("\n", 0, match.start()) + 1
        prefix = match.string[line_start:match.start()]
        if self._VERSION_CONTEXT.search(prefix):
            return match.group(0)
        self._note("ipv4")
        return "[redacted-ip4]"

    def _alias_sub(self, match, field=None, restricted=False, kind="hostname_alias"):
        # A restricted alias (short, or a marketing-name word) never
        # touches the device-tree/IORegistry name fields; everything
        # else about it behaves like any other alias.
        if restricted and isinstance(field, str) \
                and field.lower() in EXEMPT_NAME_FIELDS:
            return match.group(0)
        self._note(kind)
        return "[host]"

    def _build_rules(self):
        rules = []

        def rx(pattern, kind, repl, flags=0):
            rules.append((re.compile(pattern, flags), self._replace(kind, repl)))

        # Cred-shaped assignments first, so a value that also matches a
        # later rule is already gone. Name=NAME VALUE=VALUE keeps the name.
        rx(
            r"(?i)\b([A-Za-z0-9_]*(?:token|secret|passwd|password|api_?key|"
            r"private_?key)[A-Za-z0-9_]*)\s*([:=])\s*(\"[^\"]*\"|\S+)",
            "credential",
            r"\1\2 [redacted]",
        )
        # Known credential shapes without a key name.
        rx(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b", "credential", "[redacted]")
        rx(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "credential", "[redacted]")
        rx(r"\bAKIA[0-9A-Z]{16}\b", "credential", "[redacted]")
        rx(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "credential", "[redacted]")
        rx(r"\bsk-[A-Za-z0-9_-]{20,}\b", "credential", "[redacted]")
        rx(r"\bBearer\s+\S+", "credential", "Bearer [redacted]")
        rx(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\b",
           "credential", "[redacted]")
        # Hardware identity.
        rx(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "mac", "[redacted-mac]")
        rx(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
           r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", "uuid", "[redacted-uuid]")
        rx(r"(?i)(\"serial(?:[-_]?number)?\"\s*:\s*\")[^\"]*(\")",
           "serial", r"\1[redacted]\2")
        rx(r"(?i)\b(serial[-_]?number)\s*([=:])\s*(\S+)",
           "serial", r"\1\2 [redacted]")
        # ioreg/macOS hardware identity: "IOPlatformSerialNumber" = "C02…"
        rx(r"(?i)\"?(IOPlatformSerialNumber|IOPlatformUUID|BoardID|"
           r"board-id|serial-number)\"?\s*=\s*\"[^\"]*\"",
           "serial", r"\1 = [redacted]")
        # Bare Apple platform serial shape (e.g. C02XY9876543): a value
        # that lost its key must still not survive. Requires letter +
        # two digits + at least eight more alphanumerics, so model and
        # version tokens like H11ANEIn or t6000 never match.
        rx(r"\b[A-Z][0-9]{2}[A-Z0-9]{8,10}\b", "serial", "[redacted]")
        # Network identity. IPv6 before IPv4 so embedded v4-in-v6 is gone.
        rx(r"\b(?:fe80|fd[0-9a-f]{2}|fc[0-9a-f]{2})(?::[0-9a-fA-F]{0,4}){1,7}"
           r"(?:%\w+)?\b", "ipv6", "[redacted-ip6]", re.IGNORECASE)
        rx(r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b",
           "ipv6", "[redacted-ip6]")
        rules.append((
            re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d{1,3})?\b"),
            self._ipv4_sub,
        ))
        # Home paths: this user's home first, then any user's.
        if self.home and self.home != "/" and self.home != "":
            rules.append((
                re.compile(re.escape(self.home)),
                self._replace("home_path", "[home]"),
            ))
        rx(r"(?<![\w.-])/(?:home|Users)/[^/\s:\"'@]+", "home_path", "[home]")
        # Live host and user names, last, so path placeholders above win.
        # A short or model-word hostname is RESTRICTED: it never
        # redacts inside the device-tree/IORegistry name fields, or
        # `Apple MacBook Neo` dies to a host called `neo`.
        if self.hostname and len(self.hostname) >= 2:
            hostname_restricted = _restricted_alias(self.hostname)

            def host_fn(match, field=None, _r=hostname_restricted):
                return self._alias_sub(match, field, _r, kind="hostname")
            host_fn._needs_field = True
            rules.append((
                re.compile(r"(?<![\w.-])" + re.escape(self.hostname) +
                           r"(?![\w.-])", re.IGNORECASE),
                host_fn,
            ))
        # Derived short names survive whole-token redaction inside unit
        # files and paths (`/etc/systemd/system/<alias>-ane.service`,
        # quoted verbatim by the journal). Match them as a substring of
        # a token: the neighbor on each side must be non-alphanumeric,
        # so hex runs (`0xdeadbeef` against a `dead`-ish alias) and
        # longer words are never corrupted. Longest first. Restricted
        # aliases (short, or a marketing-name word) are field-aware:
        # never the name fields.
        for alias in host_aliases(self.hostname):
            alias_restricted = _restricted_alias(alias)

            def alias_fn(match, field=None, _r=alias_restricted):
                return self._alias_sub(match, field, _r)
            alias_fn._needs_field = True
            rules.append((
                re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) +
                           r"(?![A-Za-z0-9])", re.IGNORECASE),
                alias_fn,
            ))
        # The user name is PII inside a hyphenated token too
        # (`/tmp/steve-build`, `build-steve/out`), so `-` is not a boundary
        # for it. The hostname rule above keeps `-` as a boundary on
        # purpose: Omarchy's default hostname is `omarchy`, and every
        # `mlx-omarchy-*` / `omarchy-*` token would otherwise turn into
        # `[host]`.
        if self.username and len(self.username) >= 2:
            rules.append((
                re.compile(r"(?<![\w.])" + re.escape(self.username) +
                           r"(?![\w.])", re.IGNORECASE),
                self._replace("username", "[user]"),
            ))
        return rules

    def apply_value(self, value, field=None):
        """Redact structured observations without changing their types."""
        if isinstance(value, str):
            return self.apply(value, field=field)
        if isinstance(value, dict):
            return {key: self.apply_value(item, field=key)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [self.apply_value(item, field=field) for item in value]
        return value

    def apply(self, text, *, field=None):
        if not isinstance(text, str):
            text = str(text)
        for pattern, repl in self._rules:
            if repl == self._ipv4_sub:
                text = pattern.sub(lambda match: self._ipv4_sub(match, field), text)
            elif getattr(repl, "_needs_field", False):
                text = pattern.sub(lambda match: repl(match, field), text)
            else:
                text = pattern.sub(repl, text)
        return text


def cap_stream(text, max_chars=None):
    """Cap one captured output stream so archives stay small."""
    if text is None:
        return ""
    limit = MAX_STREAM_CHARS if max_chars is None else max_chars
    lines = text.splitlines()
    total = len(lines)
    kept = lines[:MAX_STREAM_LINES]
    out = "\n".join(kept)[:limit]
    if total > MAX_STREAM_LINES or len(text) > limit:
        out += f"\n[truncated: {total} lines, {len(text)} chars captured]"
    return out


def redact_argv(argv, redactor):
    """Record the command shape without values that could carry secrets."""
    return [redactor.apply(str(a)) for a in argv]


def run_tool(argv, redactor, label=None, timeout=30, cwd=None, env=None,
             max_chars=None, redact_output=True):
    """Run one external command; record absence, timeout, and output.

    Returns a dict. Never raises for a missing binary or a timeout.
    """
    record = {
        "label": label or argv[0],
        "argv": redact_argv(argv, redactor),
        "available": True,
        "exit_code": None,
        "error": None,
        "duration_ms": None,
        "stdout": "",
        "stderr": "",
    }
    if shutil.which(argv[0]) is None:
        record["available"] = False
        record["error"] = "not-found"
        return record
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        record["exit_code"] = proc.returncode
        stdout = cap_stream(proc.stdout or "", max_chars=max_chars)
        stderr = cap_stream(proc.stderr or "", max_chars=max_chars)
        record["stdout"] = redactor.apply(stdout) if redact_output else stdout
        record["stderr"] = redactor.apply(stderr) if redact_output else stderr
    except subprocess.TimeoutExpired:
        record["error"] = f"timeout after {timeout}s"
    except OSError as exc:
        record["available"] = False
        record["error"] = f"os-error: {exc}"
    record["duration_ms"] = int((time.monotonic() - started) * 1000)
    return record


def run_python_probe(code, redactor, label, timeout=120, env=None,
                     max_chars=None):
    """Run a python snippet in a child interpreter, bounded."""
    return run_tool(
        [sys.executable, "-c", code],
        redactor,
        label=label,
        timeout=timeout,
        env=env,
        max_chars=max_chars,
    )


def dump_json(obj):
    """Deterministic JSON text: sorted keys, fixed indent, trailing newline."""
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def json_bytes(obj):
    return dump_json(obj).encode("utf-8")


def archive_bytes(files):
    """Deterministic gzip tarball from {member name: bytes}.

    Sorted members, mtime 0, root owner, empty uname/gname, gzip mtime 0.
    Same files in, same bytes out.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.type = tarfile.REGTYPE
            tf.addfile(info, io.BytesIO(data))
    return gzip.compress(buf.getvalue(), compresslevel=9, mtime=0)


def build_manifest(archive_name, files, extra=None):
    """Manifest describing every member except itself.

    `files` must not contain "manifest.json"; the caller adds it to the
    archive after hashing the manifest bytes themselves.
    """
    entries = []
    for name in sorted(files):
        data = files[name]
        entries.append({
            "path": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_version": 1,
        "tool": "mlx-omarchy collect-deep",
        "archive": archive_name,
        "files": entries,
        "no_network": True,
        "upload": "only after explicit --submit or SUBMIT confirmation",
    }
    if extra:
        manifest.update(extra)
    return manifest


def is_native_macos(host, manifest):
    """True when a report came from native macOS MLX rather than Linux."""
    return (host.get("system") or manifest.get("system")) == "Darwin"


def _bounded_pmgr_blocks(blocks, truncated, max_blocks=8, max_children=256):
    """Cap the pmgr offset topology: 4 blocks, 256 children each.

    `children_total` from the collector records the true child count;
    the `truncated` marker says explicitly when the cap clipped (t600x
    pmgr blocks run past 256 children).
    """
    if not isinstance(blocks, list):
        return []
    kept = []
    for block in blocks[:max_blocks]:
        if not isinstance(block, dict):
            continue
        entry = dict(block)
        children = block.get("children")
        if isinstance(children, list) and len(children) > max_children:
            truncated.append(
                f"pmgr_blocks.children:{len(children) - max_children}")
            entry["children"] = children[:max_children]
        kept.append(entry)
    if len(blocks) > max_blocks:
        truncated.append(f"pmgr_blocks:{len(blocks) - max_blocks}")
    return kept


# Payload schema for every `truncated` marker list: at most 16 items,
# each at most 64 characters.
MARKER_MAX_ITEMS = 16
MARKER_MAX_LEN = 64


def bound_markers(markers, cap_marker="truncated:cap"):
    """Fit a `truncated` marker list to the payload schema.

    Each marker is cut to MARKER_MAX_LEN; past MARKER_MAX_ITEMS the
    list stops at the cap and its last slot says so (`cap_marker`).
    """
    items = [str(m)[:MARKER_MAX_LEN] for m in markers or []]
    if len(items) > MARKER_MAX_ITEMS:
        items = items[:MARKER_MAX_ITEMS - 1] + [cap_marker]
    return items


# Payload budget for the whole ane_port_detail block. 64 KiB covered
# the old devicetree-only content; deep rows also carry the 48 KiB
# omarchy_ane promotion block, so the deep path raises the budget
# (payload cap is 256 KiB, shared with ane_macos/ane_linux).
PORT_DETAIL_MAX_BYTES = 64 * 1024
PORT_DETAIL_MAX_BYTES_DEEP = 112 * 1024


def _cap_port_detail(port, redactor, max_bytes=PORT_DETAIL_MAX_BYTES):
    """Bounded `ane_port_detail` blob for the quick PAYLOAD.

    Carries the devicetree (and the runtime block if it fits) at enough
    fidelity to author the omarchy-ane overlay off-machine. The flat
    `ane_port` summary string is kept for back-compat with rows that
    were already stored before the detail object existed.

    Bounding rules, in order — each one marks `truncated` with the
    reason so a reader knows the shape of what was dropped:

      * ane_nodes / darts / phandles: hard cap of MAX_PORT_DETAIL_NODES
        (darts: MAX_PORT_DART_NODES) each; a node beyond the cap is
        dropped, not truncated in place.
      * pmgr_domains: hard cap of 64 (matches the source collector).
      * pmgr_blocks: hard cap of 4 blocks, 256 children each (matches
        the source collector; `children_total` records the true count).
      * total serialized bytes: hard cap of `max_bytes`. Drop order is
        by porting value: the runtime block first, then the phandle map
        (`iommus_resolved` already did that arithmetic for the reader),
        then the AIC block, then the structured boot provenance, then
        the ANE pmgr subset, then the full pmgr offset topology, then
        DARTs; the ane nodes — the whole point of the capture, plus the
        pmgr map a SET-base derivation needs — are protected last.
      * a macOS report carries `macos` instead of `devicetree` and is
        bounded at the source; if it somehow exceeds the budget the
        whole block is dropped.

    `redactor` is the shared one from the collector: every string value
    already passed it once during probe_ane_port; passing it again here
    is a belt-and-braces pass in case a probe missed a string-shaped
    value. Numeric and bool fields stay numeric and bool.
    """
    if not isinstance(port, dict):
        return None

    MAX_NODES = 8  # hard cap on ane_nodes / phandles entries
    MAX_DARTS = 32  # real trees carry up to ~30 DARTs (t600x/t602x)
    truncated = []

    src_devicetree = port.get("devicetree")
    src_runtime = port.get("runtime")
    src_macos = port.get("macos")

    def _walk(node):
        """Re-redact every string leaf; keep numeric/bool/list shape."""
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            return redactor.apply(node) if redactor else node
        return node

    def _size(node):
        return len(json.dumps(node, separators=(",", ":")))

    # macOS: the probe bounds its own output; carry it through and
    # re-redact. The legacy Linux devicetree keys do not apply.
    if src_macos is not None:
        out = {"macos": _walk(src_macos)}
        if isinstance(out["macos"], dict) and \
                isinstance(out["macos"].get("truncated"), list):
            out["macos"]["truncated"] = bound_markers(
                out["macos"]["truncated"])
        if truncated or _size(out) > max_bytes:
            out["truncated"] = (truncated or [])[:15] + \
                ["macos:over_budget"]
            if _size(out) > max_bytes:
                return {"available": False,
                        "error": "macos ane_port_detail over budget",
                        "truncated": out["truncated"]}
        return out

    src_devicetree = src_devicetree or {}
    def _bounded_dict(d, cap, name):
        if not isinstance(d, dict):
            return {}
        keys = sorted(d.keys())
        kept = {k: _walk(d[k]) for k in keys[:cap]}
        if len(keys) > cap:
            truncated.append(f"{name}:{len(keys) - cap}")
        return kept

    bounded = {
        "ane_node_present": bool(src_devicetree.get("ane_node_present")),
        "ane_nodes": _bounded_dict(src_devicetree.get("ane_nodes") or {},
                                   MAX_NODES, "ane_nodes"),
        "ane_reg": _walk(src_devicetree.get("ane_reg"))
            if isinstance(src_devicetree.get("ane_reg"), list) else None,
        "darts": _bounded_dict(src_devicetree.get("darts") or {},
                               MAX_DARTS, "darts"),
        "mailbox": _bounded_dict(src_devicetree.get("mailbox") or {},
                                 MAX_NODES, "mailbox"),
        "reserved_memory": _walk(src_devicetree.get("reserved_memory"))
            if isinstance(src_devicetree.get("reserved_memory"), dict) else {
                "found": False, "child_count": 0, "nodes": {}},
        "pmgr_domains": _walk(src_devicetree.get("pmgr_domains") or [])[:64]
            if len(src_devicetree.get("pmgr_domains") or []) > 64
            else _walk(src_devicetree.get("pmgr_domains") or []),
        "pmgr_blocks": _bounded_pmgr_blocks(
            src_devicetree.get("pmgr_blocks"), truncated),
        "aic": _walk(src_devicetree.get("aic"))
            if src_devicetree.get("aic") else None,
        "set_base_candidate": _walk(src_devicetree.get("set_base_candidate"))
            if isinstance(src_devicetree.get("set_base_candidate"), dict)
            else None,
        "adt": _walk(src_devicetree.get("adt"))
            if isinstance(src_devicetree.get("adt"), dict) else None,
        "adt_nodes": _walk(src_devicetree.get("adt_nodes"))
            if isinstance(src_devicetree.get("adt_nodes"), dict) else None,
        "phandles": _bounded_dict(src_devicetree.get("phandles") or {},
                                  MAX_NODES, "phandles"),
        "boot": _walk(src_devicetree.get("boot"))
            if isinstance(src_devicetree.get("boot"), dict) else None,
        "dtb_sha256": src_devicetree.get("dtb_sha256")
            if re.fullmatch(r"[0-9a-f]{64}",
                            str(src_devicetree.get("dtb_sha256") or ""))
            else None,
        "dtb_sha256_error": (str(src_devicetree.get("dtb_sha256_error"))
                             [:64]
                             if src_devicetree.get("dtb_sha256_error")
                             else None),
    }
    if len(src_devicetree.get("pmgr_domains") or []) > 64:
        truncated.append(f"pmgr_domains:"
                         f"{len(src_devicetree['pmgr_domains']) - 64}")

    out = {"devicetree": bounded}

    if isinstance(src_runtime, dict):
        bounded_runtime = _walk(src_runtime)
        # Total budget check: include runtime only if it fits. We try
        # with runtime first, then drop it if it pushes us over the cap.
        candidate = dict(out)
        candidate["runtime"] = bounded_runtime
        if _size(candidate) <= max_bytes:
            out = candidate
        else:
            truncated.append("runtime:over_budget")

    # Still over budget: drop by porting value. The phandle map goes
    # first (iommus_resolved already did that arithmetic for the
    # reader), then the AIC block, the structured boot provenance, the
    # ANE pmgr subset and the full pmgr topology, then DARTs, the
    # reserved-memory subtree, the mailbox nodes; the ane nodes go
    # last.
    if _size(out) > max_bytes:
        trimmed = out
        for field, blank in (("phandles", {}), ("aic", None),
                             ("boot", None), ("set_base_candidate", None),
                             ("pmgr_domains", []),
                             ("pmgr_blocks", []), ("darts", {}),
                             ("reserved_memory",
                              {"found": False, "child_count": 0,
                               "nodes": {}}),
                             ("mailbox", {}), ("ane_nodes", {})):
            if _size(trimmed) <= max_bytes:
                break
            dt = dict(trimmed["devicetree"])
            dt[field] = blank
            trimmed = {"devicetree": dt}
            if "runtime" in out:
                trimmed["runtime"] = out["runtime"]
            truncated.append(f"{field}:over_budget")
        if _size(trimmed) > max_bytes:
            truncated.append("devicetree:over_budget")
            return {"available": False,
                    "error": "devicetree ane_port_detail over budget",
                    "truncated": truncated[:16]}
        out = trimmed

    if truncated:
        out["truncated"] = truncated[:16]
    return out


def _cap_omarchy_ane(block, redactor, max_bytes=48 * 1024):
    """Bounded `omarchy_ane` turn-on block (Linux deep section).

    Cap order is fixed by contract: when over budget the dmesg tail is
    dropped FIRST; check / module / smoke / dmesg_faults are never
    dropped. Still over budget after that, the list-shaped fields
    (firmware, opt_in) are halved with an explicit truncated marker;
    those never lose the check or module facts.
    """
    if not isinstance(block, dict):
        return None
    out = dict(block)

    def _walk(value):
        if isinstance(value, str):
            return redactor.apply(value) if redactor else value
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value

    out = _walk(out)
    dropped = []
    if _size_json(out) > max_bytes and "dmesg" in out:
        out.pop("dmesg")
        dropped.append("dmesg")
    for field in ("firmware", "opt_in"):
        if _size_json(out) <= max_bytes:
            break
        value = out.get(field)
        if isinstance(value, list) and len(value) > 4:
            out[field] = value[:len(value) // 2]
            dropped.append(f"{field}:half")
    if dropped:
        out["truncated"] = (out.get("truncated") or [])[:8] + dropped
    return out


def _size_json(value):
    return len(json.dumps(value, separators=(",", ":")))


def _cap_ane_block(block, redactor, max_bytes=96 * 1024):
    """Bounded top-level ane_macos / ane_linux payload block.

    Bulk raw-dump fields are dropped first (they also live in the
    archive); compact identity/summary fields are protected. Every drop
    is recorded in `truncated`.
    """
    if not isinstance(block, dict):
        return None

    def _walk(value):
        if isinstance(value, str):
            return redactor.apply(value) if redactor else value
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value

    out = _walk(block)
    truncated = list(out.get("truncated") or [])
    # Drop order: the raw dumps and per-node property payloads first,
    # then the derived row lists; scalars and probe verdicts survive.
    for field in ("raw_text", "dt", "dmesg", "interrupts", "overlays",
                  "dtb_copies", "firmware", "tunables", "nodes",
                  "generation_probe", "classes"):
        if _size_json(out) <= max_bytes:
            break
        if field in out:
            out.pop(field)
            truncated.append(f"{field}:over_budget")
    if truncated:
        out["truncated"] = truncated[:16]
    return out


def build_payload(kind, quick, manifest, generated_at=None, benchmark=None,
                  redactor=None, ane_macos=None, ane_linux=None):
    """Build the strict-schema JSON summary sent with the upload.

    Must match schema/payload-v1.schema.json in services/community-data:
    fixed key set, schema_version pinned (v2 since 2026-10; the service
    accepts v1 and v2), every identity field nullable. `ane_macos` /
    `ane_linux` carry the deep collector's ANE turn-on blocks (null on
    a quick run). All values come from already-redacted data.
    `redactor` is the collector's shared Redactor; it is used to
    belt-and-braces re-redact the ane_port_detail blob in case a probe
    missed a string-shaped value. New callers should pass it; existing
    tests that do not are tolerated (re-redaction becomes a no-op).
    """
    host = quick.get("host") or {}
    dt = host.get("devicetree") or {}
    gpu = (quick.get("mesa") or {}).get("gpu") or {}
    mlx = quick.get("mlx") or {}
    distributions = mlx.get("distributions") or {}
    compatible = dt.get("compatible") or []
    # Group by SoC, not by board: an M1 MacBook Pro reports
    # ["apple,j293", "apple,t8103", "apple,arm-platform"], and only
    # apple,t8103 identifies the chip that every other M1 machine shares.
    soc = next((c for c in compatible
                if re.match(r"^apple,t\d{4}", c)), None)
    host_cpu = host.get("cpu_online")
    cpu = host.get("cpu") or {}
    boot = host.get("boot") or {}
    cmdline = host.get("cmdline")
    ane_dt = (quick.get("ane") or {}).get("devicetree") or {}
    ane_compatible = ane_dt.get("compatible")
    ane_compat_blob = " ".join(str(t) for t in ane_compatible) \
        if isinstance(ane_compatible, list) else None
    boot_chain = " ".join(
        f"{key}={boot[key]}" for key in sorted(boot)
        if isinstance(boot.get(key), str) and boot[key])
    present = cpu.get("present")
    shortfall = host.get("core_shortfall")
    # Plain wire fact: true when the report recorded an unexplained
    # shortfall, false when present/online are known and equal enough,
    # null when the counts needed to judge are missing.
    if isinstance(shortfall, dict):
        shortfall_flag = True
    elif isinstance(present, int) and isinstance(host_cpu, int):
        shortfall_flag = False
    else:
        shortfall_flag = None
    # The benchmark numbers must ride in the summary, not only inside the
    # archive: the read API serves summaries, so cross-machine comparison
    # is impossible unless the numbers travel with them.
    rows = []
    for row in (benchmark or [])[:16]:
        if not isinstance(row, dict) or not isinstance(row.get("n"), int):
            continue
        rows.append({
            "n": row["n"],
            "tflops": row.get("tflops"),
            "median_ms": row.get("median_ms"),
        })
    native = is_native_macos(host, manifest)
    # Bounded driver-port summary: enough for fleet queries (does this
    # SoC expose the ane node, how many DARTs and PMGR domains, which
    # AIC) without shipping the full devicetree dump in every payload.
    src_port = quick.get("ane_port") or {}
    macos_port = src_port.get("macos")
    if isinstance(macos_port, dict):
        cores = sorted({i.get("cores") for i in
                        (macos_port.get("instances") or [])
                        if isinstance(i.get("cores"), int)})
        ane_port = (
            "native_macos=1 instances=%d cores=%s dart_ane=%d "
            "firmware=%s" % (
                len(macos_port.get("instances") or []),
                "+".join(str(c) for c in cores) or "?",
                len(macos_port.get("dart_nodes") or []),
                "loaded" if any(i.get("firmware_loaded")
                                for i in macos_port.get("instances") or [])
                else "unknown"))[:1024]
    else:
        port = src_port.get("devicetree") or {}
        port_parts = [
            "present=" + str(bool(port.get("ane_node_present"))).lower()]
        for name, props in sorted((port.get("ane_nodes") or {}).items()):
            regs = props.get("reg") if isinstance(props, dict) else None
            port_parts.append(f"{name}={regs[0] if regs else 'no-reg'}")
        port_parts.append(f"darts={len(port.get('darts') or {})}")
        port_parts.append(
            f"pmgr_domains={len(port.get('pmgr_domains') or [])}")
        aic_compat = (port.get("aic") or {}).get("compatible") or []
        if aic_compat:
            port_parts.append(f"aic={aic_compat[0]}")
        ane_port = (" ".join(port_parts)[:1024]
                    if quick.get("ane_port") else None)
    # Bounded full-structure port detail. Capped per-section, total
    # bytes hard-bounded; truncation is recorded explicitly so a reader
    # can tell which corner the cap clipped. Re-redacts every string
    # leaf against the shared Redactor in case a probe missed one. A
    # failed probe or an over-budget detail is an explicit error
    # object in the payload, never a silent omission.
    ane_port_detail = None
    if quick.get("ane_port"):
        if src_port.get("available") is False:
            ane_port_detail = {
                "available": False,
                "error": str(src_port.get("error")
                             or "ane_port probe failed")[:256],
            }
        else:
            ane_port_detail = _cap_port_detail(
                src_port, redactor, max_bytes=(
                    PORT_DETAIL_MAX_BYTES_DEEP if kind == "deep"
                    else PORT_DETAIL_MAX_BYTES))
    kernel = host.get("kernel_release")
    if native:
        shortfall_flag = None
        kernel = f"Darwin {kernel or 'unknown'} ({host.get('os') or 'macOS'})"
    device = mlx.get("default_device")
    if native and mlx.get("metal_available"):
        device = f"Metal GPU (native macOS MLX, {device})"
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "generated_at": generated_at or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arch": host.get("arch"),
        "model": host.get("model") if native else dt.get("model"),
        "chip": host.get("chip") if native else
            soc or (compatible[0] if compatible else None),
        "kernel": kernel,
        "mesa_driver": gpu.get("driverName"),
        "mesa_device": gpu.get("deviceName"),
        "mlx_version": distributions.get("mlx-omarchy")
            or mlx.get("mlx_version"),
        "mlx_device": device,
        "source_commit": manifest.get("source_commit"),
        "repo_dirty": manifest.get("repo_dirty"),
        "cpu_present": present if isinstance(present, int) else None,
        "hotplug_control": cpu.get("hotplug_control")
            if isinstance(cpu.get("hotplug_control"), bool) else None,
        "ane_dt_node": ane_dt.get("node")
            if isinstance(ane_dt.get("node"), bool) else None,
        "ane_port": ane_port or None,
        "ane_port_detail": ane_port_detail,
        "ane_dt_compatible": ane_compat_blob[:512] if ane_compat_blob
        else None,
        "boot_chain": boot_chain[:512] or None,
        "cmdline": cmdline[:1024] if isinstance(cmdline, str) else None,
        "core_shortfall": shortfall_flag,
        "cpu_online": host_cpu if isinstance(host_cpu, int) else None,
        "benchmark": rows,
        "ane_macos": _cap_ane_block(ane_macos, redactor),
        "ane_linux": _cap_ane_block(ane_linux, redactor),
        "redaction_summary": dict(
            manifest.get("redaction_summary") or {}),
        "files": [
            {key: entry[key] for key in ("path", "bytes", "sha256")}
            for entry in manifest.get("files", [])
        ],
    }


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None
