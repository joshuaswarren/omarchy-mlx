#!/usr/bin/env python3
"""Native macOS facts for the community collectors. No MLX install required."""

import json
import platform

import bench_matrix
from collect_common import (run_tool, run_python_probe,
                            PROBE_STREAM_CHARS, bound_markers)


def not_applicable():
    return {"available": False, "error": "not applicable to native macOS MLX"}


# Runs via run_python_probe on the Mac. Read-only: ioreg queries and one
# best-effort powermetrics sample. Structured so every string that
# reaches the report has passed the Redactor back in the parent.
# The helpers below are shared with ANE_MACOS_DUMP_PROBE.
ANE_PROBE_HELPERS = r"""
import json, os, plistlib, re, subprocess

out = {"available": False, "instances": [], "ane_nodes": [],
       "dart_nodes": [], "mailbox_nodes": [], "dt_nodes": [],
       "pmgr_nodes": [], "interrupt_controllers": [],
       "coreml": {"available": False, "compute_units": None, "error": None},
       "powermetrics": {"available": False, "power_mw": None,
                        "error": None},
       "driver": None, "platform": None, "compiler": None,
       "set_base_candidate": None,
       "nodes": [], "arm_io": None, "aic": None, "classes": [],
       "kexts": [], "firmware": {"matched": 0, "files": []},
       "raw_text": None,
       "truncated": [], "stripped": []}

# Byte-order contract (oracle: t6021-test-host T6021 capture 2026-09-17,
# ane-linux-experiments receipts/2026-09-17-t6021-test-host-t6021-macos-capture/):
# ioreg OSData properties present multi-byte cells in HOST byte order
# (little-endian on arm64). AAPL,phandle <69010000> is 0x169; reg is
# u64-LE (address, size) pairs whose sizes match IODeviceMemory exactly.
# v0.6.4 read phandle big-endian and shipped swapped values in every
# macOS row (5 published rows carry garbage such as 0x69010000).

MAX_REG_BYTES = 4096   # v0.6.5: 64 could not carry a real pmgr reg
                       # (t6002 pmgr reg is 3856 bytes / 241 ranges)
MAX_RANGES = 256       # t602x pmgr reg = 73 ranges; t6002 = 241


def _raw_text(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bytes):
        value = value.split(b"\x00")[0].decode("utf-8", "replace")
    return str(value) if value is not None else None


def _text(value):
    value = _raw_text(value)
    return value[:128] if value is not None else None


# Payload schema: every `truncated` item is at most 64 characters.
MAX_MARKER = 64


def _marker(head, name, tail=""):
    # "<head>:<name><tail>" within MAX_MARKER; only the node name is cut.
    room = max(0, MAX_MARKER - len(head) - len(tail) - 1)
    return "%s:%s%s" % (head, (name or "?")[:room], tail)


def _ascii(value):
    # ASCII text out of an OSData blob (target-type, platform-name).
    if isinstance(value, bytes):
        value = value.split(b"\x00")[0]
        return value.decode("ascii", "replace") if value else None
    if isinstance(value, str):
        return value or None
    return None


def _run(argv, timeout=30):
    return subprocess.run(argv, capture_output=True, timeout=timeout)


def _plist_argv_output(argv, timeout=60):
    proc = _run(argv, timeout=timeout)
    if proc.returncode != 0:
        return None
    return plistlib.loads(proc.stdout)


def _json_argv_output(argv, timeout=30):
    # plutil -convert json emits JSON, not a plist.
    proc = _run(argv, timeout=timeout)
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout.decode("utf-8", "replace"))


def _compatible(node):
    raw = node.get("compatible")
    if isinstance(raw, bytes):
        texts = [t.decode("utf-8", "replace") for t in raw.split(b"\x00")
                 if t]
        return texts[:8]
    if isinstance(raw, list):
        texts = [_ascii(t) for t in raw if _ascii(t)]
        return texts[:8] or None
    return None


def _reg(node):
    raw = node.get("reg") or node.get("IODeviceMemory")
    if isinstance(raw, bytes):
        if len(raw) > MAX_REG_BYTES:
            out["truncated"].append(_marker("reg_bytes",
                                            _text(node.get("name"))))
            raw = raw[:MAX_REG_BYTES]
        return raw.hex()
    return None


def _reg_ranges(node):
    # Decoded reg ranges as ["0xADDR/0xSIZE", ...], host-endian cells.
    # u64 (address, size) pairs are the norm (Apple ARM reg); u32 pairs
    # cover 4/8-byte stragglers (mapper-ane0 reg is <00000000>).
    # IODeviceMemory (plist dicts) is the fallback when the raw reg
    # OSData is absent; its values are already host-native ints.
    name = _text(node.get("name"))
    raw = node.get("reg")
    if isinstance(raw, bytes):
        cell = 8 if len(raw) % 16 == 0 else 4
        if len(raw) % (2 * cell):
            out["truncated"].append(_marker("reg_ranges", name))
            return None
        vals = [int.from_bytes(raw[i:i + cell], "little")
                for i in range(0, len(raw), cell)]
        pairs = list(zip(vals[0::2], vals[1::2]))
    elif isinstance(node.get("IODeviceMemory"), list):
        pairs = []
        for entry in node["IODeviceMemory"]:
            if isinstance(entry, list) and entry and \
                    isinstance(entry[0], dict):
                pairs.append((int(entry[0].get("address") or 0),
                              int(entry[0].get("length") or 0)))
    else:
        return None
    total = len(pairs)
    ranges = ["0x%x/0x%x" % (base, size) for base, size in pairs]
    if len(ranges) > MAX_RANGES:
        out["truncated"].append(
            _marker("reg_ranges", name, ":%d" % total))
        ranges = ranges[:MAX_RANGES]
    return ranges or None


def _keep(node):
    # Keys the payload schema whitelists; AAPL,phandle is carried as
    # `phandle` below.
    keys = ("name", "compatible", "reg", "IODeviceMemory",
            "IOInterruptControllers", "IOInterruptSpecifiers", "IOClass")
    return {k: node[k] for k in keys
            if node.get(k) is not None and k != "IODeviceMemory"}


# --- shared: generic ANE-pattern match and privacy strip list ------------
# Node discovery is pattern-based so an untested generation (M3 ascwrap
# IOPs, M4 t8020-class) is captured without code changes: anything whose
# name mentions ane (ane, ane0, dart-ane0, iop-ane0, ane-ascwrap,
# mailbox-ane), or carries the ascwrap / t8020 markers, matches.
def _is_ane_name(name):
    n = (name or "").lower()
    return bool(n) and ("ane" in n or "ascwrap" in n or "t8020" in n)


_STRIP_EXACT = frozenset(k.lower() for k in (
    "serial-number", "unique-chip-id", "unique-chip", "ecid", "mlb",
    "mac-address", "local-mac-address", "device-uuid", "IOPlatformUUID",
    "boot-uuid"))
_STRIP_PREFIX = ("wifi-", "bluetooth-", "fv-")
_STRIP_SUFFIX = "-hash"


def _is_stripped_key(key):
    k = key.lower()
    return (k in _STRIP_EXACT
            or k.startswith(_STRIP_PREFIX) or k.endswith(_STRIP_SUFFIX)
            # Serial / unique-id family beyond the exact list:
            # mlb-serial-number, IOPlatformSerialNumber, board-serial,
            # *-udid, *-uuid. Removed whole, never value-blanked.
            or "serial" in k or "udid" in k or "uuid" in k)


def _strip_props(props, stripped):
    # Drop identity keys; record their NAMES (never values).
    out = {}
    for key, value in props.items():
        if _is_stripped_key(key):
            stripped.append(str(key)[:64])
            continue
        out[key] = value
    return out


def _spec_hex_list(value):
    # IOInterruptSpecifiers as hex, one entry per specifier (canonical
    # form: ["0x74030000"]). Non-bytes entries survive as text; the
    # overlay generator still reads the three legacy encodings that older
    # published rows carry (python bytes repr, latin-1 text, U+FFFD-lossy).
    out = []
    for spec in (value if isinstance(value, list) else [value]):
        if isinstance(spec, bytes):
            out.append("0x" + spec.hex())
        elif spec is None:
            continue
        else:
            out.append(str(spec)[:64])
    return out or None


# IODeviceTree-plane whitelist: exactly the keys a reader (and the
# omarchy-ane promotion tooling) consumes. A whitelist, not a blacklist:
# nothing outside this set can reach the summary macOS block.
_DT_KEYS = ("name", "compatible", "reg", "interrupts", "interrupt-names",
            "IOInterruptSpecifiers", "segment-ranges", "ane-type",
            "ane-subtype", "ane-id", "die-id", "die-ane-id", "clock-gates",
            "power-gates", "iommu-parent", "vm-base", "vm-size",
            "page-size", "sids", "bypass-15", "instance",
            "dapf-instance-0", "dart-id", "dart-options", "role",
            "device_type", "ranges", "#address-cells", "#size-cells")
MAX_DT_HEX = 8192
# Keys the payload schema types as string arrays (dt_nodes items in
# services/community-data/schema/payload-v1.schema.json). As OSData they
# arrive NUL-separated (M5 sends `compatible` this way), so split them
# into strings instead of hex-encoding them.
_DT_STRING_LISTS = ("compatible", "interrupt-names")
# Schema maxLength per item of those string arrays.
_DT_ITEM_LEN = {"compatible": 256, "interrupt-names": 128}
# Schema maxLength for string-typed dt_nodes keys tighter than the hex
# cap; anything longer is cut on this side and recorded in `truncated`.
_DT_MAX_LEN = {"name": 128, "ane-type": 256, "ane-subtype": 256,
               "ane-id": 64, "die-id": 64, "die-ane-id": 64,
               "clock-gates": 4096, "power-gates": 4096,
               "iommu-parent": 256, "vm-base": 64, "vm-size": 64,
               "page-size": 64, "sids": 4096, "bypass-15": 256,
               "instance": 64, "dapf-instance-0": 4096, "dart-id": 64,
               "dart-options": 256, "role": 128, "device_type": 128,
               "ranges": 4096}


def _full_value(value, truncated, depth=0):
    # Encode any plist property for the full dump: hex for blobs,
    # capped strings/lists/dicts, scalars kept.
    if depth > 6:
        truncated.append("value:depth")
        return None
    if isinstance(value, bytes):
        hexs = value.hex()
        if len(hexs) > 16384:
            truncated.append("hex:cap")
            hexs = hexs[:16384]
        return {"hex": hexs, "bytes": len(value)}
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value[:1024]
    if isinstance(value, list):
        items = [_full_value(v, truncated, depth + 1) for v in value[:64]]
        if len(value) > 64:
            truncated.append("list:cap")
        return items
    if isinstance(value, dict):
        out = {str(k)[:128]: _full_value(v, truncated, depth + 1)
               for k, v in list(value.items())[:64]}
        if len(value) > 64:
            truncated.append("dict:cap")
        return out
    return _text(value)


def _dt_strings(texts, total, key, name, truncated):
    # A schema string array: at most 16 items, each cut to the schema's
    # per-item maxLength; both cuts are recorded, never silent.
    if total > 16:
        truncated.append(_marker("list:" + key, name))
    cap = _DT_ITEM_LEN[key]
    kept = [t for t in texts[:16] if t]
    if any(len(t) > cap for t in kept):
        truncated.append(_marker("len:" + key, name))
    return [t[:cap] for t in kept] or None


def _dt_entry(node, path, truncated):
    # One whitelisted IODeviceTree node; binary values as hex.
    name = _text(node.get("name"))
    entry = {"path": path[:256], "name": name}
    for key in _DT_KEYS:
        value = node.get(key)
        if value is None:
            continue
        out_key = "phandle" if key == "AAPL,phandle" else key
        if isinstance(value, bytes) and key in _DT_STRING_LISTS:
            parts = [p.decode("utf-8", "replace")
                     for p in value.split(b"\x00") if p]
            entry[out_key] = _dt_strings(parts, len(parts), key, name,
                                         truncated)
        elif isinstance(value, list) and key in _DT_STRING_LISTS:
            parts = [_raw_text(v) for v in value[:16]]
            entry[out_key] = _dt_strings([t for t in parts if t],
                                         len(value), key, name, truncated)
        elif isinstance(value, bytes):
            if len(value) > MAX_DT_HEX:
                truncated.append(_marker("hex:" + key, name))
                value = value[:MAX_DT_HEX]
            entry[out_key] = value.hex()
        elif isinstance(value, list):
            if key == "IOInterruptSpecifiers":
                entry[out_key] = _spec_hex_list(value)
            else:
                texts = [_text(v) for v in value[:16]]
                if len(value) > 16:
                    truncated.append(_marker("list:" + key, name))
                kept = [t for t in texts if t is not None]
                entry[out_key] = kept or None
        else:
            entry[out_key] = _text(value)
        cap = _DT_MAX_LEN.get(out_key)
        if (cap is not None and isinstance(entry[out_key], str)
                and len(entry[out_key]) > cap):
            truncated.append(_marker("len:" + key, name))
            entry[out_key] = entry[out_key][:cap]
    return entry


"""

ANE_PROBE_CODE = ANE_PROBE_HELPERS + r"""
# --- ANE driver instances and candidate driver classes -------------------
try:
    raw = _plist_argv_output(
        ["ioreg", "-a", "-rc", "H11ANEIn", "-l"], timeout=30)
    for node in (raw or []):
        dp = node.get("DeviceProperties") or {}
        out["instances"].append({
            "name": _text(node.get("IONameMatched")),
            "matched": _text(node.get("IONameMatched")),
            "firmware_loaded": node.get("FirmwareLoaded") is True,
            "cores": dp.get("ANEDevicePropertyNumANECores"),
            "version": dp.get("ANEDevicePropertyANEVersion"),
            "minor_version": dp.get("ANEDevicePropertyANEMinorVersion"),
            "hw_board_type": dp.get("ANEDevicePropertyANEHWBoardType"),
            "arch": _text(dp.get(
                "ANEDevicePropertyTypeANEArchitectureTypeStr")),
        })
    out["available"] = bool(out["instances"])
    out["instances"] = out["instances"][:8]
except Exception as exc:
    out["truncated"].append("ioreg_instances:%s" % type(exc).__name__)

# --- Driver identity ------------------------------------------------------
try:
    driver = {}
    first = out["instances"][0] if out["instances"] else {}
    counts = {}
    for klass in ("H11ANEIn", "AppleH13ANEInterface",
                  "AppleH16ANEInterface"):
        try:
            # Zero matches: ioreg -a exits 0 and prints NOTHING (an empty
            # buffer, not a plist) - that is the load-bearing negative.
            proc = _run(["ioreg", "-a", "-rc", klass, "-l"], timeout=30)
            if proc.returncode != 0 or not proc.stdout.strip():
                counts[klass] = 0
            else:
                counts[klass] = len(plistlib.loads(proc.stdout))
        except Exception:
            counts[klass] = None
    # The negative result is load-bearing: a future generation must be
    # able to see that H13/H16 drivers were ABSENT, not merely unrecorded
    # (t6021 is an H11-driver / h14g-generation part).
    driver["classes_empty"] = [k for k in ("AppleH13ANEInterface",
                                           "AppleH16ANEInterface")
                               if counts.get(k) == 0]
    driver["matched_class"] = None
    driver["bundle_identifier"] = None
    try:
        raw = _plist_argv_output(
            ["ioreg", "-a", "-rc", "H11ANEIn", "-l"], timeout=30) or []
        if raw:
            driver["matched_class"] = _text(raw[0].get("IOClass")) \
                or "H11ANEIn"
            driver["bundle_identifier"] = _text(
                raw[0].get("CFBundleIdentifier"))
    except Exception:
        pass
    try:
        kext = _json_argv_output(
            ["plutil", "-convert", "json", "-o", "-",
             "/System/Library/Extensions/AppleH11ANEInterface.kext/"
             "Contents/Info.plist"]) or {}
        driver["kext_version"] = _text(kext.get("CFBundleVersion"))
    except Exception as exc:
        driver["kext_version"] = None
        out["truncated"].append("driver_kext:%s" % type(exc).__name__)
    driver.update({
        "matched_compatible": first.get("matched"),
        "arch": first.get("arch"),
        "cores": first.get("cores"),
        "ane_version": first.get("version"),
        "ane_minor_version": first.get("minor_version"),
        "firmware_loaded": first.get("firmware_loaded"),
        "instance_count": len(out["instances"]),
    })
    out["driver"] = driver
except Exception as exc:
    out["truncated"].append("driver:%s" % type(exc).__name__)

# --- Platform identity (root platform-expert device) ---------------------
try:
    nodes = _plist_argv_output(
        ["ioreg", "-a", "-rc", "IOPlatformExpertDevice", "-l"],
        timeout=30) or []
    root = nodes[0] if nodes else {}
    out["platform"] = {
        "target_type": _ascii(root.get("target-type")),
        # platform-name is the raw SoC id ("t6021"); decoding it here
        # means no human ever hand-decodes the hex blob again.
        "soc_id": _ascii(root.get("platform-name")),
        "compatible": _compatible(root),
        "model": _text(root.get("model")),
    }
except Exception as exc:
    out["truncated"].append("platform:%s" % type(exc).__name__)

# --- ANE compiler provenance (disk facts; binaries are cache-resident) ---
try:
    compiler = {"daemons": [], "error": None}
    fw = "/System/Library/PrivateFrameworks/ANECompiler.framework"
    try:
        blob = _json_argv_output(
            ["plutil", "-convert", "json", "-o", "-",
             fw + "/Resources/Info.plist"]) or {}
        compiler["framework_version"] = _text(
            blob.get("CFBundleShortVersionString"))
    except Exception:
        compiler["framework_version"] = None
    try:
        blob = _json_argv_output(
            ["plutil", "-convert", "json", "-o", "-",
             "/System/Library/Extensions/AppleH11ANEInterface.kext/"
             "Contents/Info.plist"]) or {}
        compiler["kext_version"] = _text(blob.get("CFBundleVersion"))
    except Exception:
        compiler["kext_version"] = None
    for daemon in ("/usr/libexec/aned", "/usr/libexec/aneuserd"):
        if _run(["test", "-f", daemon]).returncode == 0:
            compiler["daemons"].append(daemon)
    svc = "/System/Library/PrivateFrameworks/ANECompilerService.framework"
    compiler["compiler_service_present"] = \
        _run(["test", "-d", svc]).returncode == 0
    # Framework bundles ship plist-only on this generation; the binary
    # lives in the dyld shared cache. A known condition, not a failure.
    compiler["binaries_cache_resident"] = \
        _run(["test", "-f", fw + "/Versions/A/ANECompiler"]
             ).returncode != 0
    compiler["daemons"] = compiler["daemons"][:8]
    out["compiler"] = compiler
except Exception as exc:
    out["truncated"].append("compiler:%s" % type(exc).__name__)

try:
    tree = _plist_argv_output(["ioreg", "-a", "-p", "IOService", "-l"],
                              timeout=120)
    if tree is None:
        raise RuntimeError("ioreg exit nonzero")
    ane_re = re.compile(r"^(ane\d*|dart-ane\d*|mapper-ane\d*)$")
    stack = list(tree) if isinstance(tree, list) else [tree]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        name = _text(node.get("name"))
        if name and ane_re.match(name):
            entry = _keep(node)
            entry["name"] = name
            entry["compatible"] = _compatible(node)
            entry["reg"] = _reg(node)
            entry["reg_ranges"] = _reg_ranges(node)
            for key in ("IOInterruptControllers", "IOClass"):
                if key in entry:
                    entry[key] = _text(entry[key])
            if "IOInterruptSpecifiers" in entry:
                entry["IOInterruptSpecifiers"] = _spec_hex_list(
                    entry["IOInterruptSpecifiers"])
            ph = node.get("AAPL,phandle")
            if isinstance(ph, bytes) and len(ph) == 4:
                # Host byte order: <69010000> is 0x169, not 0x69010000.
                ph = int.from_bytes(ph, "little")
            entry["phandle"] = ph
            if name.startswith(("dart-", "mapper-")):
                dart_id = node.get("dart-id")
                entry["dart_id"] = int.from_bytes(dart_id, "little") \
                    if isinstance(dart_id, bytes) and len(dart_id) == 4 \
                    else None
                # #iommu-cells is a devicetree-only property; IORegistry
                # never surfaces it. null here is a platform miss, not a
                # dropped value (marker added below).
                entry["iommu_cells"] = None
                # dart-options is a dart-internal ID (t6021: 0x25-class),
                # NOT a stream map; recorded for completeness.
                entry["dart_options"] = _text(node.get("dart-options"))
                out["dart_nodes"].append(entry)
            else:
                # Engine sub-window hunt: record IORegistry children
                # (name + reg) when the registry exposes any. t6021
                # mining showed macOS exposes NO ane0 engine sub-window;
                # an empty/null capture is a platform fact.
                kids = node.get("IORegistryEntryChildren")
                if isinstance(kids, list):
                    entry["children"] = [
                        {"name": _text(ch.get("name")), "reg": _reg(ch)}
                        for ch in kids[:16] if isinstance(ch, dict)]
                    if len(kids) > 16:
                        out["truncated"].append("ane_children:cap")
                out["ane_nodes"].append(entry)
        elif name == "pmgr":
            # The IORegistry exposes the pmgr BLOCK base (reg first range
            # / IORegistryEntryLocation) but not the power-controller
            # children, so macOS can cross-check a derived SET base's
            # block but never supply the pwrstate offset itself.
            out["pmgr_nodes"].append({
                "name": name,
                "location": _text(node.get("IORegistryEntryLocation")),
                "reg": _reg(node),
                "reg_ranges": _reg_ranges(node),
            })
        elif name and name.startswith("aic"):
            # Interrupt-controller table row: device IOInterruptSpecifiers
            # decode against this phandle. IORegistry rarely surfaces
            # #interrupt-cells; null is a platform miss, not a failure.
            ph = node.get("AAPL,phandle")
            if isinstance(ph, bytes) and len(ph) == 4:
                ph = int.from_bytes(ph, "little")
            cells = node.get("#interrupt-cells")
            out["interrupt_controllers"].append({
                "name": name,
                "compatible": _compatible(node),
                "phandle": ph,
                "interrupt_cells": cells if isinstance(cells, int) else None,
            })
        elif name and "ane" in name.lower() and "mailbox" in name.lower():
            # The ANE mailbox (send-empty line pick + reg window).
            entry = _keep(node)
            entry["name"] = name
            entry["compatible"] = _compatible(node)
            entry["reg"] = _reg(node)
            entry["reg_ranges"] = _reg_ranges(node)
            if "IOInterruptControllers" in entry:
                entry["IOInterruptControllers"] = _text(
                    entry["IOInterruptControllers"])
            if "IOInterruptSpecifiers" in entry:
                entry["IOInterruptSpecifiers"] = _spec_hex_list(
                    entry["IOInterruptSpecifiers"])
            ph = node.get("AAPL,phandle")
            if isinstance(ph, bytes) and len(ph) == 4:
                ph = int.from_bytes(ph, "little")
            entry["phandle"] = ph
            out["mailbox_nodes"].append(entry)
        children = node.get("IORegistryEntryChildren")
        if isinstance(children, list):
            stack.extend(children)
    out["ane_nodes"] = out["ane_nodes"][:8]
    out["dart_nodes"] = out["dart_nodes"][:8]
    out["mailbox_nodes"] = out["mailbox_nodes"][:8]
    out["pmgr_nodes"] = out["pmgr_nodes"][:8]
    out["interrupt_controllers"] = out["interrupt_controllers"][:8]
    # reg_ranges_total records the true range count even when the cap
    # dropped ranges (t602x pmgr reg: 73 ranges / 1168 bytes); null
    # means unknown (undecodable cells or capped list).
    for pmgr in out["pmgr_nodes"]:
        ranges = pmgr.get("reg_ranges")
        pmgr["reg_ranges_total"] = len(ranges) \
            if ranges and len(ranges) < MAX_RANGES else None
    if len(out["ane_nodes"]) == 8 or len(out["dart_nodes"]) == 8 \
            or len(out["pmgr_nodes"]) == 8:
        out["truncated"].append("devicetree:node_cap")
    if out["dart_nodes"]:
        out["truncated"].append("iommu_cells:unavailable_on_macos")
except Exception as exc:
    out["truncated"].append("ioreg_tree:%s" % type(exc).__name__)

# --- IODeviceTree plane (the ADT-shaped nodes) ---------------------------
# The driver windows (segment-ranges, ane-type, die-id, vm-base, sids,
# tunable inputs) live on this plane, not on IOService. Whitelist keys
# only; the generic _is_ane_name match covers M3 (iop-ane/ascwrap),
# M4 (ane,t8020) and later generations without code changes.
try:
    iodt = _plist_argv_output(
        ["ioreg", "-a", "-p", "IODeviceTree", "-l"], timeout=120)
    if iodt is None:
        raise RuntimeError("ioreg IODeviceTree exit nonzero")
    stack = [(list(iodt) if isinstance(iodt, list) else [iodt], "")]
    while stack:
        nodes, parent = stack.pop()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            name = _text(node.get("name"))
            kids = node.get("IORegistryEntryChildren")
            path = parent + "/" + (name or "?")
            if isinstance(kids, list):
                stack.append((kids, path))
            if _is_ane_name(name):
                out["dt_nodes"].append(_dt_entry(node, path,
                                                 out["truncated"]))
    out["dt_nodes"] = out["dt_nodes"][:16]
    if len(out["dt_nodes"]) == 16:
        out["truncated"].append("dt_nodes:cap")
except Exception as exc:
    out["truncated"].append("iodt_tree:%s" % type(exc).__name__)

# --- SET-base candidate (the hypothesis label is the point) --------------
# On T8103/T6001 the ANE SET region is the ane_* pwrstate cluster at
# pmgr block base + 0xc000, and that sum equals the m1n1 ps_map
# constant. macOS cannot enumerate pwrstate offsets, but when Apple's
# own ane0 device is GRANTED a reg window at exactly base+0xc000 (as on
# t6021: ane0 reg range 3 = 0x8e08c000), that corroborates the
# hypothesis from Apple's own address assignment. A candidate, never a
# derived fact; the Linux/m1n1 side enumerates the cluster itself.
try:
    pmgr_base = None
    for pmgr in out["pmgr_nodes"]:
        ranges = pmgr.get("reg_ranges") or []
        if ranges:
            pmgr_base = int(ranges[0].split("/")[0], 16)
            break
    if pmgr_base is not None:
        candidate = pmgr_base + 0xc000
        confirms = False
        kinds = []
        for ane in out["ane_nodes"]:
            ane_kinds = []
            for rng in ane.get("reg_ranges") or []:
                base_s, size_s = rng.split("/")
                base, size = int(base_s, 16), int(size_s, 16)
                if base == candidate and size > 0:
                    confirms = True
                if base == candidate:
                    ane_kinds.append("pmgr_plus_c000")
                elif base == pmgr_base:
                    ane_kinds.append("pmgr_block")
                else:
                    ane_kinds.append("ane_mmio")
            if ane.get("reg_ranges"):
                ane["range_kinds"] = ane_kinds
        out["set_base_candidate"] = {
            "pmgr_block": "0x%x" % pmgr_base,
            "offset": "0xc000",
            "base": "0x%x" % candidate,
            "driver_window_confirms": confirms,
        }
except Exception as exc:
    out["truncated"].append("set_base:%s" % type(exc).__name__)

# CoreML compute-unit availability when pyobjc is present; otherwise a
# recorded miss, never a crash.
try:
    import CoreML  # type: ignore
    from CoreML import MLComputeUnits  # type: ignore
    out["coreml"] = {"available": True,
                     "compute_units": str(MLComputeUnits.all)[:64],
                     "error": None}
except Exception as exc:
    out["coreml"] = {"available": False, "compute_units": None,
                     "error": type(exc).__name__[:64]}

# ANE power/utilization needs root; record the miss cleanly without it.
try:
    proc = subprocess.run(
        ["powermetrics", "--samplers", "ane_power", "-n", "1", "-i", "100"],
        capture_output=True, timeout=30, text=True)
    if proc.returncode != 0:
        out["powermetrics"]["error"] = proc.stderr.strip()[:128] \
            or "exit %d" % proc.returncode
    else:
        watts = [float(m) for m in
                 re.findall(r"ANE Power: ([0-9.]+) mW", proc.stdout)]
        out["powermetrics"] = {"available": True,
                               "power_mw": watts[0] if watts else None,
                               "error": None}
except Exception as exc:
    out["powermetrics"]["error"] = type(exc).__name__[:64]

def _plain(value):
    if isinstance(value, bytes):
        return _text(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


print(json.dumps(_plain(out))[:200000])
"""

ANE_MACOS_DUMP_PROBE = ANE_PROBE_HELPERS + r"""
# --- Full-property IODeviceTree dump (deep collector only) ---------------
# Generic M3-M6 discovery: /arm-io nodes (children one level down)
# whose name / compatible / device_type / role matches the ANE family
# (ane, iop-ane, ascwrap, exclave, sk-, sio-ane, dart-ane, ane-dart,
# t8020), plus pmgr / aic / product. Root and /chosen are NEVER dumped.
# ALL properties, strip list applied, hex for blobs, raw plist text for
# the archive member. A pattern with no match is matched: 0, not an
# omission.
MATCH_RE = re.compile(
    r"ane|iop-ane|ascwrap|exclave|sk-|sio-ane|dart-ane|ane-dart|t8020",
    re.I)
NEVER_DUMP = {"chosen", "ramdisk", "diags"}
iodt = None
raw_nodes = []
parent_cells = {}
try:
    iodt = _plist_argv_output(
        ["ioreg", "-a", "-p", "IODeviceTree", "-l"], timeout=120)
    if iodt is None:
        raise RuntimeError("ioreg IODeviceTree exit nonzero")
    stack = [(list(iodt) if isinstance(iodt, list) else [iodt], "")]
    matched = 0
    while stack:
        nodes, parent = stack.pop()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            name = _text(node.get("name"))
            kids = node.get("IORegistryEntryChildren")
            path = parent + "/" + (name or "?")
            if isinstance(kids, list):
                stack.append((kids, path))
            if parent == "/arm-io" or parent == "/":
                cells = {k: _full_value(node.get(k), out["truncated"])
                         for k in ("ranges", "#address-cells",
                                   "#size-cells") if node.get(k) is not None}
                if cells:
                    parent_cells[parent] = cells
            if not name or name.lower() in NEVER_DUMP:
                continue
            hit = bool(MATCH_RE.search(name))
            if not hit:
                for key in ("compatible", "device_type", "role"):
                    v = node.get(key)
                    texts = v if isinstance(v, list) else [v]
                    if any(t is not None and MATCH_RE.search(_text(t) or "")
                           for t in texts):
                        hit = True
                        break
            if not hit:
                continue
            matched += 1
            if matched > 24:
                if matched == 25:
                    out["truncated"].append("dump_nodes:cap")
                continue
            props = {}
            for key in sorted(node.keys()):
                if key == "IORegistryEntryChildren":
                    continue
                props[key] = _full_value(node[key], out["truncated"])
            props = _strip_props(props, out["stripped"])
            out["nodes"].append({
                "path": path[:256], "name": name,
                "compatible": _compatible(node), "props": props,
            })
            raw_nodes.append({k: v for k, v in node.items()
                              if k != "IORegistryEntryChildren"})
    out["arm_io"] = parent_cells.get("/arm-io")
    if len(raw_nodes) > 24:
        raw_nodes = raw_nodes[:24]
except Exception as exc:
    out["truncated"].append("iodt_dump:%s" % type(exc).__name__)

# AIC identity for the interrupt chain (max-irq, #interrupt-cells).
try:
    stack = [(list(iodt) if isinstance(iodt, list) else [iodt], "")]
    while stack:
        nodes, parent = stack.pop()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            name = _text(node.get("name"))
            kids = node.get("IORegistryEntryChildren")
            if isinstance(kids, list):
                stack.append((kids, parent + "/" + (name or "?")))
            if name and name.startswith("aic") and out["aic"] is None:
                props = {key: _full_value(node[key], out["truncated"])
                         for key in sorted(node.keys())
                         if key != "IORegistryEntryChildren"}
                out["aic"] = {"path": (parent + "/" + name)[:256],
                              "props": _strip_props(props, out["stripped"])}
except Exception as exc:
    out["truncated"].append("aic_walk:%s" % type(exc).__name__)

# pmgr ANE power rows: devices entries and power-gate names matching
# ANE (ANE_SYS, ANE_CPU, ANE_TD, ANE_BASE, ANE_SET*, PMP), ps-regs
# index, parent, flags; raw hex kept for later decode.
try:
    for node_entry in out["nodes"]:
        if (node_entry.get("name") or "").lower() != "pmgr":
            continue
        props = node_entry["props"]
        rows = []
        for key, value in props.items():
            lk = key.lower()
            if lk in ("devices", "power-gates", "ps-regs",
                      "voltage-states1", "voltage-states2",
                      "voltage-states5", "clock-gates", "perf-regs") \
                    or "tunables" in lk:
                rows.append({"name": key, "value": value})
        if rows:
            node_entry["pmgr_ane_rows"] = rows
except Exception as exc:
    out["truncated"].append("pmgr_rows:%s" % type(exc).__name__)

# Tunables: every *tunables* property on captured nodes is already in
# props; index them so a reader does not re-scan hex blobs.
try:
    tunables = []
    for node_entry in out["nodes"]:
        for key, value in (node_entry.get("props") or {}).items():
            if "tunables" in key.lower():
                tunables.append({"node": node_entry["path"],
                                 "property": key, "value": value})
    out["tunables"] = tunables[:32]
except Exception as exc:
    out["truncated"].append("tunables:%s" % type(exc).__name__)

# IOService class discovery: any ANE-related driver class, current or
# future (H11ANEIn, AppleH1xANE*, AppleExclave*/SEP ANE, ANECompiler).
CLASS_RE = re.compile(
    r"AppleH1[0-9A-Za-z]*ANE|AppleH[0-9]+ANE|AppleANE|ANEInterface|"
    r"ANECompiler|IOANE|AppleT[0-9]+ANE|AppleExclave.*ANE|AppleSEP.*ANE",
    re.I)
try:
    tree = _plist_argv_output(
        ["ioreg", "-a", "-p", "IOService", "-l"], timeout=120)
    if tree is None:
        raise RuntimeError("ioreg IOService exit nonzero")
    stack = list(tree) if isinstance(tree, list) else [tree]
    seen = set()
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        kids = node.get("IORegistryEntryChildren")
        if isinstance(kids, list):
            stack.extend(kids)
        haystack = " ".join(
            _text(node.get(k)) or "" for k in
            ("IOClass", "IOProviderClass", "CFBundleIdentifier"))
        if not CLASS_RE.search(haystack):
            continue
        ioclass = _text(node.get("IOClass"))
        if ioclass in seen:
            continue
        seen.add(ioclass)
        out["classes"].append({
            "ioclass": ioclass,
            "provider_class": _text(node.get("IOProviderClass")),
            "bundle_id": _text(node.get("CFBundleIdentifier")),
            "user_client": _text(node.get("IOUserClientClass")),
            "property_names": sorted(str(k) for k in node.keys())[:64],
        })
    out["classes"] = out["classes"][:16]
except Exception as exc:
    out["truncated"].append("classes:%s" % type(exc).__name__)

# Loaded ANE kexts: bundle id + version; Mach-O size + sha256 when the
# file is readable (SIP/KDK: say so, never skip silently).
try:
    proc = _run(["kmutil", "showloaded", "--list", "--no-statistics"],
                timeout=60)
    if proc.returncode == 0:
        text = proc.stdout.decode("utf-8", "replace")
        for line in text.splitlines():
            if "ANE" not in line and "ane" not in line:
                continue
            bundle = None
            for token in line.split():
                if token.startswith("com.apple."):
                    bundle = token.strip("()")
                    break
            entry = {"line": line.strip()[:256], "bundle_id": bundle,
                     "version": None, "macho_bytes": None,
                     "sha256": None, "unavailable": None}
            if bundle:
                kpath = "/System/Library/Extensions/%s.kext" % bundle
                try:
                    blob = _json_argv_output(
                        ["plutil", "-convert", "json", "-o", "-",
                         kpath + "/Contents/Info.plist"])
                    entry["version"] = _text(
                        (blob or {}).get("CFBundleVersion"))
                except Exception:
                    entry["unavailable"] = "info_plist_unreadable"
                macho = "%s/Contents/MacOS/%s" % (
                    kpath, bundle.rsplit(".", 1)[-1])
                try:
                    import hashlib as _khashlib
                    digest = _khashlib.sha256()
                    with open(macho, "rb") as kfh:
                        size = 0
                        for chunk in iter(lambda: kfh.read(1024 * 1024),
                                          b""):
                            digest.update(chunk)
                            size += len(chunk)
                    entry["macho_bytes"] = size
                    entry["sha256"] = digest.hexdigest()
                except OSError as kexc:
                    entry["unavailable"] = \
                        entry["unavailable"] or ("macho: %s"
                                                 % type(kexc).__name__)
            out["kexts"].append(entry)
        out["kexts"] = out["kexts"][:16]
    else:
        out["kexts"] = [{"unavailable": "kmutil_exit_%d" % proc.returncode}]
except Exception as exc:
    out["kexts"] = [{"unavailable": "kmutil:%s" % type(exc).__name__}]

# OS-shipped ANE firmware images: names, sizes, sha256 of public paths
# only. Never the files themselves.
try:
    import hashlib as _hashlib
    roots = (
        "/usr/standalone/firmware",
        "/System/Library/ExtensionKit",
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework",
        "/System/Library/Frameworks/CoreML.framework",
    )
    total = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs.sort()
            for fname in sorted(files):
                low = fname.lower()
                if not ("ane" in low and (
                        "fw" in low or "_ane_" in low or low.startswith(
                            "anef") or low.endswith(".im4p"))):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(path)
                    if total + size > 64 * 1024 * 1024 or \
                            len(out["firmware"]["files"]) >= 24:
                        out["firmware"]["truncated"] = "firmware:cap"
                        raise StopIteration
                    digest = _hashlib.sha256()
                    with open(path, "rb") as fh:
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            digest.update(chunk)
                    total += size
                except StopIteration:
                    raise
                except OSError:
                    continue
                out["firmware"]["files"].append({
                    "path": path[:512], "size": size,
                    "sha256": digest.hexdigest()})
                out["firmware"]["matched"] = \
                    len(out["firmware"]["files"])
except StopIteration:
    pass
except Exception as exc:
    out["truncated"].append("firmware:%s" % type(exc).__name__)

# Raw plist text of the matched nodes for the archive member (strip
# list already applied above); capped at 1 MiB.
try:
    blob = plistlib.dumps(raw_nodes or [])
    if len(blob) > 1024 * 1024:
        out["truncated"].append("raw_text:cap")
        blob = blob[:1024 * 1024]
    out["raw_text"] = blob.decode("utf-8", "replace")
except Exception as exc:
    out["truncated"].append("raw_text:%s" % type(exc).__name__)

out["available"] = bool(out["nodes"])


def _plain(value):
    if isinstance(value, bytes):
        return _text(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


print(json.dumps(_plain(out))[:900000])
"""

ANE_MACOS_SMOKE_PROBE = r"""
import json, time
out = {"available": False, "calls": 20, "min_ms": None, "median_ms": None,
       "error": None}
try:
    import coremltools as ct
    import numpy as np
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder
    builder = NeuralNetworkBuilder(
        [("x", datatypes.Array(2)), ("y", datatypes.Array(2))],
        [("z", None)])
    builder.add_elementwise(name="add", input_names=["x", "y"],
                            output_name="z", mode="ADD")
    model = ct.models.MLModel(builder.spec)
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([3.0, 4.0], dtype=np.float32)
    times = []
    for _ in range(out["calls"]):
        start = time.perf_counter()
        got = model.predict({"x": x, "y": y})["z"]
        times.append((time.perf_counter() - start) * 1000)
        if not np.allclose(got, x + y, atol=1e-4):
            out["error"] = "wrong result"
            break
    else:
        out["available"] = True
        out["min_ms"] = round(min(times), 3)
        out["median_ms"] = round(sorted(times)[len(times) // 2], 3)
except Exception as exc:
    out["error"] = ("%s: %s" % (type(exc).__name__, exc))[:200]
print(json.dumps(out))
"""


def probe_ane_port(redactor):
    """ANE porting facts from the working macOS installation.

    The macOS equivalents of the Linux devicetree capture: the driver
    instances (core count, hardware generation, firmware state), the
    DT-shaped ANE provider and ANE DART/mapper nodes with their MMIO
    ranges and interrupt specifiers, CoreML compute-unit availability,
    and a root-gated powermetrics sample. Everything is read-only; only
    powermetrics wants root, and its absence is recorded, not fatal.
    """
    rec = run_python_probe(ANE_PROBE_CODE, redactor,
                           label="ane-port macos", timeout=180)
    if rec["exit_code"] != 0 or not rec["stdout"].strip():
        return {"available": False, "macos": None,
                "error": rec["error"] or rec["stderr"][:256] or
                "ane probe exit %s" % rec["exit_code"]}
    try:
        detail = json.loads(rec["stdout"])
    except ValueError:
        return {"available": False, "macos": None, "error": "bad-probe-json"}
    detail = {key: detail[key] for key in
              ("available", "instances", "ane_nodes", "dart_nodes",
               "mailbox_nodes", "dt_nodes", "pmgr_nodes", "coreml",
               "powermetrics", "truncated", "driver", "platform",
               "compiler", "set_base_candidate")
              if key in detail}
    macos = redactor.apply_value(detail)
    if "truncated" in macos:
        macos["truncated"] = bound_markers(macos["truncated"])
    return {"available": bool(detail.get("available")), "macos": macos}


def probe_ane_dump(redactor):
    """Full-property IODeviceTree dump for the deep collector.

    Generic ANE-family discovery (ane / iop-ane / ascwrap / exclave /
    dart-ane / t8020 patterns) so M3-M6 machines are captured without
    code changes: every matched node with ALL properties (strip list
    applied, hex for blobs), the AIC identity, pmgr ANE power rows,
    ANE-related tunables, ANE driver classes, loaded ANE kexts, and the
    OS-shipped ANE firmware images (hash + size only, never the files).
    `raw_text` is the redacted plist text of the matched nodes; the
    deep collector stores it as an archive member.
    """
    rec = run_python_probe(ANE_MACOS_DUMP_PROBE, redactor,
                           label="ane-dt dump macos", timeout=240,
                           max_chars=PROBE_STREAM_CHARS)
    if rec["exit_code"] != 0 or not rec["stdout"].strip():
        return {"available": False, "error": rec["error"]
                or rec["stderr"][:256] or "dump probe exit %s"
                % rec["exit_code"]}
    try:
        detail = json.loads(rec["stdout"])
    except ValueError:
        return {"available": False, "error": "bad-dump-json"}
    detail = {key: detail[key] for key in
              ("available", "nodes", "arm_io", "aic", "classes", "kexts",
               "firmware", "tunables", "raw_text", "truncated", "stripped")
              if key in detail}
    return {"available": bool(detail.get("available")),
            "dump": redactor.apply_value(detail)}


def probe_ane_smoke(redactor):
    """Opt-in macOS ANE smoke: a tiny CoreML add model, 20 calls.

    Records min/median wall time; coremltools absence is data
    ("unavailable"), never a failure.
    """
    rec = run_python_probe(ANE_MACOS_SMOKE_PROBE, redactor,
                           label="ane smoke macos", timeout=180)
    if rec["exit_code"] != 0 or not rec["stdout"].strip():
        return {"available": False, "unavailable": "probe exit %s"
                % rec["exit_code"], "error": rec["error"]
                or rec["stderr"][:256]}
    try:
        result = json.loads(rec["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"available": False, "unavailable": "unparseable smoke output"}
    return redactor.apply_value(result)



def _text(record):
    return record["stdout"].strip() if record["exit_code"] == 0 else None


def _int(value):
    return int(value) if value and str(value).isdigit() else None


def probe_host(redactor):
    facts = bench_matrix.host_facts()
    model = run_tool(["sysctl", "-n", "hw.model"], redactor, timeout=10)
    active = run_tool(["sysctl", "-n", "hw.activecpu"], redactor, timeout=10)
    memory = facts.get("memsize_bytes")
    return {
        "available": True,
        "system": "Darwin",
        "arch": facts.get("machine"),
        "kernel_release": platform.release(),
        "os": facts.get("os"),
        "model": _text(model),
        "chip": facts.get("chip"),
        "cpu_online": _int(_text(active)),
        "cpu": {"present": _int(facts.get("cores")), "hotplug_control": None},
        "memory_total_mib": memory // (1024 * 1024) if memory else None,
        "gpu": facts.get("gpu"),
    }


def measurement_context():
    power = bench_matrix.power_state()
    processes = bench_matrix.clean_check()
    return {
        "power": {key: power.get(key) for key in ("source", "percent", "charging")}
        if power else None,
        "model_processes": {
            "status": processes["status"],
            "scanned": processes["scanned"],
            "matched_count": len(processes["matched"]),
        },
        "limits": "Process scan covers known model tools only. Other GPU activity "
                  "and temperature are not measured. Timings are observations, "
                  "not a controlled performance comparison.",
    }
