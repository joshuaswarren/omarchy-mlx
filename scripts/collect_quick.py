#!/usr/bin/env python3
"""Collect a fast, privacy-safe capability report for mlx-omarchy.

The report answers the questions a maintainer asks first: which Apple
Silicon machine, which kernel, which Mesa/Honeykrisp Vulkan stack, is the
ANE visible, and which mlx-omarchy wheel is installed. It also records
the CPU topology (present cores versus online), the boot-chain firmware
identity from the devicetree /chosen node, the redacted kernel command
line, whether the booted devicetree carries an ANE node, and the full
driver-port devicetree capture (ane node, DARTs, PMGR domains, AIC)
for porting omarchy-ane to a new SoC. It finishes
in seconds, downloads nothing, and never touches the network.

On macOS, it records native Mac, OS, and Metal facts instead of Linux
driver and devicetree diagnostics. Native MLX is optional.

Personal data never reaches the output: every captured command output and
every free-text field passes through the shared Redactor, and the script
never reads serial numbers, machine IDs, or network configuration at all.

Usage:
  python3 scripts/collect_quick.py            # JSON on stdout
  python3 scripts/collect_quick.py --out f.json

Section shape (schema_version 1): every section carries "available"; a
missing tool or an import failure is recorded data, not an error. The
script exits 0 when the report was produced, 1 on an internal failure.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collect_macos
from collect_common import (SCHEMA_VERSION, Redactor, build_payload,
                            is_identity_prop,
                            dump_json, host_aliases, json_bytes, run_tool,
                            local_hostname)

DT_BASE = "/sys/firmware/devicetree/base"


def probe_host(redactor):
    if platform.system() == "Darwin":
        return collect_macos.probe_host(redactor)
    out = {"available": True}
    info = os.uname()
    out["arch"] = info.machine
    out["kernel_release"] = info.release
    try:
        out["cpu_online"] = len(os.sched_getaffinity(0))
    except AttributeError:
        out["cpu_online"] = os.cpu_count()
    try:
        out["page_size_bytes"] = os.sysconf("SC_PAGESIZE")
    except (ValueError, OSError):
        out["page_size_bytes"] = None
    mem = _mem_total()
    out["memory_total_mib"] = mem
    out["devicetree"] = _devicetree(redactor)
    out["cpu"] = _cpu_topology()
    out["boot"] = _boot_chain(redactor)
    out["cmdline"] = _kernel_cmdline(redactor)
    out["core_shortfall"] = _core_shortfall(out["cpu"],
                                            out["cmdline"] or "")
    return out


def _mem_total():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _devicetree(redactor):
    """Machine model and chip compatible strings; never the serial."""
    out = {"model": None, "compatible": None}
    model = _read_dt_file("model")
    if model:
        out["model"] = redactor.apply(model.strip("\x00\n"), field="model")
    compat = _read_dt_file("compatible")
    if compat:
        parts = [p for p in compat.split("\x00") if p]
        out["compatible"] = parts
    return out


def _read_dt_file(name, base=DT_BASE):
    try:
        with open(os.path.join(base, name), "rb") as fh:
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None


CPU_SYSFS = "/sys/devices/system/cpu"


def _read_cpu_list(path):
    """(ids, raw) from a sysfs cpu list like "0-3,8"; (None, None) absent."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
    except OSError:
        return None, None
    ids = []
    for part in raw.split(","):
        lo, sep, hi = part.partition("-")
        try:
            first = int(lo)
            last = int(hi) if sep else first
        except ValueError:
            return [], raw
        if last < first or last - first > 65535:
            return [], raw
        ids.extend(range(first, last + 1))
    return ids, raw


def _cpu_topology(base=CPU_SYSFS):
    """Core counts from sysfs: the machine's real size, not the taskset.

    host.cpu_online is scheduler affinity, so a contributor pinned to one
    core of eight looked identical to a one-core machine. `present` says
    how many cores the machine has. `hotplug_control` records whether
    cpuN/online switches exist; their absence is what spin-table bringup
    looks like on Apple Silicon.
    """
    out = {"present": None, "possible": None, "online": None,
           "offline": None, "present_list": None, "possible_list": None,
           "online_list": None, "offline_list": None,
           "hotplug_control": None}
    for key in ("present", "possible", "online", "offline"):
        ids, raw = _read_cpu_list(os.path.join(base, key))
        if raw is not None:
            out[f"{key}_list"] = raw or None
            out[key] = len(ids)
    try:
        entries = os.listdir(base)
    except OSError:
        return out
    out["hotplug_control"] = any(
        re.fullmatch(r"cpu([1-9]\d*)", name)
        and os.path.exists(os.path.join(base, name, "online"))
        for name in entries)
    return out


CHOSEN_PROPERTIES = (
    # (property under chosen, report key): boot firmware identity, never
    # personal. A bootloader devicetree override (GRUB's `devicetree`
    # command) REPLACES the m1n1-patched tree with a frozen snapshot, so
    # these strings describe whatever tree actually booted -- the
    # 2026-09-03 one-core incident came from exactly that override.
    ("asahi,m1n1-stage1-version", "m1n1_stage1"),
    ("asahi,m1n1-stage2-version", "m1n1_stage2"),
    ("asahi,iboot1-version", "iboot1"),
    ("asahi,iboot2-version", "iboot2"),
    ("asahi,system-fw-version", "system_fw"),
    ("asahi,os-fw-version", "os_fw"),
)


def _boot_chain(redactor, base=DT_BASE):
    """Boot firmware identity from the live /chosen node."""
    out = {}
    for prop, key in CHOSEN_PROPERTIES:
        raw = _read_dt_file(os.path.join("chosen", prop), base)
        out[key] = redactor.apply(raw.strip("\x00\n")) if raw else None
    return out


def _kernel_cmdline(redactor, path="/proc/cmdline"):
    """The live kernel command line, redacted.

    Carries root=UUID=... (the UUID rule replaces it) and reveals
    maxcpus/nosmp clamps or a custom-devicetree boot.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None
    return redactor.apply(raw.strip()) or None


def _core_shortfall(cpu, cmdline):
    """Plain fact: fewer cores running than present, unexplained.

    Recorded only when the command line carries no maxcpus=/nosmp clamp;
    a clamp is a deliberate choice, not a failure. Numbers, not alarms.
    """
    present = cpu.get("present")
    online = cpu.get("online")
    if not isinstance(present, int) or not isinstance(online, int):
        return None
    if present <= online:
        return None
    if cmdline and re.search(r"\bmaxcpus=\d+\b|\bnosmp\b", cmdline):
        return None
    return {"present": present, "online": online}


def _ane_devicetree(base=DT_BASE):
    """ANE node visibility in the booted devicetree, unlike /dev/ane.

    Packaged t8103 dtbs ship no ane node; a node appears only when the
    bootloader overrides the tree. This answers whether the running
    kernel was even offered an ANE by its boot chain.
    """
    out = {"node": False, "compatible": None}
    matches = []
    for dirpath, _dirs, _files in os.walk(base):
        named = bool(re.search(r"(?:^|/)ane(?:@[0-9a-f]+)?$", dirpath))
        tokens = [t for t in (_read_dt_file("compatible", dirpath) or "")
                  .split("\x00") if t]
        hit = [t for t in tokens if t == "apple,ane" or t.endswith("-ane")]
        if named or hit:
            matches.extend(hit or tokens or [os.path.basename(dirpath)])
    if matches:
        out["node"] = True
        out["compatible"] = sorted(set(matches))[:8]
    return out

def _read_dt_raw(name, base=DT_BASE):
    """Raw property bytes; None when absent."""
    try:
        with open(os.path.join(base, name), "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _dt_strings(raw):
    """Null-terminated string list from a devicetree property blob."""
    parts = [p for p in raw.split(b"\x00") if p]
    try:
        texts = [p.decode("utf-8") for p in parts]
    except UnicodeDecodeError:
        return None
    if any(not t.isprintable() for t in texts):
        return None
    return texts


def _dt_u32s(raw):
    """Big-endian u32 cell list from a devicetree property blob."""
    if len(raw) % 4 or not raw:
        return None
    return [int.from_bytes(raw[i:i + 4], "big")
            for i in range(0, len(raw), 4)]


def _dt_reg(raw, base):
    """reg as ["0xADDR/0xSIZE", ...] using the PARENT node's cell
    counts. DT semantics put #address-cells/#size-cells on the parent
    of the node carrying reg; a live t8103 ane node carries none of its
    own, so reading them from the node itself mis-parses every reg."""
    cells = _dt_u32s(raw)
    if cells is None:
        return None
    parent = os.path.dirname(base)
    if not os.path.isdir(parent):
        parent = base

    def count(name, default):
        raw_n = _read_dt_raw(name, parent)
        vals = _dt_u32s(raw_n) if raw_n else None
        return vals[0] if vals else default

    ac = count("#address-cells", 2)
    sc = count("#size-cells", 1)
    width = ac + sc
    out = []
    for i in range(0, len(cells) - width + 1, width):
        addr = 0
        for c in cells[i:i + ac]:
            addr = (addr << 32) | c
        size = 0
        for c in cells[i + ac:i + width]:
            size = (size << 32) | c
        out.append(f"0x{addr:x}/0x{size:x}")
    return out


def _dt_props(node_dir, redactor, cap=16):
    """Every property of one devicetree node, decoded and bounded.

    Strings stay strings, reg gets address/size decoding, everything
    numeric becomes a u32 cell list.
    """
    out = {}
    for name in sorted(os.listdir(node_dir)):
        path = os.path.join(node_dir, name)
        if not os.path.isfile(path):
            continue
        if is_identity_prop(name):
            redactor._note("additional_identity_properties_removed")
            continue
        try:
            with open(path, "rb") as fh:
                raw = fh.read(4096)
        except OSError:
            continue
        if name == "reg":
            value = _dt_reg(raw, node_dir)
        else:
            value = _dt_strings(raw)
            if value is not None and len(value) == 1:
                value = value[0]
            elif value is None:
                value = _dt_u32s(raw)
            if value is None:
                value = raw[:64].hex()
        if isinstance(value, list):
            value = value[:cap]
        elif isinstance(value, str):
            value = redactor.apply(value[:512])
        out[name] = value
    return out


_ADT_CANDIDATES = (
    ("/boot/efi/m1n1", "m1n1"),
    ("/boot/m1n1", "m1n1"),
    ("/boot/asahi/adt", "adt"),
    ("/boot/efi/asahi/adt", "adt"),
    ("/sys/firmware/fdt", "fdt"),
)


def _probe_adt(redactor):
    """Best-effort Apple Device Tree artifact discovery: presence + sha256.

    The bytes stay on the machine; the hash pins which tree a capture
    explains. `source` records which candidate matched: m1n1 stage2
    (embeds the ADT), a distro ADT dump, or the live FDT. found=false
    with source=null is a platform fact, not a probe failure.
    """
    for path, source in _ADT_CANDIDATES:
        try:
            with open(path, "rb") as fh:
                data = fh.read(8 * 1024 * 1024)
        except OSError:
            continue
        if not data:
            continue
        return {"found": True,
                "path": redactor.apply(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source": source}
    return {"found": False, "path": None, "sha256": None, "source": None}


def _ane_reserved_memory(redactor, base=DT_BASE, max_nodes=32):
    """The devicetree reserved-memory subtree, bounded.

    The ANE driver allocates its fence/IOVA pools inside the reserved
    regions iBoot carves; a port to an untested chip needs every child
    (name, reg, no-map, reusable, compatible) to place them. Each item
    a reader cannot get is an explicit fact, never an omission: a tree
    without reserved-memory reports found=false.
    """
    parent = os.path.join(base, "reserved-memory")
    out = {"found": False, "child_count": 0, "nodes": {}}
    if not os.path.isdir(parent):
        return out
    out["found"] = True
    children = sorted(d for d in os.listdir(parent)
                      if os.path.isdir(os.path.join(parent, d)))
    out["child_count"] = len(children)
    for child in children[:max_nodes]:
        out["nodes"][child] = _dt_props(os.path.join(parent, child), redactor)
    if len(children) > max_nodes:
        out["truncated"] = f"nodes:{len(children) - max_nodes}"
    return out


def _ane_scan_nodes(redactor, base):
    """One device-tree walk classifying every node: ANE-family nodes,
    DARTs, the ANE mailbox, pmgr blocks/domains, AIC, and the phandle map.

    Generic ANE-family match: node names vary per generation (ane@,
    dart-ane@, iop-ane@, ane-ascwrap@, mailbox-ane@, M5 exclave nodes)
    and compatibles carry ascwrap-v6 / t8020-ane style markers, so match
    patterns, not fixed names. The tree root itself never matches by name
    (a temp dir's random name must not decide this). Children of a pmgr
    block are power domains (ane_sys / ane_set* pwrstates), captured via
    pmgr_domains below - never the ANE device itself.
    """
    ane_nodes = {}
    darts = {}
    mailbox = {}
    phandles = {}
    pmgr_dirs = set()
    pmgr_domains = []
    pmgr_blocks = []
    aic = None
    for dirpath, dirs, _files in os.walk(base):
        dirs.sort()
        compat = _dt_strings(_read_dt_raw("compatible", dirpath) or b"") \
            or []
        rel = os.path.relpath(dirpath, base)
        ph = _dt_u32s(_read_dt_raw("phandle", dirpath) or b"")
        if ph:
            phandles[ph[0]] = rel
        named = rel != "." and bool(re.search(
            r"(?:^|/)[^/]*(?:(?<![a-z])ane(?![a-z])|ascwrap|exclave)[^/]*$",
            dirpath))
        hit = [t for t in compat
               if t == "apple,ane" or t.endswith("-ane")
               or "ascwrap" in t or "exclave" in t or t.startswith("ane,")]
        if any(t == "apple,pmgr" for t in compat):
            pmgr_dirs.add(rel)
        if os.path.dirname(rel) in pmgr_dirs or \
                os.path.basename(rel).startswith("power-controller@"):
            named = False
            hit = []
        if named or hit:
            ane_nodes[rel] = _dt_props(dirpath, redactor)
        # The ANE mailbox (`apple,<soc>-mailbox-ane` and friends): the
        # driver needs its reg window and its AIC line (the send-empty
        # pick) on SoCs where the stock apple-mailbox path binds. Node
        # names vary across ADT generations; the compatible pair
        # ("mailbox" + "ane") is the stable marker.
        mb = [t for t in compat if "mailbox" in t and "ane" in t]
        if mb or ("mailbox" in rel and "ane" in rel):
            mailbox[rel] = _dt_props(dirpath, redactor)
        # Real trees name DART nodes `iommu@<addr>` and tag them
        # `apple,<soc>-dart` (t600x drops the legacy `apple,dart`
        # fallback entirely), so match on the compatible suffix, not the
        # node name alone.
        if any(t == "apple,dart" or t.endswith("-dart") for t in compat) \
                or re.search(r"(?:^|/)dart[0-9a-f@-]", rel):
            darts[rel] = _dt_props(dirpath, redactor)
        if any(t == "apple,pmgr" for t in compat):
            children = []
            ane_children = []
            for child in dirs:
                cpath = os.path.join(dirpath, child)
                if not os.path.isdir(cpath):
                    continue
                ccompat = _dt_strings(
                    _read_dt_raw("compatible", cpath) or b"") or []
                label = _dt_strings(_read_dt_raw("label", cpath) or b"")
                label = (label or [None])[0]
                children.append({
                    "name": child[:128],
                    "label": label,
                    "compatible": ccompat[:8],
                })
                if re.search(r"ane", (label or "") + " " + child, re.I):
                    ane_children.append({
                        "path": os.path.relpath(cpath, base)[:256],
                        "label": label,
                        "compatible": ccompat[:8],
                    })
            reg_raw = _read_dt_raw("reg", dirpath)
            pmgr_blocks.append({
                "path": rel[:256],
                "reg": _dt_reg(reg_raw, dirpath) if reg_raw else None,
                "children": children[:256],
                "children_total": len(children),
            })
            pmgr_domains.extend(ane_children)
        # t8103 uses "apple,aic"; t600x/t602x use "apple,<soc>-aic",
        # "apple,aic2".
        if aic is None and any(t in ("apple,aic", "apple,aic2")
                               or t.endswith("-aic") for t in compat):
            aic = {"path": rel, "compatible": compat[:8]}
    return ane_nodes, darts, mailbox, phandles, pmgr_domains, pmgr_blocks, aic


def _ane_resolve_iommus(ane_nodes, darts, phandles):
    """Resolve iommu phandles to DART paths so the contributor does not
    have to do phandle arithmetic by hand. Each entry is one phandle
    plus #iommu-cells specifiers from the target DART. Mutates ane_nodes."""
    for props in ane_nodes.values():
        iommus = props.get("iommus")
        if not isinstance(iommus, list):
            continue
        resolved = []
        i = 0
        while i < len(iommus) and len(resolved) < 8:
            cell = iommus[i]
            i += 1
            if not isinstance(cell, int):
                continue
            target = phandles.get(cell, f"phandle:{cell}")
            resolved.append(target)
            dart = darts.get(target) or {}
            ncells = dart.get("#iommu-cells")
            if isinstance(ncells, list) and ncells and \
                    isinstance(ncells[0], int):
                i += ncells[0]
        props["iommus_resolved"] = resolved


def _ane_reg_view(ane_nodes):
    """Flat ANE reg/size view: present when the DT declares the node
    (M1 family), explicitly None when it does not (t602x)."""
    ane_reg = []
    for props in ane_nodes.values():
        regs = props.get("reg")
        if isinstance(regs, list):
            ane_reg.extend(r for r in regs if isinstance(r, str))
    return ane_reg[:8] or None


def _ane_boot_provenance(redactor, base):
    """Boot provenance as structured fields: every asahi,* property under
    /chosen (m1n1 stages, iBoot, system/os FW), plus model and root
    compatible. Strings pass the redactor like every free-text field."""
    chosen = {}
    for name in sorted(os.listdir(os.path.join(base, "chosen"))
                       if os.path.isdir(os.path.join(base, "chosen"))
                       else []):
        if not name.startswith("asahi,"):
            continue
        raw = _read_dt_file(os.path.join("chosen", name), base)
        value = redactor.apply(raw.strip("\x00\n")[:256]) if raw else None
        if value:
            chosen[name] = value
        if len(chosen) >= 32:
            break
    model = _read_dt_file("model", base)
    return {
        "model": redactor.apply(model.strip("\x00\n")[:256], field="model")
        if model else None,
        "compatible": (_dt_strings(
            _read_dt_raw("compatible", base) or b"") or [])[:8] or None,
        "chosen": chosen or None,
    }


def _dtb_identity(fdt_path):
    """DTB identity: hash only, never the blob, so a submission's device
    tree is reproducible/comparable across kernels. /sys/firmware/fdt
    is root-only on every shipped kernel (59 published rows carry
    null); the explicit reason is recorded, and the collector never
    calls sudo itself."""
    try:
        with open(fdt_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest(), None
    except PermissionError:
        return None, "needs root"
    except OSError:
        return None, "unreadable"


def _referenced_phandles(ane_nodes, darts, phandles):
    """Ship only the phandles the porting data actually resolves: the
    iommus / power-domains cells of the ane nodes and DARTs. The full
    map runs to hundreds of entries on t600x and blew the 64 KiB
    payload budget before the DARTs it exists to explain did."""
    referenced = set()
    for props in list(ane_nodes.values()) + list(darts.values()):
        for key in ("iommus", "power-domains"):
            cells = props.get(key)
            if isinstance(cells, list):
                referenced.update(c for c in cells
                                  if isinstance(c, int) and c in phandles)
    return referenced


def _ane_set_base_candidate(pmgr_domains):
    """SET-candidate labeling (Linux side): the ane_set* power-controller
    children are 4-byte pwrstate CELLS (t602x cluster at pmgr+0x4000),
    never the SET MMIO window. The device tree cannot source the SET
    base: it needs the macOS driver-window capture (macos.
    set_base_candidate with driver_window_confirms=true) or an m1n1
    ANE.ps_map probe. Same field name as the macOS side so a generator
    can never conflate the pwrstate cluster with the SET candidate."""
    ane_pwrstate_cells = [
        {"label": e["label"],
         "offset": int(e["path"].rsplit("@", 1)[-1], 16)}
        for e in pmgr_domains
        if e.get("label") and "@" in (e.get("path") or "")
        and e["path"].rsplit("@", 1)[-1].isalnum()
    ][:16]
    return {
        "status": "not_available_from_device_tree",
        "ane_pwrstate_cells": ane_pwrstate_cells,
        "note": "ane_set* entries are 4-byte power-controller pwrstate "
                "cells, not the SET MMIO window; the SET base must come "
                "from a macOS set_base_candidate (driver_window_confirms) "
                "or m1n1 ANE.ps_map, never from these offsets",
    }


def _ane_port_devicetree(redactor, base=DT_BASE,
                         fdt_path="/sys/firmware/fdt"):
    """Everything a contributor needs to port omarchy-ane to this SoC.

    Captures the ane node(s) in full (MMIO reg, reg-names, IRQs, iommus,
    power-domains, status, compatible), every DART node with its full
    props (reg ranges, #iommu-cells, interrupts), the COMPLETE pmgr
    offset topology (block reg plus every power-controller child with
    node name / label / compatible), the ANE-labelled pmgr subset, the
    AIC compatible, structured boot provenance (all asahi,* /chosen
    properties, model, root compatible), and the sha256 of the booted
    DTB so a submission's tree is reproducible across kernels.

    The pmgr offset map is what derives the ANE SET-block base on a new
    SoC: on the known-good references the driver constant equals the
    pmgr block base + 0xc000 (t8103 0x23b700000 -> 0x23b70c000, t6001
    0x28e080000 -> 0x28e08c000), and no power-controller node exists at
    that offset. A tree with NO ane node still dumps DART/PMGR/AIC:
    that is exactly what authoring the overlay requires.
    """
    ane_nodes, darts, mailbox, phandles, pmgr_domains, pmgr_blocks, aic = \
        _ane_scan_nodes(redactor, base)
    _ane_resolve_iommus(ane_nodes, darts, phandles)
    ane_reg = _ane_reg_view(ane_nodes)
    boot = _ane_boot_provenance(redactor, base)
    dtb_sha256, dtb_sha256_error = _dtb_identity(fdt_path)
    referenced = _referenced_phandles(ane_nodes, darts, phandles)
    return {
        "ane_node_present": bool(ane_nodes),
        "ane_nodes": ane_nodes,
        "ane_reg": ane_reg,
        "darts": darts,
        "mailbox": mailbox,
        "reserved_memory": _ane_reserved_memory(redactor, base=base,
                                                max_nodes=8),
        "pmgr_domains": pmgr_domains[:64],
        "pmgr_blocks": pmgr_blocks[:8],
        "aic": aic,
        "set_base_candidate": _ane_set_base_candidate(pmgr_domains),
        "phandles": {str(k): phandles[k] for k in sorted(referenced)},
        "boot": boot,
        "dtb_sha256": dtb_sha256,
        "dtb_sha256_error": dtb_sha256_error,
        "adt": _probe_adt(redactor),
        "adt_nodes": None,
    }


_FIRMWARE_ROOTS = (
    "/lib/firmware/apple/ane",
    "/var/lib/omarchy-ane",
    "/lib/firmware",
    "/usr/lib/firmware",
)


def _ane_firmware(redactor, max_files=32, roots=None):
    """ANE firmware placement: names, sizes, sha256; bytes stay local.

    Walks the known firmware roots and hashes every regular file whose
    name mentions the ANE (the Asahi layout keeps the per-SoC blobs
    under /lib/firmware/apple/ane; omarchy-ane pins copies under
    /var/lib/omarchy-ane). Symlinks are skipped so a linked duplicate
    is hashed once.
    """
    roots = _FIRMWARE_ROOTS if roots is None else roots
    out = {"files": [], "roots": list(roots[:2]), "truncated": None}
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirs, files in os.walk(root):
            dirs.sort()
            for name in sorted(files):
                if "ane" not in name.lower():
                    continue
                path = os.path.join(dirpath, name)
                if path in seen or os.path.islink(path):
                    continue
                seen.add(path)
                if len(out["files"]) >= max_files:
                    out["truncated"] = "files:max"
                    return out
                try:
                    with open(path, "rb") as fh:
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                            digest.update(chunk)
                    size = os.path.getsize(path)
                except OSError:
                    continue
                out["files"].append({
                    "name": redactor.apply(name)[:256],
                    "path": redactor.apply(path)[:512],
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                })
    return out


def _ane_port_runtime(redactor):
    """Live runtime facts about the ane driver, redacted and bounded."""
    out = {"iomem": None, "module_version": None, "srcversion": None,
           "loaded": None, "dmesg": None}
    try:
        with open("/proc/iomem", "r", encoding="utf-8",
                  errors="replace") as fh:
            lines = [redactor.apply(line.rstrip("\n")[:512])
                     for line in fh
                     if re.search(r"ane|dart", line, re.I)]
        out["iomem"] = lines[:32] or None
    except OSError:
        pass
    for key, name in (("module_version", "version"),
                      ("srcversion", "srcversion")):
        try:
            with open(os.path.join("/sys/module/ane", name),
                      "r", encoding="utf-8") as fh:
                out[key] = redactor.apply(fh.read().strip()[:128]) or None
        except OSError:
            pass
    try:
        with open("/proc/modules", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("ane "):
                    out["loaded"] = redactor.apply(line.strip()[:256])
                    break
    except OSError:
        pass
    dmesg = run_tool(["sh", "-c",
                      "dmesg 2>/dev/null | grep -iE 'ane|dart|pmgr' "
                      "| tail -n 64"],
                     redactor, label="dmesg ane/dart/pmgr", timeout=15)
    if dmesg["exit_code"] == 0 and dmesg["stdout"].strip():
        out["dmesg"] = [redactor.apply(line[:512])
                        for line in dmesg["stdout"].splitlines()[-64:]]
    out["firmware"] = _ane_firmware(redactor)
    return out


def probe_ane_port(redactor):
    """Driver-port capture: devicetree plus live ane driver facts."""
    if platform.system() == "Darwin":
        return collect_macos.probe_ane_port(redactor)
    out = {"available": True}
    out["devicetree"] = _ane_port_devicetree(redactor)
    out["runtime"] = _ane_port_runtime(redactor)
    return out


def _device_blocks(text):
    """Split `vulkaninfo --summary` into one dict per GPUn: block."""
    devices = []
    current = None
    for line in text.splitlines():
        if re.match(r"^GPU\d+:$", line.strip()):
            current = {}
            devices.append(current)
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and " " not in key:
            current[key] = value.strip()
    return devices


def _primary_device(devices):
    """The device that actually matters for this project.

    A host with Honeykrisp also enumerates llvmpipe, and llvmpipe is
    listed last. Reporting it would make every submission useless for
    driver work, so prefer the real GPU: Honeykrisp first, then any
    non-CPU device, and only then whatever came first.
    """
    if not devices:
        return {}
    for dev in devices:
        if "HONEYKRISP" in dev.get("driverID", "").upper():
            return dev
    for dev in devices:
        if dev.get("deviceType", "") != "PHYSICAL_DEVICE_TYPE_CPU":
            return dev
    return devices[0]


def probe_mesa(redactor):
    """Vulkan stack identity from `vulkaninfo --summary`."""
    if platform.system() == "Darwin":
        return collect_macos.not_applicable()
    out = {"available": False, "properties": {}, "summary": None}
    rec = run_tool(["vulkaninfo", "--summary"], redactor,
                   label="vulkaninfo --summary", timeout=30)
    out["vulkaninfo"] = rec
    if not rec["available"] or rec["exit_code"] != 0:
        return out
    devices = _device_blocks(rec["stdout"])
    primary = _primary_device(devices)
    out["devices"] = devices
    out["device_count"] = len(devices)
    out["properties"] = primary
    known = ("deviceName", "driverName", "driverVersion", "apiVersion",
             "vendorID", "deviceID", "deviceType", "conformanceVersion",
             "driverID", "driverInfo")
    out["gpu"] = {k: primary[k] for k in known if k in primary}
    out["available"] = bool(out["gpu"])
    return out


def probe_mesa_package(redactor):
    """Mesa package version as a cross-check; distro-specific and optional."""
    if platform.system() == "Darwin":
        return collect_macos.not_applicable()
    return {
        "pacman": run_tool(["pacman", "-Q", "mesa"], redactor,
                           label="pacman -Q mesa", timeout=15),
        "dpkg": run_tool(["dpkg-query", "-W", "-f=${Package} ${Version}\\n",
                          "mesa"], redactor, label="dpkg-query mesa",
                         timeout=15),
    }

def probe_ane(redactor):
    """Apple Neural Engine visibility: device node and libane."""
    if platform.system() == "Darwin":
        return collect_macos.not_applicable()
    node = os.path.exists("/dev/ane")
    out = {"available": node, "device_node": node}
    out["devicetree"] = _ane_devicetree()
    lib = run_tool(["sh", "-c", "ldconfig -p 2>/dev/null | grep -i libane"],
                   redactor, label="ldconfig libane", timeout=15)
    out["libane"] = lib["stdout"].strip() if lib["exit_code"] == 0 else None
    out["libane_probe"] = lib
    pkg = run_tool(["pkg-config", "--modversion", "libane"], redactor,
                   label="pkg-config libane", timeout=15)
    out["libane_version"] = pkg["stdout"].strip() \
        if pkg["exit_code"] == 0 else None
    out["pkgconfig"] = pkg
    return out


MLX_PROBE_CODE = r"""
import json
out = {"distributions": {}, "import_ok": False, "import_error": None,
       "info_tool": None, "default_device": None, "mlx_version": None}
import importlib.metadata
for dist in ("mlx-omarchy", "mlx"):
    try:
        out["distributions"][dist] = importlib.metadata.version(dist)
    except Exception:
        pass
import pathlib
try:
    import mlx.core as mx
    out["import_ok"] = True
    import platform
    if platform.system() == "Darwin":
        out["metal_available"] = mx.metal.is_available()
        if out["metal_available"]:
            mx.set_default_device(mx.gpu)
    out["default_device"] = str(mx.default_device())
    out["mlx_version"] = getattr(mx, "__version__", None)
    # mlx is a namespace package (mlx.__file__ is None); anchor on the
    # extension module, which lives beside the shipped bin/ directory.
    cand = pathlib.Path(mx.__file__).resolve().parent / "bin" / "mlx-omarchy-info"
    if cand.exists():
        out["info_tool"] = str(cand)
except Exception as exc:
    out["import_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def probe_mlx(redactor):
    """Installed wheel identity plus the shipped capability tool."""
    rec = run_tool([sys.executable, "-c", MLX_PROBE_CODE], redactor,
                   label="mlx import probe", timeout=90)
    out = {"available": False, "probe": rec, "info": None,
           "capabilities": None}
    if not rec["available"] or rec["exit_code"] != 0:
        return out
    try:
        found = json.loads(rec["stdout"].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return out
    out["available"] = bool(found.get("import_ok"))
    for key in ("distributions", "import_error", "default_device",
                "mlx_version", "info_tool", "metal_available"):
        out[key] = found.get(key)
    tool = found.get("info_tool")
    if tool:
        cap = run_tool([tool, "--json"], redactor,
                       label="mlx-omarchy-info --json", timeout=60)
        out["info"] = cap
        if cap["exit_code"] == 0:
            try:
                out["capabilities"] = json.loads(cap["stdout"])
            except ValueError:
                out["capabilities"] = None
    return out


def _strip_volatile(value):
    """Drop run-to-run noise so equal machine state gives equal output."""
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items()
                if k != "duration_ms"}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


DEFAULT_PROBES = {
    "host": probe_host,
    "mesa": probe_mesa,
    "mesa_package": probe_mesa_package,
    "ane": probe_ane,
    "ane_port": probe_ane_port,
    "mlx": probe_mlx,
}


def collect(probes=None):
    """Build the report dict. `probes` overrides sections for tests."""
    active = dict(DEFAULT_PROBES)
    if probes:
        active.update(probes)
    redactor = Redactor()
    report = {
        "report": "mlx-omarchy-quick",
        "schema_version": SCHEMA_VERSION,
    }
    for name in sorted(active):
        try:
            report[name] = redactor.apply_value(active[name](redactor))
        except Exception as exc:
            report[name] = {
                "available": False,
                "error": redactor.apply(f"{type(exc).__name__}: {exc}"),
            }
    return _strip_volatile(report)


def main():
    import collect_submit

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", metavar="FILE",
                    help="also write the JSON report to FILE")
    ap.add_argument("--submit", nargs="?", default=None,
                    const=collect_submit.DEFAULT_ENDPOINT, metavar="URL",
                    help="publish this report to the public community "
                         "endpoint (the default URL) or to URL, after "
                         "printing it")
    args = ap.parse_args()
    report = collect()
    text = dump_json(report)
    if args.out:
        with open(args.out, "wb") as fh:
            fh.write(json_bytes(report))
        print(f"[receipt] wrote {args.out} ({len(json_bytes(report))} bytes)")
    print(text, end="" if text.endswith("\n") else "\n")
    raise SystemExit(maybe_submit(args, report))


def maybe_submit(args, report):
    """Publish only after the report was shown and consent was given.

    The import is local so that a plain run of this script loads no
    network module at all.
    """
    import collect_submit

    endpoint = collect_submit.endpoint_from_args(args)
    if endpoint is None:
        return 0
    payload = build_payload("quick", report, {}, redactor=Redactor())
    digest = collect_submit.sha256_hex(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    if not args.submit and not collect_submit.confirm_interactive(
            endpoint, "quick report", digest):
        print("[submit] declined; nothing was uploaded")
        return 0
    try:
        receipt = collect_submit.submit_payload(
            endpoint, payload, token=collect_submit.token_from_env(),
            aliases=host_aliases(local_hostname()))
    except collect_submit.SubmitError as exc:
        print(f"[submit] FAILED: {exc}", file=sys.stderr)
        return 4
    print(f"[receipt] public URL: {receipt['url']} "
          f"(deduplicated={receipt['deduplicated']})")
    return 0


if __name__ == "__main__":
    main()
