#!/usr/bin/env python3
"""Generate an ANE overlay .dts + ane_soc driver row from community deep submissions.

Input: one or more published submission rows (the JSON served by
/v1/results/<sha256>, or query_community_data.py show --json), any mix of
Linux and macOS rows for any SoCs.

Output per SoC (written to --outdir, default ./ane-soc-out):
  <soc>-ane-overlay.dts   device-tree overlay adding the ane node (needs a
                          same-soc Linux deep row for board topology)
  <soc>-ane-soc.c         the `struct ane_soc` entry + of_device_id line

Derivation rules (hard, by design):
  * The SET base comes ONLY from the macOS `set_base_candidate`
    (pmgr_block + 0xc000) with `driver_window_confirms` true -- i.e. Apple's
    own driver maps a window at pmgr+0xc000 for the ANE device. A
    driver_window_confirms=false row (t6002, t6030) yields NO set_base: the
    SoC is not blacklisted, the row is simply omitted with the reason.
  * Linux DT `ane_set*` nodes are 4-byte power-controller pwrstate cells
    (t602x cluster at pmgr+0x4000). They are never used as the SET base.
  * macOS MMIO addresses are 32-bit; Linux physicals need the high bits.
    Translation: take a Linux ANE pmgr block reg base whose low-32 equals
    the macOS pmgr_block (same-soc rows first), keep its high 32 bits,
    replace the low 32 bits with the macOS ane/pmgr/SET low bits. The
    low-32 equality is asserted and the source row is recorded.
  * Everything emitted is ANE_RECOGNIZED: constants present, no execution
    ever run. No SET writes, no ane_exec, no kernel probing here.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

SET_OFFSET = 0xC000


def _int(s):
    return int(s, 0) if isinstance(s, str) else s


def _parse_reg(s):
    base, _, size = s.partition("/")
    return _int(base), _int(size)


def _u64le_pairs(hexstr):
    """macOS reg OSData hex: u64-LE (address, size) pairs."""
    b = bytes.fromhex(hexstr)
    out = []
    for i in range(0, len(b) - 15, 16):
        addr = int.from_bytes(b[i:i + 8], "little")
        size = int.from_bytes(b[i + 8:i + 16], "little")
        if addr:
            out.append((addr, size))
    return out


def _irq_bytes(spec):
    """Decode one IOInterruptSpecifiers value to bytes, or None.

    Published rows carry FOUR encodings (community rows 2026-09 and
    earlier predate the hex form):
      * canonical hex string  "0x74030000" (current collector),
      * python bytes repr     "b't\\x03\\x00\\x00'",
      * latin-1 text          the raw control-character string,
      * lossy U+FFFD text     the bytes lost their values -> None.
    """
    if isinstance(spec, bytes):
        return spec
    if not isinstance(spec, str):
        return None
    text = spec.strip()
    if text.startswith("0x"):
        try:
            return bytes.fromhex(text[2:])
        except ValueError:
            return None
    if "\ufffd" in text:
        return None
    if text.startswith(("b'", 'b"')):
        # A python bytes repr: printable bytes stay printable ('t' IS
        # 0x74), so decode with ast, not an escape scan.
        try:
            value = ast.literal_eval(text)
            return value if isinstance(value, bytes) else None
        except (ValueError, SyntaxError):
            return None
    try:
        return text.encode("latin-1")
    except UnicodeEncodeError:
        return None


def _irq_number(spec):
    """IOInterruptSpecifiers (any of the four encodings) -> AIC number."""
    if isinstance(spec, list):
        spec = spec[0] if spec else None
    b = _irq_bytes(spec)
    if not b:
        return None
    return int.from_bytes(b[:4].ljust(4, b"\x00"), "little") or None


def _soc_of_linux(row):
    """SoC id from the Linux chip compatible ("apple,t6020"); macOS chip
    strings ("Apple M2 Max") do not match and return None."""
    m = re.fullmatch(r"apple,t\d+\w*", row.get("chip") or "")
    return m.group(0).split(",")[-1] if m else None


def _macos(row):
    apd = (row.get("summary") or {}).get("ane_port_detail") or {}
    mac = apd.get("macos")
    return mac if isinstance(mac, dict) and mac.get("available") else None


def _devtree(row):
    apd = (row.get("summary") or {}).get("ane_port_detail") or {}
    dt = apd.get("devicetree")
    return dt if isinstance(dt, dict) else None


def _row_sha(row):
    return (row.get("content_sha256") or "")[:12]


class Soc:
    """Merged per-SoC view over one or more rows on each side.

    Field-wise union; a row only fills fields no earlier row supplied, so
    a v0.6.4 deep row (raw reg hex) pairs cleanly with a v0.6.5+ quick row
    (set_base_candidate) for the same machine.
    """

    def __init__(self, soc):
        self.soc = soc
        self.models = set()
        self.macos = {}
        self.macos_shas = []
        self.linux = {}
        self.linux_shas = []

    def add(self, row, sha):
        self.models.add(row.get("model"))
        mac = _macos(row)
        if mac:
            for k, v in mac.items():
                self.macos.setdefault(k, v)
            self.macos_shas.append(sha)
        dt = _devtree(row)
        if dt:
            for k, v in dt.items():
                self.linux.setdefault(k, v)
            self.linux_shas.append(sha)


def _ane_ranges(node, cand):
    """(base, size) ranges of a macOS ANE device node."""
    if node.get("reg_ranges"):
        return [_parse_reg(r) for r in node["reg_ranges"]]
    if node.get("reg"):
        return _u64le_pairs(node["reg"])
    return []


def _range_for(ranges, kinds, want, cand):
    """Range of a given kind; falls back to matching the candidate
    constants when the row predates range_kinds (v0.6.4)."""
    for r, k in zip(ranges, kinds or []):
        if k == want:
            return r
    key = "base" if want == "pmgr_plus_c000" else "pmgr_block"
    expect = _int(cand.get(key))
    for r in ranges:
        if expect is not None and r[0] == expect:
            return r
    return None


def _ane_node_of(mac):
    """Pick THE ane device node: ane0 / die 0 first (community rows
    bugfix: the T6022 row 90c6b9bf3e03 has ane0+ane1 and the old
    first-entry pick chose ane1 and refused)."""
    nodes = [n for n in (mac.get("ane_nodes") or [])
             if isinstance(n, dict) and n.get("name")]

    def rank(node):
        name = str(node["name"]).lower()
        if name == "ane0":
            return (0, name)
        if name == "ane":
            return (1, name)
        m = re.fullmatch(r"ane(\d+)", name)
        if m:
            return (2, int(m.group(1)))
        return (3, name)

    return min(nodes, key=rank) if nodes else None


def derive(socs):
    """Per-SoC emission record or refusal reason."""
    out = {}
    for soc, s in socs.items():
        if not s.macos:
            out[soc] = {"refused": "no macOS deep row for this SoC; the SET "
                                   "candidate only comes from a macOS "
                                   "driver-window capture"}
            continue
        mac = s.macos
        cand = mac.get("set_base_candidate") or {}
        pmgr_low = _int(cand.get("pmgr_block"))
        set_low = _int(cand.get("base"))
        if pmgr_low is None or set_low is None:
            out[soc] = {"refused": "no set_base_candidate in any macOS row; "
                                   "submit a v0.6.5+ macOS deep row"}
            continue
        if not cand.get("driver_window_confirms"):
            out[soc] = {"refused":
                        "set_base omitted: driver_window_confirms=false "
                        "(no pmgr+0xc000 window in the ane device reg); SoC "
                        "not blacklisted - an m1n1 ANE.ps_map probe or a "
                        "macOS row with the +0xc000 window advances it"}
            continue
        if _int(cand.get("offset") or SET_OFFSET) != SET_OFFSET:
            out[soc] = {"refused": f"set_base_candidate offset "
                                   f"{cand.get('offset')} != 0xc000"}
            continue
        node = _ane_node_of(mac)
        ranges = _ane_ranges(node, cand) if node else []
        window = _range_for(ranges, node.get("range_kinds"),
                            "pmgr_plus_c000", cand)
        if window is None or window[0] != set_low:
            out[soc] = {"refused": "macOS ane.reg lacks the pmgr+0xc000 "
                                   "window; set_base omitted"}
            continue
        mmio = _range_for(ranges, node.get("range_kinds"), "ane_mmio", cand)
        if mmio is None:
            out[soc] = {"refused": "macOS ane.reg lacks an ane_mmio range"}
            continue
        # High bits: a Linux ANE pmgr block whose low-32 equals the macOS
        # pmgr_block; same-soc Linux rows first.
        src = None
        for want_soc in (soc, None):
            for lsoc, other in sorted(socs.items()):
                if want_soc and lsoc != want_soc:
                    continue
                for blk in other.linux.get("pmgr_blocks") or []:
                    regs = blk.get("reg") or []
                    if not regs:
                        continue
                    base = _parse_reg(regs[0])[0]
                    if base & 0xFFFFFFFF == pmgr_low:
                        src = (lsoc, other.linux_shas[0], base, blk, other)
                        break
                if src:
                    break
            if src:
                break
        if src is None:
            out[soc] = {"refused": f"no Linux deep row carries a pmgr block "
                                   f"with low-32 {pmgr_low:#x}; high bits "
                                   f"unknown, set_base omitted (a Linux "
                                   f"deep run on any die sharing this "
                                   f"pmgr block sources them)"}
            continue
        lsoc, lsha, lbase, blk, linux_soc = src
        high = lbase & ~0xFFFFFFFF

        def tr(addr):
            return high | (addr & 0xFFFFFFFF)

        darts = []
        for d in mac.get("dart_nodes") or []:
            ranges_d = ([_parse_reg(r) for r in d["reg_ranges"]]
                        if d.get("reg_ranges")
                        else _u64le_pairs(d["reg"]) if d.get("reg") else [])
            if ranges_d:
                darts.append({"name": d.get("name"), "ranges": ranges_d,
                              "irq": _irq_number(d.get("IOInterruptSpecifiers")),
                              "dart_id": d.get("dart_id")})
        out[soc] = {
            "soc": soc,
            "compatible": f"apple,{soc}-ane",
            "ps_base": tr(set_low),
            "ps_source": (f"macOS set_base_candidate {set_low:#x} "
                          f"(pmgr_block {pmgr_low:#x} + 0xc000, "
                          f"driver_window_confirms) translated with high "
                          f"bits of the Linux pmgr block {lbase:#x} from "
                          f"row {lsha} (soc {lsoc}, low-32 match)"),
            "ane_reg": (tr(mmio[0]), mmio[1]),
            "pmgr_phys": tr(pmgr_low),
            "irq": _irq_number(node.get("IOInterruptSpecifiers")),
            "darts": darts,
            "macos_shas": s.macos_shas,
            "linux_shas": s.linux_shas,
            "linux_row_soc": lsoc,
            "linux": linux_soc.linux,
            "linux_blk": blk,
        }
    return out


def _pwrstate_children(blk):
    """(ane_cpu child offset, [(n, offset, name)]) of the ANE pmgr block."""
    cpu, sets = None, []
    for ch in blk.get("children") or []:
        label, name = ch.get("label") or "", ch.get("name") or ""
        m = re.match(r"ane_set(\d+)$", label)
        off = _int("0x" + name.split("@")[-1]) if "@" in name else None
        if m and off is not None:
            sets.append((int(m.group(1)), off, name))
        elif label in ("ane_cpu", "ane_sys_cpu") and off is not None:
            cpu = off
    sets.sort()
    return cpu, sets


def _reg_cells(addr, size):
    return f"<0x{addr >> 32:x} 0x{addr & 0xFFFFFFFF:x} 0x0 0x{size:x}>"


def emit_overlay(r, path):
    """Board overlay: aliases the existing pwrstate/AIC nodes (the proven
    target-path alias shape, no phandle arithmetic) and adds the ANE DARTs
    plus the engine node under /soc."""
    blk = r["linux_blk"]
    cpu, sets = _pwrstate_children(blk)
    if cpu is None or not sets:
        return "Linux row's ANE pmgr block lacks ane_cpu/ane_set* children"
    aic = r["linux"].get("aic") or {}
    if not aic.get("path"):
        return "Linux row lacks the AIC node"
    darts = [d for d in r["darts"] if d["ranges"]]
    if not darts:
        return "macOS row lacks dart-ane reg ranges"
    irq = r["irq"]
    if irq is None:
        return "cannot decode the ane IRQ number from the macOS row"
    pmgr_path = "/" + (blk.get("path") or "")
    if pmgr_path == "/":
        return "Linux row's ANE pmgr block has no node path"
    base, size = r["ane_reg"]
    root_compat = (r["linux"].get("boot") or {}).get("compatible") or []
    board = root_compat[0] if root_compat else f"apple,{r['soc']}"
    dart0 = darts[0]

    fragn = 0

    def alias_frag(label, target):
        nonlocal fragn
        f = (f"\tfragment@{fragn} {{\n"
             f"\t\ttarget-path = \"{target}\";\n"
             f"\t\t{label}: __overlay__ {{\n\t\t}};\n"
             f"\t}};\n\n")
        fragn += 1
        return f

    out = []
    dart_list = ", ".join(f"{b:#x}/{s:#x}" for b, s in dart0["ranges"])
    out.append(
        "// SPDX-License-Identifier: GPL-2.0+ OR MIT\n"
        f"/* Generated by mlx-omarchy scripts/ane_soc_from_collect.py from\n"
        f" * community deep submissions: macOS"
        f" {', '.join(r['macos_shas'])}; Linux"
        f" {', '.join(r['linux_shas'])} (soc {r['linux_row_soc']}).\n"
        f" *\n"
        f" * SET MMIO (driver ps_base) = {r['ps_base']:#x}: the macOS\n"
        f" * driver-window candidate at pmgr+0xc000, translated with the\n"
        f" * high bits of the attested Linux pmgr block. The device tree's\n"
        f" * ane_set* pwrstate cells are 4-byte power-controller cells,\n"
        f" * never the SET MMIO base. RECOGNIZED tier: nothing here has\n"
        f" * executed on silicon; the SET block is mapped read-only.\n"
        f" *\n"
        f" * macOS {dart0['name']} reg ranges (translated where used):\n"
        f" *   {dart_list}\n"
        f" */\n"
        "/dts-v1/;\n"
        "/plugin/;\n\n"
        "/ {\n"
        f"\tcompatible = \"{board}\", \"apple,{r['soc']}\";\n\n")
    # Empty aliased fragments hand phandles to EXISTING nodes: the ANE
    # cpu pwrstate (genpd root), the ane_setN pwrstates (genpd sets), and
    # the AIC (interrupt-parent).
    out.append(alias_frag("ane_cpu_pd",
                          f"{pmgr_path}/power-controller@{cpu:x}"))
    for n, off, cname in sets[:4]:
        out.append(alias_frag(f"ps_ane_set{n}", f"{pmgr_path}/{cname}"))
    out.append(alias_frag("aic_ane", "/" + aic["path"]))
    dart_defs, dart_refs = [], []
    # One dart node per 64K-aligned instance range (the proven T6001
    # shape binds three); extra banks (t602x's +0x4000 range) stay in
    # the header comment.
    instances = [(b, s) for d in darts for b, s in d["ranges"]
                 if b & 0xFFFF == 0][:4]
    dart_irq = next((d["irq"] for d in darts if d["irq"]), None)
    if dart_irq is None:
        return "cannot decode the dart-ane IRQ number from the macOS row"
    # Dart compatible attested by the Linux row's own DART nodes.
    dart_compat = f"apple,{r['soc']}-dart"
    for props in (r["linux"].get("darts") or {}).values():
        for c in props.get("compatible") or []:
            if isinstance(c, str) and \
                    c == f"apple,{r['soc']}-dart":
                dart_compat = c
                break
    for i, (db, ds) in enumerate(instances):
        dphys = (r["ps_base"] & ~0xFFFFFFFF) | db
        dart_defs.append(
            f"\t\t\tane_dart{i}: iommu@{dphys:x} {{\n"
            f"\t\t\t\tcompatible = \"{dart_compat}\";\n"
            f"\t\t\t\treg = {_reg_cells(dphys, ds)};\n"
            f"\t\t\t\t#iommu-cells = <1>;\n"
            f"\t\t\t\tinterrupt-parent = <&aic_ane>;\n"
            f"\t\t\t\tinterrupts = <0 0 {dart_irq} 4>;\n"
            f"\t\t\t\tpower-domains = <&ane_cpu_pd>;\n"
            f"\t\t\t\tstatus = \"okay\";\n"
            f"\t\t\t}};\n")
        dart_refs.append(f"<&ane_dart{i} 0>")
    pd = "<&ane_cpu_pd>, " + ", ".join(f"<&ps_ane_set{n}>" for n, _, _ in sets[:4])
    sep = ", "
    out.append(
        f"\tfragment@{fragn} {{\n"
        f"\t\ttarget-path = \"/soc\";\n"
        f"\t\t__overlay__ {{\n"
        f"\t\t\t#address-cells = <2>;\n"
        f"\t\t\t#size-cells = <2>;\n"
        + "".join(dart_defs) +
        f"\t\t\tane@{base:x} {{\n"
        f"\t\t\t\tcompatible = \"{r['compatible']}\";\n"
        f"\t\t\t\treg-names = \"engine\";\n"
        f"\t\t\t\treg = {_reg_cells(base, size)};\n"
        f"\t\t\t\tinterrupt-parent = <&aic_ane>;\n"
        f"\t\t\t\tinterrupt-names = \"ane\";\n"
        f"\t\t\t\tinterrupts = <0 0 {irq} 4>;\n"
        f"\t\t\t\tiommus = {sep.join(dart_refs)};\n"
        f"\t\t\t\tpower-domains = {pd};\n"
        f"\t\t\t\tstatus = \"okay\";\n"
        f"\t\t\t}};\n"
        f"\t\t}};\n"
        f"\t}};\n"
        "};\n")
    path.write_text("".join(out))
    return None


def emit_c_row(r, path):
    linux_note = (f"; Linux {', '.join(r['linux_shas'])} (soc"
                  f" {r['linux_row_soc']})"
                  if r["linux_shas"] else
                  f"; no same-soc Linux row (high bits from soc"
                  f" {r['linux_row_soc']})")
    path.write_text(
        f"/* Generated by mlx-omarchy scripts/ane_soc_from_collect.py from\n"
        f" * community deep submissions: macOS"
        f" {', '.join(r['macos_shas'])}{linux_note}.\n"
        f" *\n"
        f" * {r['ps_source']}. Engine window"
        f" {r['ane_reg'][0]:#x}/{r['ane_reg'][1]:#x} (translated macOS\n"
        f" * ane_mmio range). RECOGNIZED: constants present from community\n"
        f" * deep collect, execution never run on this silicon; the SET\n"
        f" * block is mapped read-only only.\n"
        f" */\n"
        f"static const struct ane_soc ane_soc_{r['soc']} = {{\n"
        f"\t.ps_base = {r['ps_base']:#x}ULL,\n"
        f"\t.qual = ANE_RECOGNIZED,\n"
        f"}};\n\n"
        f"/* ane_of_match[] entry: */\n"
        f"\t{{ .compatible = \"{r['compatible']}\","
        f" .data = &ane_soc_{r['soc']} }},\n")
    return None


def group_rows(pairs):
    """Group (row, sha) pairs into per-SoC views.

    Rows carrying a SoC id (Linux chip compatible, macOS platform.soc_id)
    group directly; rows too old to carry one (v0.6.4) attach to the
    group whose board model they share, else form their own.
    """
    socs = {}
    deferred = []
    for row, sha in pairs:
        mac = _macos(row)
        soc = (((mac.get("platform") or {}).get("soc_id")) if mac else None) \
            or _soc_of_linux(row)
        if soc:
            socs.setdefault(soc, Soc(soc)).add(row, sha)
        else:
            deferred.append((row, sha, row.get("model")))
    for row, sha, model in deferred:
        match = next((s for s in socs.values()
                      if model and model in s.models), None)
        if match is None:
            match = socs.setdefault(model or "unknown", Soc(model or "unknown"))
        match.add(row, sha)
    return socs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rows", nargs="+",
                    help="published row JSONs (/v1/results/<sha> bodies)")
    ap.add_argument("--outdir", default="ane-soc-out")
    args = ap.parse_args(argv)

    pairs = []
    for p in args.rows:
        with open(p) as fh:
            row = json.load(fh)
        pairs.append((row, _row_sha(row)))
    socs = group_rows(pairs)

    derived = derive(socs)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    emitted = 0
    for soc in sorted(derived):
        r = derived[soc]
        print(f"== {soc} ==")
        if "refused" in r:
            print(f"  REFUSE: {r['refused']}")
            continue
        emitted += 1
        print(f"  ps_base (SET):     {r['ps_base']:#x}")
        print(f"  pmgr (Linux phys): {r['pmgr_phys']:#x}")
        print(f"  ane_reg (engine):  {r['ane_reg'][0]:#x}/{r['ane_reg'][1]:#x}")
        print(f"  source: {r['ps_source']}")
        cpath = outdir / f"{soc}-ane-soc.c"
        emit_c_row(r, cpath)
        print(f"  driver row: {cpath}")
        if r["linux_row_soc"] == soc and r["linux"]:
            opath = outdir / f"{soc}-ane-overlay.dts"
            err = emit_overlay(r, opath)
            print(f"  overlay: {opath}" + (f" REFUSED: {err}" if err else ""))
        else:
            print("  overlay: omitted - needs a Linux deep row on this SoC "
                  "(board topology: pmgr paths, AIC, darts)")
    return 0 if (emitted or any("refused" not in d for d in derived.values())) \
        else 2


if __name__ == "__main__":
    sys.exit(main())
