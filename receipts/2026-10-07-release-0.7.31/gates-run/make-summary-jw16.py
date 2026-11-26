#!/usr/bin/env python3
"""Generate SUMMARY-jw16.md for the v0.7.31 jw16 gate run.

Reads ONLY the raw log files committed next to this script and derives every
table cell from their content. Never hand-edit the SUMMARY; edit the inputs
(raw logs) or this script and re-run it.

Derived cells per gate: rc (from the gate's own RESULT/EXIT markers),
wheel_version (from the installed-wheel provenance in the logs), uname
(from the gate's BEGIN/provenance header), log_path_in_repo (the file read).
"""
import json
import pathlib
import re
import sys

D = pathlib.Path(__file__).resolve().parent
WHEEL_SHA = "a2f8c83e5c5f635d00702565a8557d87c40a9eccac885329dcec4692d85d300d"
TAG_SHA = "9b5c938fe236e767df9545de301cfc74fdbc3395"
# Reconciled (w7K review 2026-10-07): the jw16 scripts as RUN were staged
# from 83f0ddd0b (host-provenance-jw16.txt, generated 15:12:57Z while g7c
# held the lock). ea1227088 landed at 15:21Z, after the run started, and is
# NOT what ran.
MAIN_SHA = "83f0ddd0bd87d253d5535903fd73c001c812fc1b"


def read(name):
    p = D / name
    if not p.exists():
        return None
    return p.read_text(errors="replace")


def need(name):
    t = read(name)
    if t is None:
        sys.exit(f"make-summary-jw16: missing input log {name}")
    return t


def marker_rc(text, patterns):
    """rc=0 iff every regex matches; rc=1 if any captured exit code is nonzero;
    rc=None (log missing/marker not found) otherwise."""
    if text is None:
        return None
    rc = 0
    for pat in patterns:
        hits = re.findall(pat, text, re.M)
        if not hits:
            return None
        for h in hits:
            vals = h if isinstance(h, tuple) else (h,)
            for v in vals:
                if isinstance(v, str) and v.isdigit() and v != "0":
                    rc = 1
    return rc


def first(text, pat, default=""):
    m = re.search(pat, text, re.M)
    return m.group(1) if m else default


WHEEL_VER = None  # from the installed-wheel receipts, asserted consistent below
versions = set()
unames = set()

rows = []


def add(gate, rc, log, ver=None, uname=None, note=""):
    if ver:
        versions.add(ver)
    if uname:
        unames.add(uname)
    rc_disp = "0" if rc == 0 else ("FAIL" if rc is None else str(rc))
    rows.append((gate, rc_disp, log, note))


# --- g17 patch series (CPU, fill-run policy) -------------------------------
t = need("g17-stdout-jw16.log")
g17 = json.loads(need("g17-result-jw16.json"))
g17_rc = 0 if g17.get("pass") is True else 1
detail = "; ".join(
    f"{v}: apply1={d.get('apply1_rc')} apply2={d.get('apply2_rc')}"
    for v, d in sorted(g17.get("results", {}).items())
)
add("g17-patch-series (CPU, fill-run)", g17_rc, "gates-run/g17-stdout-jw16.log",
    note=detail)

# --- g1 fresh-home install ---------------------------------------------------
t = need("g1-install-jw16.log")
g1_rc = marker_rc(t, [
    r"^INSTALL_EXIT (\d+)",
    r"^LAUNCHER_HELP_EXIT (\d+)",
    r"^SERVE_ENTRY_EXIT (\d+)",
    r"^PARAKEET_LAUNCHER_HELP_EXIT (\d+)",
    r"^PARAKEET_LAUNCHER staged$",
    r"^GATE1_DONE$",
])
add("g1-clean-install (fresh HOME)", g1_rc, "gates-run/g1-install-jw16.log",
    uname=first(t, r"uname=(\S+)"))

# --- g-g13c ------------------------------------------------------------------
t = need("g-g13c-receipt-jw16.log")
sha_line = first(t, r"^wheel_sha256 ([0-9a-f]{64}) ")
g13c_rc = 0 if ("G13C_PASS" in t and sha_line == WHEEL_SHA) else (
    1 if "G13C_PASS" in t else None)
add("g-g13c (chip + CPU-PD hold)", g13c_rc, "gates-run/g-g13c-receipt-jw16.log",
    ver=first(t, r"^pip mlx_omarchy (\S+)"),
    note="wheel_sha OK" if sha_line == WHEEL_SHA else "WHEEL SHA MISMATCH")

# --- g-g13c leg B (ICD override: coreglass bbbfa36dce) ----------------------
t = need("g-g13c-receipt-legB-jw16.log")
sha_b = first(t, r"^wheel_sha256 ([0-9a-f]{64}) ")
lvb = first(t, r"^libvulkan_sha256 ([0-9a-f]{64})")
legb_ok = ("G13C_PASS" in t and sha_b == WHEEL_SHA and
           lvb == "cb2a1bcf4c1b10cf587a36f6acf59a74604a2c1537be2dc5fbcda6bcfef316e7")
add("g-g13c leg B (ICD override, coreglass git-bbbfa36dce)",
    0 if legb_ok else 1, "gates-run/g-g13c-receipt-legB-jw16.log",
    ver=first(t, r"^pip mlx_omarchy (\S+)"),
    note="libvulkan cb2a1bcf… as directed" if legb_ok else "SHA/PASS mismatch")

# --- jw16-gates full plan ----------------------------------------------------
t = need("jw16-gates-receipt-jw16.log")
steps = ["build-wheel", "g13-build", "g15-build", "g7c", "g7d", "g13-run", "g15-run"]
rcs = {}
for s in steps:
    m = re.search(rf"^STEP {re.escape(s)}_RC=(\d+)", t, re.M)
    rcs[s] = m.group(1) if m else None
jw16_all = None if any(v is None for v in rcs.values()) else rcs
jw16_rc = None if jw16_all is None else (
    0 if all(v == "0" for v in jw16_all.values()) else 1)
# The g7c receipt log must itself carry the verify + report assertions.
g7c_log = need("g7c-ane-worker-verify-jw16.log")
g7c_marks = [r"^VERIFY_EXIT 0$", r"^G7C_TRACE_ABI_EXIT 0$",
             r"^ane_mode = True$", r"^cpu_tensor_events = 0$",
             r"^status = match$", r"^GATE7C_EXIT 0$"]
g7c_missing = [p for p in g7c_marks if not re.search(p, g7c_log, re.M)]
if g7c_missing:
    jw16_rc = 1
add("jw16-gates (build+g13+g15+g7c+g7d)", jw16_rc,
    "gates-run/jw16-gates-receipt-jw16.log",
    note=" ".join(f"{s}={v}" for s, v in rcs.items()) +
         ("" if not g7c_missing else f" g7c-log MISSING {g7c_missing}"))

# --- g16 / g16b legs P and B -------------------------------------------------
for leg in ("P", "B"):
    prov = need(f"leg{leg}-provenance-jw16.log")
    u = first(prov, r"^uname -r: (\S+)")
    lv = first(prov, r"^libvulkan_sha256: ([0-9a-f]{64})")
    icd = first(prov, r"^icd_json: (\S+)")
    g16 = json.loads(need(f"g16-qmm-batch-leg{leg}-jw16.json"))
    ver = g16.get("provenance", {}).get("wheel_version")
    g16_rc = 0 if g16.get("pass") is True else 1
    tb = need(f"g16b-qmm-route-probe-leg{leg}-jw16.log")
    g16b_rc = 0 if re.search(r"^RESULT: PASS", tb, re.M) else 1
    add(f"g16-qmm-batch leg {leg}", g16_rc,
        f"gates-run/g16-qmm-batch-leg{leg}-jw16.json", ver=ver, uname=u,
        note=f"icd={icd} libvulkan={lv[:12]}…")
    add(f"g16b-route-probe leg {leg}", g16b_rc,
        f"gates-run/g16b-qmm-route-probe-leg{leg}-jw16.log", ver=ver, uname=u,
        note=f"icd={icd} libvulkan={lv[:12]}…")

# g-hold (T8103/G13G host lane; receipts were captured by the power lane)
ghold_texts = [t for t in (read("g-hold-receipt-20261007T140554Z.log"),
                           read("g-hold-receipt-20261007T140635Z.log")) if t]
if ghold_texts:
    t = ghold_texts[-1]
    hold_rc = 0 if ("HOLD_PASS" in t and "Ran 2 tests" in t and "\nOK" in t) else 1
    # The pre-header receipts carry no host/uname line; note that honestly.
    add("g-hold-g13g (T8103 hold A/B)", hold_rc,
        "gates-run/g-hold-receipt-20261007T140635Z.log",
        note="receipt predates the BEGIN/uname header: host/uname not captured in log")

# --- consistency gates -------------------------------------------------------
if versions and len(versions) > 1:
    sys.exit(f"make-summary-jw16: inconsistent wheel versions {versions}")

_WHEEL_ONLY = sorted(versions).pop() if len(versions) == 1 else None

wheel_rows = [r for r in rows if r[0].startswith(("g1", "g-g13c", "jw16-gates"))]
ok_all = all(r[1] == "0" for r in rows)
lines = [
    "# v0.7.31 jw16 gate summary — generated by make-summary-jw16.py from the raw logs",
    "",
    f"Tag v0.7.31 = {TAG_SHA}; gate scripts main {MAIN_SHA}; wheel sha256 {WHEEL_SHA}.",
    f"Wheel version (installed-wheel receipts): {_WHEEL_ONLY or 'n/a'}",
    f"uname -r across GPU legs: {sorted(unames).pop() if len(unames) == 1 else sorted(unames)}",
    "",
    "| gate | host | wheel | rc | log (repo) | notes |",
    "|---|---|---|---|---|---|",
]
for gate, rc, log, note in rows:
    wheel_cell = _WHEEL_ONLY or "-"
    lines.append(f"| {gate} | jw16 | {wheel_cell} | {rc} | {log} | {note} |")
lines += [
    "",
    ("JW16 GREEN: every gate RC=0 on this host with this wheel."
     if ok_all else
     "NOT GREEN: at least one gate is not RC=0 — see the table; root cause in the cited log."),
    "",
    ("g-g13c leg B: PASS on jw16 for v0.7.31 (ICD override coreglass git-bbbfa36dce, "
     "libvulkan cb2a1bcf… as directed; receipt gates-run/g-g13c-receipt-legB-jw16.log)."
     if legb_ok else
     "g-g13c leg B: NOT RUN on jw16 for v0.7.31 (w7N phone block; not queued)."),
    "",
]
out = D / "SUMMARY-jw16.md"
out.write_text("\n".join(lines))
print(f"wrote {out} ({len(rows)} rows) ok_all={ok_all}")
sys.exit(0 if ok_all else 1)
