#!/usr/bin/env python3
"""Rewrite the benchmark tables in README.md and docs/BENCHMARKS.md from
docs/bench-rows.tsv. Idempotent. --check exits non-zero on drift.

The script only edits the text between marker comments:

  README.md (inside "## How close to macOS"):
    <!-- BEGIN GENERATED: bench-main -->
    ... generated body ...
    <!-- END GENERATED: bench-main -->

  docs/BENCHMARKS.md (blocks under "## Generated scoreboard tables
  (per chip)" at the end of the file; the section and its markers are
  appended on first run):
    <!-- BEGIN GENERATED: bench-chip: g13c -->
    ... generated body ...
    <!-- END GENERATED: bench-chip: g13c -->
    ... (g13g, g14c likewise)

Every other line of either file is left alone. The script never edits
prose; it only fills the marked region.

README.md has no bench-main marker pair until someone adds one, and the script
never inserts into the hand-written table. When a filler-off row exists and the
pair is missing, the script exits non-zero before it writes any file and says
what to do: add the BEGIN and END bench-main pair inside "## How close to macOS"
around the table to be generated, then rerun.

Row policy (review fixes):
  - No ANE / Parakeet encoder rows in these GPU tables.
  - No 13-inch Qwen3.5-9B row: those cells stay in the hand-written
    BENCHMARKS history only.
  - Shares print one decimal (a sub-1% share shows as 0.3%, never 0%).
  - 13-inch Llama-3.1-8B / Mellum2-12B are Linux-only (no macOS share)
    and carry the unexplained-drop flag linking to docs/KNOWN-ISSUES.md.
  - Every table carries Driver and Wheel columns; Runs is the real run
    count from the logs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

CHIPS = [
    ("g13c", "16-inch M1 Max (G13C)"),
    ("g13g", "13-inch M1 (G13G)"),
    ("g14c", "14-inch M2 Max (G14C)"),
]
README_KEY = "bench-main"
CHIP_KEY_PREFIX = "bench-chip:"
BENCH_SECTION = "## Generated scoreboard tables (per chip)"
BEGIN = "<!-- BEGIN GENERATED: {key} -->"
END = "<!-- END GENERATED: {key} -->"

# Public display names for the Mesa Honeykrisp driver and the ANE driver.
DRIVER_DISPLAY = {
    "package": "Honeykrisp (lab package)",
    "omarchy-ane": "omarchy-ane",
}

ANE_METRICS = {"ane-encoder"}
G13G_EXCLUDED_MODELS = {"Qwen3.5-9B"}
G13G_DROP_FLAG_MODELS = {"Llama-3.1-8B", "Mellum2-12B-A2.5B"}
DROP_ANCHOR = (
    "decode-on-the-13-inch-m1-changed-with-the-cpu-governor-with-no-ane-work-running"
)
DROP_FLAG = (
    "decode inside the unexplained 13-inch drop, see "
    f"[KNOWN-ISSUES](KNOWN-ISSUES.md#{DROP_ANCHOR})"
)
# The PR 124 merge wheel: its interleaved control is the staged gather_qmm route on the same
# wheel and the same installed driver, not the v3 ICD.
STAGED_WHEEL = "18059c80"

LEGEND = (
    "_A cell shows Linux tokens per second, or Linux / macOS = share where "
    "both sides produce identical tokens. P is the default direct route. V is "
    "the staged route on the same wheel and driver. P/V = P over V (decode / "
    "prefill). Control spread = spread of the V arm. Governor: the CPU governor, "
    "with the ANE filler off or on._"
)
LEGEND_LINUX_ONLY = LEGEND[:-1] + " No macOS denominator: Linux-only rows._"

NUMERIC_COLUMNS = ("linux", "macos", "start_load_max", "control_spread_pct", "p_spread_pct",
                   "pv_decode", "pv_prefill")


def load_rows(path: Path) -> list[dict]:
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = []
        for n, line in enumerate(f, start=2):
            if not line.strip():
                continue
            cells = line.rstrip("\n").split("\t")
            if len(cells) != len(header):
                raise SystemExit(f"{path}:{n}: {len(cells)} columns, expected {len(header)}")
            row = dict(zip(header, cells))
            for col in NUMERIC_COLUMNS:
                if row.get(col):
                    try:
                        float(row[col])
                    except ValueError:
                        raise SystemExit(f"{path}:{n}: column {col} is not a number: {row[col]}") from None
            rows.append(row)
        return rows


def chip_key_of(row: dict) -> str:
    for key, _label in CHIPS:
        if row["chip"].endswith(f"({key.upper()})"):
            return key
    return ""


def keep_row(row: dict) -> bool:
    if row["metric"] in ANE_METRICS or row.get("noisy") == "1":
        return False
    if chip_key_of(row) == "g13g" and row["model"] in G13G_EXCLUDED_MODELS:
        return False
    return True


def governor_of(row: dict) -> str:
    """Series key: rows of different governor/filler series never share a table cell."""
    g = row.get("governor", "")
    return g if g.startswith("performance") else "schedutil"


def governor_label(rows: list[dict]) -> str:
    for r in rows:
        if r.get("governor", "").startswith("performance"):
            return r["governor"]
    return rows[0].get("governor", "") or "schedutil (not recorded)"


def group_by_chip_model(rows: list[dict]) -> dict:
    g: dict = defaultdict(list)
    for r in rows:
        g[(r["chip"], r["model"], governor_of(r))].append(r)
    return g


def sorted_metrics(rows: list[dict]) -> tuple[dict | None, dict | None]:
    decode = next((r for r in rows if r["metric"] == "decode"), None)
    prefill = next((r for r in rows if r["metric"] == "prefill"), None)
    return decode, prefill


def metric_cell(row: dict | None) -> str:
    if row is None:
        return "-"
    linux = row.get("linux", "")
    macos = row.get("macos", "")
    if linux and macos:
        try:
            share = float(linux) / float(macos) * 100.0
            return f"{linux} / {macos} = {share:.1f}%"
        except ZeroDivisionError:
            return f"{linux} / {macos}"
    return linux or "-"


def flags_for(rows: list[dict], chip_key: str, model: str) -> str:
    parts = []
    loads = [float(r["start_load_max"]) for r in rows if r.get("start_load_max")]
    if loads and max(loads) > 0.5:
        parts.append(f"start load {max(loads):.2f}")
    decode, _prefill = sorted_metrics(rows)
    p_spread = (decode or {}).get("p_spread_pct", "")
    if p_spread:
        parts.append(f"P spread {p_spread}%")
    if chip_key == "g13g" and model in G13G_DROP_FLAG_MODELS:
        parts.append(DROP_FLAG)
    return "; ".join(parts)


def driver_label(rows: list[dict]) -> str:
    r = rows[0]
    name = r.get("driver_name", "")
    commit = r.get("driver_commit", "")
    disp = DRIVER_DISPLAY.get(name, name)
    return f"{disp} {commit}".strip()


def wheel_label(rows: list[dict]) -> str:
    return rows[0].get("wheel_commit", "") or "-"


def runs_label(decode: dict | None, prefill: dict | None) -> str:
    nd = (decode or {}).get("runs", "")
    np_ = (prefill or {}).get("runs", "")
    if not nd or not np_ or nd == np_:
        return f"n={nd or np_}" if (nd or np_) else "-"
    return f"n={nd} / n={np_}"


def control_cell(rows: list[dict]) -> str:
    decode, _prefill = sorted_metrics(rows)
    spread = ""
    for r in sorted(rows, key=lambda r: r.get("metric") != "decode"):
        if r.get("control_spread_pct", ""):
            spread = r["control_spread_pct"]
            break
    head = decode or rows[0]
    wheel = head.get("wheel_commit", "")
    has_pv = any(r.get("pv_decode", "") or r.get("pv_prefill", "") for r in rows)
    if wheel == STAGED_WHEEL:
        return (f"staged gather_qmm route, same wheel; control spread {spread}%"
                if spread else "staged gather_qmm route, same wheel")
    if has_pv:
        if spread:
            return f"v3 ICD, same wheel; control spread {spread}%"
        return "v3 ICD, same wheel"
    return "-"


def pv_cell(rows: list[dict]) -> str:
    decode, prefill = sorted_metrics(rows)
    d = (decode or {}).get("pv_decode", "")
    p = (prefill or {}).get("pv_prefill", "")
    if not d and not p:
        return ""
    fmt = lambda v: f"{float(v):.3f}" if v else "-"
    return f"{fmt(d)} / {fmt(p)}"


def model_row(rows: list[dict], chip_key: str, bench: bool, has_pv: bool = False) -> list[str]:
    decode, prefill = sorted_metrics(rows)
    model = rows[0]["model"]
    row = [
        model,
        metric_cell(decode),
        metric_cell(prefill),
        driver_label(rows),
        wheel_label(rows),
        runs_label(decode, prefill),
        governor_label(rows),
    ]
    if bench:
        if has_pv:
            row.append(pv_cell(rows) or "-")
        row.append(control_cell(rows))
    row.append(flags_for(rows, chip_key, model))
    return row


def render_table(chip_key: str, rows: list[dict], bench: bool,
                 has_pv: bool = False) -> str:
    header = ["Model", "Decode (Linux / macOS = share)" if not bench else "Decode",
              "Prefill", "Driver", "Wheel", "Runs", "Governor"]
    if bench:
        header.append("Control")
    header.append("Flags")
    if bench and has_pv:
        header.insert(-2, "P/V")
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    g = group_by_chip_model(rows)
    order, seen = [], set()
    for r in rows:
        k = (r["chip"], r["model"], governor_of(r))
        if k not in seen:
            seen.add(k)
            order.append(k)

    def series_rank(key: tuple) -> int:
        return 1 if key[2] == "performance (filler on)" else 0

    order.sort(key=lambda k: (k[1], series_rank(k)))
    for chip, model, _gov in order:
        lines.append("| " + " | ".join(
            model_row(g[(chip, model, _gov)], chip_key, bench, has_pv)) + " |")
    return "\n".join(lines) + "\n"


def readme_rows(rows: list[dict]) -> list[dict]:
    # The README table is the 16-inch M1 Max table in the normal user state: filler-on rows go to
    # docs/BENCHMARKS.md only.
    return [r for r in rows if chip_key_of(r) == "g13c" and "filler on" not in r.get("governor", "")]


def render_readme_main(rows: list[dict]) -> str:
    table = render_table("g13c", readme_rows(rows), bench=False)
    footers, seen = [], set()
    for r in readme_rows(rows):
        key = (r.get("driver_name", ""), r.get("driver_commit", ""))
        if key in seen:
            continue
        seen.add(key)
        name, commit = key
        disp = DRIVER_DISPLAY.get(name, name)
        footers.append(f"Driver build: {disp} at Mesa git {commit}")
    body = table
    for f in footers:
        body += "\n" + f + "\n"
    return body


def chip_rows(rows: list[dict], chip_key: str) -> list[dict]:
    return [r for r in rows if chip_key_of(r) == chip_key]


def render_chip(chip_key: str, rows: list[dict]) -> str:
    label = next(label for k, label in CHIPS if k == chip_key)
    body_rows = chip_rows(rows, chip_key)
    has_pv = any(r.get("pv_decode", "") or r.get("pv_prefill", "") for r in body_rows)
    lines = [f"### {label}", ""]
    if chip_key == "g14c":
        lines += ["_preemption kernel option on_", ""]
    if body_rows:
        lines.append(render_table(chip_key, body_rows, bench=True, has_pv=has_pv).rstrip("\n"))
        lines.append("")
        lines.append(LEGEND_LINUX_ONLY if chip_key == "g14c" else LEGEND)
    else:
        lines.append("_no sourced rows for this chip._")
    return "\n".join(lines) + "\n"


def replace_region(text: str, key: str, new_body: str) -> str:
    b = BEGIN.format(key=key)
    e = END.format(key=key)
    if b not in text or e not in text:
        raise SystemExit(
            f"missing markers for {key!r}: add {b} and {e} around the table to generate "
            "(for bench-main, inside '## How close to macOS' in README.md), then rerun. "
            "No file was changed.")
    i = text.index(b) + len(b)
    j = text.index(e, i)
    return text[:i] + "\n" + new_body + text[j:]


def append_missing_chip_blocks(text: str, bodies: dict) -> str:
    """First-run path for docs/BENCHMARKS.md: append the generated
    section (with the title if absent) and any missing chip blocks."""
    missing = [k for k in bodies if BEGIN.format(key=f"{CHIP_KEY_PREFIX}{k}") not in text]
    if not missing:
        return text
    out = text.rstrip("\n") + "\n\n"
    if BENCH_SECTION not in text:
        out += BENCH_SECTION + "\n\n"
    for k in missing:
        out += BEGIN.format(key=f"{CHIP_KEY_PREFIX}{k}") + "\n"
        out += bodies[k]
        out += END.format(key=f"{CHIP_KEY_PREFIX}{k}") + "\n\n"
    return out.rstrip("\n") + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_tsv", nargs="?", default="docs/bench-rows.tsv")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--benchmarks", default="docs/BENCHMARKS.md")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if either file would change")
    args = ap.parse_args()

    rows = [r for r in load_rows(Path(args.bench_tsv)) if keep_row(r)]
    bp = Path(args.benchmarks)
    if not bp.exists():
        raise SystemExit(f"{args.benchmarks} does not exist")
    btext = bp.read_text()
    # A chip with rows, or one that already has a generated block, is rendered: a block whose rows
    # were all removed is rewritten to the placeholder instead of keeping stale data.
    chip_bodies = {
        k: render_chip(k, rows) for k, _label in CHIPS
        if chip_rows(rows, k) or BEGIN.format(key=f"{CHIP_KEY_PREFIX}{k}") in btext
    }

    pending = []
    if readme_rows(rows):
        rp = Path(args.readme)
        if not rp.exists():
            raise SystemExit(f"{args.readme} does not exist")
        rtext = rp.read_text()
        rnew = replace_region(rtext, README_KEY, render_readme_main(rows))
        if rnew != rtext:
            pending.append((rp, rnew))

    bnew = btext
    for chip_key in chip_bodies:
        region = f"{CHIP_KEY_PREFIX}{chip_key}"
        if BEGIN.format(key=region) in bnew:
            bnew = replace_region(bnew, region, chip_bodies[chip_key])
    bnew = append_missing_chip_blocks(bnew, chip_bodies)
    if bnew != btext:
        pending.append((bp, bnew))

    for path, text in pending:
        if args.check:
            print(f"DRIFT: {path}")
        else:
            path.write_text(text)
            print(f"updated {path}")
    return 1 if args.check and pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
