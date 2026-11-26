#!/usr/bin/env python3
"""Classify junit failure messages from the py phase of run-upstream-suite.sh.

Buckets mirror receipts/2026-09-01-upstream-suite-coverage.md:
  named    RuntimeError: [omarchy] ... is not implemented ...
           RuntimeError: [omarchy] ... is refused ...
  cpucpu   IndexError: vector::_M_range_check (cpu stream table)
  ncpuimpl RuntimeError: ... has no CPU implementation
  assert   AssertionError (wrong-value candidates; each verified by hand)
  other    anything else
Usage: tools/analyze-py-suite.py <py-xml-dir> [--csv OUT.csv]
"""
import collections
import glob
import os.path
import re
import sys
import xml.etree.ElementTree as ET

NAMED = re.compile(r"RuntimeError: \[omarchy\] (.+?) is (?:not implemented|refused)")
FIRST_LINE = lambda s: (s or "").strip().split("\n")[0]


def kind_of(msg):
    first = FIRST_LINE(msg)
    m = NAMED.match(first)
    if m:
        return "named", m.group(1)
    if "vector::_M_range_check" in first:
        return "cpucpu", "IndexError cpu stream table"
    if "has no CPU implementation" in first:
        return "ncpuimpl", FIRST_LINE(first.replace("RuntimeError: ", ""))[:90]
    if first.startswith("AssertionError") or first.startswith("AssertionError:"):
        return "assert", ""
    if first.startswith("UnboundLocalError"):
        return "other", "UnboundLocalError custom_kernel (upstream harness artifact)"
    return "other", first[:90]


def main():
    if len(sys.argv) < 2:
        print("ERROR: report directory argument is required", file=sys.stderr)
        return 2
    xml_dir = sys.argv[1]
    csv_path = None
    if "--csv" in sys.argv:
        index = sys.argv.index("--csv")
        if index + 1 == len(sys.argv):
            print("ERROR: --csv requires an output path", file=sys.stderr)
            return 2
        csv_path = sys.argv[index + 1]

    if not os.path.isdir(xml_dir):
        print(f"ERROR: report directory not found: {xml_dir}", file=sys.stderr)
        return 2
    xml_files = sorted(glob.glob(f"{xml_dir}/*.xml"))
    if not xml_files:
        print(f"ERROR: no XML reports in directory: {xml_dir}", file=sys.stderr)
        return 2

    kinds = collections.Counter()
    named = collections.Counter()
    asserts = []
    rows = []
    for xf in xml_files:
        stem = os.path.basename(xf)[:-4]
        try:
            tree = ET.parse(xf)
        except ET.ParseError as e:
            print(f"ERROR: malformed XML report {xf}: {e}", file=sys.stderr)
            return 2
        except OSError as e:
            print(f"ERROR: cannot read XML report {xf}: {e}", file=sys.stderr)
            return 2
        cases = tree.findall(".//testcase")
        if not any(tc.find("skipped") is None for tc in cases):
            # Fully-skipped files (e.g. conv_transpose on this backend)
            # are legitimate: nothing executed, nothing to classify.
            print(f"WARNING: no executed test cases (all skipped): {xf}",
                  file=sys.stderr)
            continue
        for tc in cases:
            for fail in list(tc.findall("failure")) + list(tc.findall("error")):
                msg = fail.get("message") or (fail.text or "")
                k, detail = kind_of(msg)
                kinds[k] += 1
                if k == "named":
                    named[detail] += 1
                elif k == "assert":
                    asserts.append((stem, tc.get("name"), FIRST_LINE(msg)[:160]))
                rows.append((stem, tc.get("name"), k, detail))
    if csv_path:
        with open(csv_path, "w") as out:
            out.write("file,case,kind,detail\n")
            for r in rows:
                out.write(",".join(x.replace(",", ";") for x in r) + "\n")
    print("failure kinds:")
    for k, n in kinds.most_common():
        print(f"  {k:10s} {n}")
    print("\nnamed-gap histogram:")
    for name, n in named.most_common():
        print(f"  {n:6d}  {name}")
    print(f"\nAssertionError cases ({len(asserts)}):")
    for stem, case, msg in asserts:
        print(f"  {stem}::{case}\n      {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
