#!/usr/bin/env python3
# Classify doctest XML failure reports from tools/run-upstream-suite.sh
# (phase 1, C++) into the coverage buckets used by
# receipts/2026-09-01-upstream-suite-coverage.md:
#   a  genuine omarchy gap, named [omarchy] error
#      (flagged "masked" when every failed assert is the test-suite's own
#       array_equal machinery hitting the bool And/Equal gap)
#   b  CPU-backend-absence artifact (explicit Device::cpu use in the test)
#   d  wrong values without a named error
#
# Usage: tools/analyze-upstream-suite.py <cpp-xml-dir> [--csv OUT.csv]
import collections
import glob
import os.path
import re
import sys
import xml.etree.ElementTree as ET

MACHINERY = re.compile(r'^"\[omarchy\] (And|Equal) dtype is not implemented.*dtype=bool')
RANGE_CHECK = "vector::_M_range_check"


def classify_failure(first_exc):
    if first_exc and RANGE_CHECK in first_exc:
        return "b"
    return "a"


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

    metadata = ("flags().", "data_size()", "siblings()")
    rows = []
    executed_total = 0
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
        cases = [c for c in tree.findall(".//TestCase")
                 if c.get("skipped") != "true"]
        if not cases:
            print(f"ERROR: XML report has no executed test cases: {xf}",
                  file=sys.stderr)
            return 2
        executed_total += len(cases)
        for c in cases:
            a = c.find("OverallResultsAsserts")
            if a is None or a.get("test_case_success") not in ("true", "false"):
                print(f"ERROR: incomplete test case in {xf}", file=sys.stderr)
                return 2
            if a.get("test_case_success") == "true":
                continue
            name = c.get("name")
            excs, noexc = [], []
            for e in c.findall(".//Expression[@success='false']"):
                exc = e.find("Exception")
                if exc is not None:
                    excs.append((exc.text or "").strip().split("\n")[0])
                else:
                    orig = (e.findtext("Original") or "").strip()
                    exp = (e.findtext("Expanded") or "").strip()
                    noexc.append((e.get("line"), orig, exp))
            case_level = [(x.text or "").strip() for x in c.findall("Exception")]
            excs += case_level
            first_exc = excs[0] if excs else ""
            cat = classify_failure(first_exc)
            if any(not any(p in o for p in metadata) for _, o, _ in noexc):
                cat = "d"
            masked = (
                cat == "a"
                and bool(excs)
                and all(MACHINERY.match(x) for x in excs if x.startswith('"'))
                and not case_level
            )
            rows.append((stem, name, cat, "masked" if masked else "",
                         first_exc[:130]))

    out = open(csv_path, "w") if csv_path else sys.stdout
    if csv_path:
        out.write("file,case,category,flag,first_error\n")
    for r in rows:
        line = f'{r[0]},{r[1]},{r[2]},{r[3]},"{r[4]}"\n'
        if csv_path:
            out.write(line)
        else:
            print(line.replace('"', ""))
    if csv_path:
        out.close()

    per = collections.defaultdict(collections.Counter)
    for r in rows:
        per[r[0]][r[2]] += 1
    print("per-file failing-case categories:", file=sys.stderr)
    for f in sorted(per):
        c = per[f]
        print(f"  {f:26s} a={c['a']:3d} b={c['b']:2d} d={c['d']:2d}",
              file=sys.stderr)
    tot = collections.Counter(r[2] for r in rows)
    masked_n = sum(1 for r in rows if r[3] == "masked")
    print(f"TOTAL executed cases: {executed_total}; failing cases: {len(rows)}; "
          + " ".join(f"{k}={tot[k]}" for k in "abd")
          + f"; a-masked={masked_n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
