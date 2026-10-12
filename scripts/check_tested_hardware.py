"""Check every chip in a community dataset snapshot appears in docs/TESTED-HARDWARE.md.

Regression guard for issue #147: apple,t6031 (16-inch M3 Max) landed in the
dataset while the hardware matrix had no row for it.

Usage:
    python3 scripts/check_tested_hardware.py latest.jsonl [--docs docs/TESTED-HARDWARE.md]

Exits 1 and names each chip that is in the dataset but missing from the matrix.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = REPO_ROOT / "docs" / "TESTED-HARDWARE.md"


def dataset_chips(path):
    """Distinct non-empty chip values in a latest.jsonl, sorted."""
    chips = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chip = json.loads(line).get("chip")
            if chip:
                chips.add(chip)
    return sorted(chips)


def matrix_chips(docs_path):
    """Chip names listed in the Apple-silicon collector matrix."""
    chips = set()
    in_apple_section = False
    past_header = False
    for line in Path(docs_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            in_apple_section = line.startswith("## Apple silicon")
            past_header = False
            continue
        if not in_apple_section or not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            past_header = True
            continue
        if not past_header:
            continue
        name = cells[0]
        if name and name != "chip not captured":
            chips.add(name)
    return chips


def missing(dataset, matrix):
    """Sorted chips present in the dataset but absent from the matrix."""
    return sorted(set(dataset) - set(matrix))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="path to latest.jsonl")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS,
                        help="hardware matrix markdown (default: %(default)s)")
    args = parser.parse_args(argv)
    gone = missing(dataset_chips(args.dataset), matrix_chips(args.docs))
    for chip in gone:
        print(f"chip in dataset but missing from {args.docs.name}: {chip}",
              file=sys.stderr)
    if gone:
        return 1
    print(f"ok: every dataset chip appears in {args.docs.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
