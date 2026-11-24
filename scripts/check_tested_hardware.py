"""Check every chip in a community dataset snapshot appears in docs/TESTED-HARDWARE.md.

Regression guard for issue #147: apple,t6031 (16-inch M3 Max) landed in the
dataset while the hardware matrix had no row for it.

Usage:
    python3 scripts/check_tested_hardware.py latest.jsonl [--docs docs/TESTED-HARDWARE.md]

Exits 1 and names each chip that is in the dataset but missing from the matrix.
Exits 1 with an error line when either file cannot be read or parsed.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = REPO_ROOT / "docs" / "TESTED-HARDWARE.md"


class DatasetError(ValueError):
    """A dataset or matrix file this check cannot read."""


def dataset_chips(path):
    """Distinct non-empty chip values in a latest.jsonl, sorted.

    A line that is not UTF-8, not a JSON object, or has a chip that is not a
    string raises DatasetError naming the line: a corrupt snapshot fails the
    check rather than passing it.
    """
    chips = set()
    # surrogateescape keeps an undecodable byte on its own line, so the
    # error can name that line; strict UTF-8 never yields lone surrogates.
    with open(path, encoding="utf-8", errors="surrogateescape") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            where = f"{path}:{number}"
            try:
                line.encode("utf-8")
            except UnicodeEncodeError:
                raise DatasetError(f"{where}: not UTF-8 text") from None
            try:
                record = json.loads(line)
            except (ValueError, RecursionError) as exc:
                raise DatasetError(f"{where}: not a JSON line: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetError(
                    f"{where}: record is {type(record).__name__}, not an object")
            chip = record.get("chip")
            if chip is None:
                continue
            if not isinstance(chip, str):
                raise DatasetError(
                    f"{where}: chip is {type(chip).__name__}, not a string")
            chip = chip.strip()
            if chip:
                chips.add(chip)
    return sorted(chips)


def matrix_chips(docs_path):
    """Chip names listed in the Apple-silicon collector matrix."""
    chips = set()
    in_apple_section = False
    past_header = False
    try:
        text = Path(docs_path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{docs_path}: not UTF-8 text: {exc}") from exc
    for line in text.splitlines():
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
    try:
        gone = missing(dataset_chips(args.dataset), matrix_chips(args.docs))
    except (OSError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for chip in gone:
        print(f"chip in dataset but missing from {args.docs.name}: {chip}",
              file=sys.stderr)
    if gone:
        return 1
    print(f"ok: every dataset chip appears in {args.docs.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
