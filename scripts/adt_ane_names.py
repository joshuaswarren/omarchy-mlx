#!/usr/bin/env python3
"""Print the ANE nodes of the boot loader's copy of the Apple device tree (ADT), read-only.

aurora's `aurora-adt-extract.py` (the reader the M3/M4 kit runs) prints an allowlist that has no
ANE node, so its output cannot say whether the ADT holds one. This wrapper loads that same reader,
checks it against the pinned SHA-256, widens its allowlist to the ANE nodes only, and also lists
the NAMES (nothing else) of the children of /arm-io. Everything the reader drops stays dropped:
serial numbers, MAC addresses, ECIDs, keys and any value that holds one.

  python3 adt_ane_names.py --reader aurora-adt-extract.py --check /dev/mtd1ro   # reads nothing
  sudo python3 adt_ane_names.py --reader aurora-adt-extract.py /dev/mtd1ro      # prints

Get the reader from the aurora release the kit names and check its hash first:
  curl -fsSLO https://github.com/iconidentify/aurora-linux/releases/download/aurora-2026.10.09.8/aurora-adt-extract.py
  sha256sum aurora-adt-extract.py   # 3563a3bb7ff832bf94401a9ec751afa0424156ac8a1b160813cdfa96d592a72b
PATH is /dev/mtdNro (the read-only node of the MTD device named adt) or a regular file holding an ADT.
"""

import argparse
import hashlib
import importlib.util
import os
import sys

READER_SHA256 = "3563a3bb7ff832bf94401a9ec751afa0424156ac8a1b160813cdfa96d592a72b"
ANE_SUBTREES = ("/arm-io/ane*", "/arm-io/dart-ane*", "/arm-io/mapper-ane*")


def load_reader(path):
    with open(path, "rb") as handle:
        got = hashlib.sha256(handle.read()).hexdigest()
    if got != READER_SHA256:
        raise SystemExit(f"adt_ane_names: {path} has sha256 {got}, not the pinned {READER_SHA256}")
    spec = importlib.util.spec_from_file_location("aurora_adt_extract", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"adt_ane_names: cannot load {path} as a python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(reader, data, source):
    """The ANE allowlist report plus the names of the /arm-io children, as one string."""
    reader.ALLOWED_SUBTREES = ANE_SUBTREES
    root, used, nodes, props = reader.parse_adt(data)
    text = reader.render(root, source, used, nodes, props)
    arm_io = [child for child in root.children if child.name == "arm-io"]
    names = sorted(child.name or "?" for node in arm_io for child in node.children)
    ane_found = any(name.startswith("ane") for name in names)
    return (text + f"# /arm-io children (names only): {len(names)}\n" + "\n".join(f"#   {n}" for n in names)
            + f"\n# ane node in /arm-io: {'yes' if ane_found else 'no'}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="adt_ane_names")
    parser.add_argument("--reader", required=True, help="path to aurora-adt-extract.py")
    parser.add_argument("--check", action="store_true", help="check the device, read nothing")
    parser.add_argument("path", nargs="?", help="/dev/mtdNro or a regular file holding an ADT")
    args = parser.parse_args(argv)
    reader = load_reader(args.reader)
    host = reader.Host()
    if args.path is not None and reader._is_regular_file(args.path, host):
        if args.check:
            print(f"{args.path}: a regular file, no device to check")
            return 0
        data = reader.read_file(os.path.realpath(args.path))
        source = f"file {os.path.basename(args.path)} ({len(data)} bytes)"
    else:
        binding = reader.check_binding(args.path, host)
        if args.check:
            print(binding.describe())
            return 0
        data = reader.read_device(binding, host)
        source = binding.describe()
    sys.stdout.write(report(reader, data, source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
