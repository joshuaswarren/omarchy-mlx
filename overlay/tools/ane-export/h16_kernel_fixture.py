#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Regenerate the H16G add-128 fixture from an aurora kernel tree.

The kernel's KUnit header drivers/accel/ane/ane_h16_add128_fixture.h
(SPDX GPL-2.0-only OR MIT) stores the pinned add-128 HWX (49152 bytes,
SHA-256 231f54c2...) and the four H16_LOAD sections that the kernel's
tools/testing/selftests/drivers/ane/h16-sections.py builds for it, as runs
of nonzero bytes. This writes them as plain files and refuses a header whose
HWX does not hash to the pinned digest.

  h16_kernel_fixture.py --header ane_h16_add128_fixture.h --out-dir DIR
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys

HWX_SHA256 = "231f54c28ea6b14a2867fac78eac1a2cb22aa3b0eeaa28113ed03c520b83a1d3"
ARRAYS = (
    ("hwx", "ADD128_HWX_SIZE", "model.hwx"),
    ("generic", "ADD128_GENERIC_SIZE", "generic.bin"),
    ("operation", "ADD128_OPERATION_SIZE", "operation.bin"),
    ("procedure", "ADD128_PROCEDURE_SIZE", "procedure.bin"),
    ("properties", "ADD128_PROPERTIES_SIZE", "text-properties.bin"),
)
RUN = re.compile(r"\{\s*(0x[0-9a-fA-F]+|\d+),\s*(\d+),\s*\(const u8\[\]\)\{([^}]*)\}\s*\}")


def array_body(text, name):
    start = text.index("add128_%s_runs[] = {" % name)
    end = text.index("\n};", start)
    return text[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--header", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    text = args.header.read_text()
    outputs = {}
    for name, size_macro, filename in ARRAYS:
        size = int(re.search(r"#define %s (0x[0-9a-fA-F]+|\d+)" % size_macro, text).group(1), 0)
        data = bytearray(size)
        for offset, length, body in RUN.findall(array_body(text, name)):
            values = bytes(int(v, 16) for v in re.findall(r"0x[0-9a-fA-F]{2}", body))
            offset = int(offset, 0)
            if len(values) != int(length) or offset + len(values) > size:
                print(f"FAIL: malformed run in {name}")
                return 1
            data[offset:offset + len(values)] = values
        outputs[filename] = bytes(data)
    digest = hashlib.sha256(outputs["model.hwx"]).hexdigest()
    if digest != HWX_SHA256:
        print(f"FAIL: HWX SHA-256 {digest} is not the pinned add-128 fixture")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in outputs.items():
        (args.out_dir / filename).write_bytes(data)
    print(json.dumps({name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for name, data in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
