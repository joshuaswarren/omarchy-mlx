#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Differential check: the C++ H16 section builder against h16-sections.py.

Feeds the add-128 fixture, 4000 mutations of it and the synthetic variants
the C++ driver emits to both the kernel tree's reference client and the C++
builder, then requires identical section bytes, task counts, BAR references
and buffer IDs for every accepted file and identical refusal text for every
refused one. Where the reference stops with an unnamed Python error
(struct.error, UnicodeDecodeError) the builder must refuse with a named one.

  h16_sections_diff.py --reference h16-sections.py --driver ./h16_sections_diff \
      --work DIR
"""

import argparse
import importlib.util
import pathlib
import random
import struct
import subprocess
import sys

FIXTURE = pathlib.Path(__file__).with_name("fixtures") / "h16-add128" / "model.hwx"


def mutations(base, text_off, text_size, count, seed=20261102):
    rng = random.Random(seed)
    for n in range(count):
        b = bytearray(base)
        kind = n % 4
        if kind == 0:
            for _ in range(rng.randint(1, 3)):
                b[rng.randrange(0, 0x400)] = rng.randrange(256)
        elif kind == 1:
            for _ in range(rng.randint(1, 3)):
                b[text_off + rng.randrange(0, text_size)] = rng.randrange(256)
        elif kind == 2:
            b = b[:rng.randrange(0, len(b))]
        else:
            o = rng.randrange(0, 0x400) & ~3
            struct.pack_into("<I", b, o, rng.choice(
                [0, 1, 2, 3, 4, 8, 0x19, 0x40, 0xffffffff, rng.randrange(1 << 32)]))
        yield bytes(b)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--driver", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path, required=True)
    parser.add_argument("--mutations", type=int, default=4000)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("h16_sections", args.reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    base = FIXTURE.read_bytes()
    _, text, _, _, _ = ref.parse(base)
    args.work.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, data in enumerate([base, *mutations(base, text[2], text[1], args.mutations)]):
        path = args.work / f"m{i}.hwx"
        path.write_bytes(data)
        paths.append(path)
    synth = args.work / "synthetic"
    synth.mkdir(exist_ok=True)
    subprocess.run([str(args.driver), "emit", str(synth)], check=True, stdout=subprocess.DEVNULL)
    paths += sorted(synth.glob("*.hwx"), key=lambda p: int(p.stem[1:]))

    expect = []
    for path in paths:
        try:
            sections, ids, refs, tasks = ref.build(path.read_bytes())
            line = ("OK " + " ".join(s.hex() for s in sections) + f" tasks={tasks} refs="
                    + "".join(f"{s}:{t}," for s, t in refs))
            expect.append(("OK", line, ids))
        except ref.HwxError as error:
            expect.append(("ERR", f"ERR {error}", None))
        except Exception as error:  # the reference's unnamed refusals
            expect.append(("PYERR", type(error).__name__, None))
    got = subprocess.run([str(args.driver)], input=("\n".join(map(str, paths)) + "\n").encode(),
                         capture_output=True, check=True).stdout.decode(
                             "utf-8", "surrogateescape").splitlines()
    stats = {"ok_match": 0, "err_match": 0, "unnamed_vs_named": 0, "mismatch": 0}
    for path, (kind, line, ids), out in zip(paths, expect, got):
        if kind == "OK":
            head, _, tail = out.partition(" ids=")
            got_ids = {}
            for item in filter(None, tail.rstrip(",").split(",")):
                key, value = item.rsplit("=", 1)
                got_ids[key] = int(value)
            ok = head == line and got_ids == ids
        elif kind == "ERR":
            ok = out == line
        else:
            ok = out.startswith("ERR ")
        key = {"OK": "ok_match", "ERR": "err_match", "PYERR": "unnamed_vs_named"}[kind]
        stats[key if ok else "mismatch"] += 1
        if not ok:
            print(f"MISMATCH {path.name}: {line[:100]} | {out[:100]}")
    print(f"{len(paths)} files: {stats}")
    return 1 if stats["mismatch"] or len(got) != len(paths) else 0


if __name__ == "__main__":
    sys.exit(main())
