#!/usr/bin/env python3
"""Compare Exl3Decode GPU stage dumps against the CPU reference stages.

Pairing: manifest.txt carries one "call=..." line per GPU dispatch (written
by Exl3Decode::eval_gpu under MLX_OMARCHY_EXL3_DEBUG_DUMP) and one "ref=..."
line per CPU reference (written by the doctest). Entries pair by
(in, out, bits, window, dtype), zipped in order within each tuple group.

Stages: dec16 (uint32 words, one zero-extended f16 bit pattern each,
block-local row-major), tmp (f32, row H128 + suh), out (final output in the
call's dtype; the reference is f32, RNE-rounded to the dtype before compare).

Usage: exl3_dump_compare.py <dump-dir> [--stage dec16|tmp|out]
"""

import struct
import sys
from collections import defaultdict

WORD = 4


def read_manifest(path):
    calls, refs = [], []
    with open(path) as f:
        for line in f:
            fields = dict(
                kv.split("=", 1) for kv in line.split() if "=" in kv
            )
            entry = {
                "in": int(fields["in"]),
                "out": int(fields["out"]),
                "bits": int(fields["bits"]),
                "window": int(fields["window"]),
                "dtype": fields["dtype"],
            }
            if "call" in fields:
                entry["idx"] = int(fields["call"])
                calls.append(entry)
            else:
                entry["idx"] = int(fields["ref"])
                refs.append(entry)
    return calls, refs


def tuple_of(e):
    return (e["in"], e["out"], e["bits"], e["window"], e["dtype"])


def f32_to_f16_rne(value_bytes):
    """f32 (4 bytes LE) -> f16 bit pattern, round-to-nearest-even."""
    (b,) = struct.unpack("<I", value_bytes)
    sign = (b >> 16) & 0x8000
    exp = (b >> 23) & 0xFF
    man = b & 0x7FFFFF
    if exp == 0xFF:
        return sign | 0x7C00 | (0x200 if man else 0)
    ne = exp - 127 + 15
    if ne >= 31:
        return sign | 0x7C00
    if ne <= 0:
        if ne < -10:
            return sign
        m = man | 0x800000
        shift = 14 - ne
        rem = m & ((1 << shift) - 1)
        h = m >> shift
        if rem > (1 << (shift - 1)) or (
            rem == (1 << (shift - 1)) and (h & 1)
        ):
            h += 1
        return sign | h
    h = (ne << 10) | (man >> 13)
    rem = man & 0x1FFF
    if rem > 0x1000 or (rem == 0x1000 and ((man >> 13) & 1)):
        h += 1
    if h >= 0x7C00:
        return sign | 0x7C00
    return sign | h


def f32_to_bf16_rne(value_bytes):
    """f32 (4 bytes LE) -> bf16 bit pattern, round-to-nearest-even."""
    (b,) = struct.unpack("<I", value_bytes)
    bias = 0x7FFF + ((b >> 16) & 1)
    return ((b + bias) >> 16) & 0xFFFF


def load(path):
    with open(path, "rb") as f:
        return f.read()


def first_mismatch(a, b, width):
    n = min(len(a), len(b)) // width
    for i in range(n):
        if a[i * width:(i + 1) * width] != b[i * width:(i + 1) * width]:
            return i, n
    if len(a) != len(b):
        return n, n
    return None, n


def block_r_c(idx):
    return idx // 16384, (idx % 16384) // 128, idx % 128


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--stage="):
            only = a.split("=", 1)[1]
    dump_dir = args[0] if args else "."
    calls, refs = read_manifest(f"{dump_dir}/manifest.txt")

    by_tuple_calls = defaultdict(list)
    for e in calls:
        by_tuple_calls[tuple_of(e)].append(e)
    by_tuple_refs = defaultdict(list)
    for e in refs:
        by_tuple_refs[tuple_of(e)].append(e)

    failures = 0
    for t in sorted(by_tuple_calls):
        cs = by_tuple_calls[t]
        rs = by_tuple_refs.get(t, [])
        in_f, out_f, bits, window, dtype = t
        blocks = (in_f // 128) * (out_f // 128)
        print(
            f"tuple in={in_f} out={out_f} bits={bits} window={window} "
            f"dtype={dtype}: {len(cs)} gpu call(s), {len(rs)} ref(s)"
        )
        for k, c in enumerate(cs):
            if k >= len(rs):
                print(f"  call{c['idx']}: NO REF PAIRED")
                failures += 1
                continue
            r = rs[k]
            base_c = f"{dump_dir}/call{c['idx']}"
            base_r = f"{dump_dir}/ref{r['idx']}"
            stages = [
                ("dec16", f"{base_c}_dec16_u32.bin",
                 f"{base_r}_dec16_u32.bin", WORD),
                ("tmp", f"{base_c}_tmp_f32.bin",
                 f"{base_r}_tmp_f32.bin", WORD),
                ("out", f"{base_c}_out_{dtype}.bin",
                 f"{base_r}_out_f32.bin", None),
            ]
            for name, cf, rf, width in stages:
                if only and name != only:
                    continue
                ca, ra = load(cf), load(rf)
                if name == "out" and dtype != "f32":
                    width = 2
                    if dtype == "f16":
                        ra = b"".join(
                            struct.pack("<H", f32_to_f16_rne(ra[i:i + 4]))
                            for i in range(0, len(ra), 4)
                        )
                    else:
                        ra = b"".join(
                            struct.pack("<H", f32_to_bf16_rne(ra[i:i + 4]))
                            for i in range(0, len(ra), 4)
                        )
                idx, n = first_mismatch(ca, ra, width or WORD)
                if idx is None:
                    print(f"  call{c['idx']} vs ref{r['idx']} {name}: "
                          f"OK ({n} elements)")
                else:
                    line = (f"  call{c['idx']} vs ref{r['idx']} {name}: "
                            f"FIRST MISMATCH at {idx}/{n}")
                    if name in ("dec16", "tmp"):
                        b_, r_, c_ = block_r_c(idx)
                        line += (f" (block {b_}, row {r_}, col {c_})")
                    print(line)
                    failures += 1
    for t in sorted(by_tuple_refs):
        extra = len(by_tuple_refs[t]) - len(by_tuple_calls.get(t, []))
        if extra > 0:
            print(f"tuple {t}: {extra} unused ref(s)")
    print("RESULT:", "FAIL" if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
