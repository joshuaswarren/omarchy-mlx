#!/usr/bin/env python3
# prefill_gap_reader.py CENSUS_DIR_OR_TARBALL — read a prefill_census.sh
# capture and name which family owns the prompt-length-scaling dispatch
# work: the coopmat-off tile fallback, or GEMM coverage.
#
# The capture holds histograms of "[rtmod] DISPATCH kernel=<id> count=<n>
# gx= gy= gz=" at two prompt sizes (default 512 and 4096). The positive
# delta (big - small) is the work that grows with prompt length — the
# prefill path itself, with fixed per-run cost cancelled.
#
# Verdict rules, on that delta:
#   - zero coopmat-family dispatches while tile/vec matmul carries the
#     growth  -> COOPMAT-OFF TILE FALLBACK owns the gap
#   - Dequant* + generic Matmul* growth while fused Qmm* growth is ~0
#     -> GEMM COVERAGE owns the gap (ops fall back to dequant+matmul)
#   - both/ambiguous -> print both shares, no verdict
# Note: a census counts dispatches, not nanoseconds. It names which path
# runs; pair it with a timed profile to name the ns owner.
import re
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from compute_kernel_names import KERNEL_NAMES
except ImportError:
    KERNEL_NAMES = []

FAMILIES = {
    "coopmat": re.compile(r"Coopmat|CoopOp"),
    "tile_or_vec_matmul": re.compile(
        r"^(Matmul|QmmTile|QmmVec|GatherQmm|QmmPrefillFma|MatmulBf16Fma|MatmulDirect|MatmulRb|Bonsai|Int8Matmul|QmmVec6|MatmulVec)"
    ),
    "fused_qmm": re.compile(r"^Qmm"),
    "dequant": re.compile(r"^Dequant"),
    "sdpa": re.compile(r"^Sdpa"),
}


def kname(kid):
    return KERNEL_NAMES[kid] if 0 <= kid < len(KERNEL_NAMES) else f"kernel#{kid}"


def load_hist(path):
    hist = Counter()
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*(\d+)\s+kernel=(\d+) count=(\d+)", line)
        if m:
            runs, kid, cnt = int(m.group(1)), int(m.group(2)), int(m.group(3))
            hist[kid] += runs * cnt
    return hist


def main():
    arg = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as td:
        if arg.suffix == ".gz":
            tarfile.open(arg).extractall(td)
            base = Path(td)
        else:
            base = arg
        hists = {p.stem: load_hist(p) for p in base.glob("hist-*.txt")}
        meta = next(base.glob("meta.txt"), None)
    if len(hists) < 2:
        sys.exit("need hist-p*.txt from prefill_census.sh")
    small = hists[min(hists, key=lambda k: int(k.split("-p")[1]))]
    big = hists[max(hists, key=lambda k: int(k.split("-p")[1]))]
    if meta:
        print(meta.read_text().strip())

    delta = {k: v - small.get(k, 0) for k, v in big.items() if v > small.get(k, 0)}
    total = sum(delta.values()) or 1
    fam = Counter()
    for kid, n in delta.items():
        name = kname(kid)
        for f, pat in FAMILIES.items():
            if pat.search(name):
                fam[f] += n
                break
        else:
            fam["other"] += n

    print(f"\nlength-scaling dispatches: {total}")
    for f, n in fam.most_common():
        print(f"  {f:20s} {n:9d}  {n/total*100:5.1f}%")
    print("\ntop kernels in the delta:")
    for kid, n in sorted(delta.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {kname(kid):42s} x{n}")

    coop = fam["coopmat"]
    fused = fam["fused_qmm"]
    deq = fam["dequant"]
    matmulish = fam["tile_or_vec_matmul"] + coop
    print()
    if coop == 0 and fam["tile_or_vec_matmul"] / total > 0.5:
        print("VERDICT: coopmat-off tile fallback owns the length-scaling work")
        print(f"(0 coopmat dispatches; {fam['tile_or_vec_matmul']/total*100:.0f}% of the delta is tile/vec matmul)")
    elif deq / total > 0.05 and fused / max(matmulish, 1) < 0.2:
        print("VERDICT: GEMM coverage owns the length-scaling work")
        print(f"(dequant+generic matmul grows; fused Qmm* covers only {fused/max(matmulish,1)*100:.0f}% of matmul dispatches)")
    else:
        print(f"VERDICT: ambiguous — coopmat={coop} fused_qmm={fused} dequant={deq} "
              f"tile_vec={fam['tile_or_vec_matmul']} of {total}")


if __name__ == "__main__":
    main()
