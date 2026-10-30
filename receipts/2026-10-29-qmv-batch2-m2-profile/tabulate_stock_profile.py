import glob
import json
import statistics as st
import sys

d = sys.argv[1]
print("| set | wheel | step wall unprofiled, ms (3 rounds) | median | dispatches per step | profiled wall, ms | GPU busy, ms | batch kernel, ms | PfWide per step |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for s in ("E2", "P2", "E3", "P3"):
    for w in ("S", "H"):
        wall = [json.load(open(p))["wall_ms"] for p in sorted(glob.glob(f"{d}/{w}-{s}-r?-u-wall.json"))]
        sums = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/{w}-{s}-r?-sum.json"))]
        pf = st.median([x["by_name"].get("GatedDeltaDecodeBF16PfWide", {"n": 0})["n"] for x in sums])
        print(f"| {s} | {'shipped' if w == 'S' else 'hoisted'} | {', '.join(f'{x:.2f}' for x in wall)} | {st.median(wall):.2f} | "
              f"{st.median([x['disp'] for x in sums]):.0f} | {st.median([x['wall_ms'] for x in sums]):.1f} | "
              f"{st.median([x['gpu_ms'] for x in sums]):.2f} | {st.median([x['batch_kernel_ms'] for x in sums]):.3f} | {pf:.0f} |")
print()
for s in ("E2", "P2", "E3", "P3"):
    h = st.median([json.load(open(p))["wall_ms"] for p in glob.glob(f"{d}/H-{s}-r?-u-wall.json")])
    sh = st.median([json.load(open(p))["wall_ms"] for p in glob.glob(f"{d}/S-{s}-r?-u-wall.json")])
    print(f"{s}: shipped/hoisted speed ratio {sh / h:.3f}")
