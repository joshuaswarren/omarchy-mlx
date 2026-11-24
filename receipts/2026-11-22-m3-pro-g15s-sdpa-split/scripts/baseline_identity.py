"""docs/parity-contract.md 4.2 baseline noise: usage baseline_identity.py GATE_DIR BASELINE_DIR. For each model, whether each of the five
identical-protocol main captures NB1..NB5 equals the gate capture NG-main bit for bit, and the spread (max - min) of their |log(PPL/PPL_ref)|
against each reference present."""
import glob, json, sys
import numpy as np
g, b = sys.argv[1], sys.argv[2]
for m in ("qwen3-4b-2507-4bit", "qwen3.8-2b-4bit"):
    main = np.load(f"{g}/NG-main-{m}.npz")
    caps = [np.load(f"{b}/NB{i}-main-{m}.npz") for i in range(1, 6)]
    same = [all(np.array_equal(c[k], main[k]) for k in ("argmax", "top1", "top2", "target_lp")) for c in caps]
    spread = {}
    for ref in sorted(glob.glob(f"{g}/NG-*-{m}.npz")):
        name = ref.rsplit("/", 1)[1][3:-4]
        if not (name.startswith("fp32") or name.startswith("cpuref")) or "first2560" in name:
            continue
        r = np.load(ref)
        n = len(r["target_lp"])
        devs = [abs(float(np.mean(r["target_lp"]) - np.mean(c["target_lp"][:n]))) for c in caps]
        spread[name] = max(devs) - min(devs)
    print(json.dumps({"model": m, "captures": 5, "bit_identical_to_NG_main": same, "deviation_spread": spread}))
