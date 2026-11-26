#!/usr/bin/env python3
"""Compare Jw16JwmTransfer control vs candidate pp_cells outputs.
usage: compare_cells.py CTL_DIR CAND_DIR
Prints per-cell medians, % delta, digest pin matches, kb hash diffs, logits gate lines."""
import json, os, sys

PINS = {
    "d64": "c84b3e7a", "d128": "8e7b5dd9", "d256": "7824b835", "d512": "eaaa7206",
    "pf512": "100a61b62470",
}
LOGITS_PINS = {"512": "00d7ed153c08", "1024": "2a89e403678b", "2048": "9e540a29a347"}

def load(path):
    with open(path) as f:
        return json.load(f)

def decode_cell(d):
    dr = d.get("decode_tok_rate", {})
    return dr.get("median"), dr.get("stdev"), d.get("ordered_records_sha256", "")

def pf_rate(d):
    return d.get("pure_prefill", {}).get("pure_prefill_tok_rate")

def logits_lines(d):
    out = {}
    if isinstance(d, dict):
        out[str(d.get("tokens") or d.get("T") or "")] = d.get("logits_sha256")
    return out

def main(ctl, cand):
    print(f"control={ctl}\ncandidate={cand}")
    print("\n== decode cells ==")
    for c in ("d64", "d128", "d256", "d512"):
        cj, nj = os.path.join(ctl, f"qwen-gpu-{c}-n5.json"), os.path.join(cand, f"qwen-gpu-{c}-n5.json")
        if not (os.path.exists(cj) and os.path.exists(nj)):
            print(f"{c}: MISSING ({cj if not os.path.exists(cj) else nj})"); continue
        cm, cs, cdig = decode_cell(load(cj)); nm, ns, ndig = decode_cell(load(nj))
        delta = (nm - cm) / cm * 100 if cm and nm else float("nan")
        pin = PINS[c]
        print(f"{c}: ctl {cm} ±{cs} cand {nm} ±{ns}  delta {delta:+.2f}%  "
              f"digests ctl={cdig[:8]} cand={ndig[:8]} pin={pin} "
              f"CTL{'OK' if cdig.startswith(pin) else 'MISMATCH'} CAND{'OK' if ndig.startswith(pin) else 'MISMATCH'}")
    print("\n== pf cells (5 runs each) ==")
    for c in ("pf512", "pf1024", "pf2048"):
        crates, nrates = [], []
        for i in range(1, 6):
            cj, nj = os.path.join(ctl, f"qwen-gpu-{c}-{i}.json"), os.path.join(cand, f"qwen-gpu-{c}-{i}.json")
            if os.path.exists(cj): crates.append(pf_rate(load(cj)))
            if os.path.exists(nj): nrates.append(pf_rate(load(nj)))
        if not crates or not nrates:
            print(f"{c}: MISSING (ctl {len(crates)}, cand {len(nrates)} runs)"); continue
        cm, nm = sorted(crates)[len(crates)//2], sorted(nrates)[len(nrates)//2]
        delta = (nm - cm) / cm * 100
        extra = ""
        if c in PINS:
            cdig = load(os.path.join(ctl, f"qwen-gpu-{c}-1.json")).get("ordered_records_sha256", "")
            ndig = load(os.path.join(cand, f"qwen-gpu-{c}-1.json")).get("ordered_records_sha256", "")
            pin = PINS[c]
            extra = f" digests ctl={cdig[:12]} cand={ndig[:12]} pin={pin} CTL{'OK' if cdig.startswith(pin) else 'MISMATCH'} CAND{'OK' if ndig.startswith(pin) else 'MISMATCH'}"
        print(f"{c}: ctl median {cm} (n={len(crates)}) cand median {nm} (n={len(nrates)})  delta {delta:+.2f}%{extra}")
    print("\n== logits gates ==")
    for tag in ("ctl", "cand"):
        base = ctl if tag == "ctl" else cand
        for c in ("pf512", "pf1024", "pf2048"):
            p = os.path.join(base, f"ld-{c}.err")
            if os.path.exists(p):
                for line in open(p):
                    if "logits_sha256" in line:
                        print(f"{tag} {c}: {line.strip()} [-12:]={line.strip().split()[-1][-12:]}")
    print("\n== kernel_bits ==")
    kb_c = os.path.join(ctl, "kb.json"); kb_n = os.path.join(cand, "kb.json")
    if os.path.exists(kb_c) and os.path.exists(kb_n):
        ckb, nkb = load(kb_c), load(kb_n)
        def flat(d, pre=""):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict): out.update(flat(v, f"{pre}{k}."))
                elif isinstance(v, str) and len(v) == 16: out[f"{pre}{k}"] = v
            return out
        cf, nf = flat(ckb), flat(nkb)
        same = diff = missing = 0
        for k in sorted(set(cf) | set(nf)):
            if k not in cf or k not in nf: missing += 1; continue
            if cf[k] == nf[k]: same += 1
            else: diff += 1; print(f"  kb DIFF {k}: ctl {cf[k]} cand {nf[k]}")
        print(f"kb hashes: {same} identical, {diff} different, {missing} missing")
    else:
        print("kb.json missing", kb_c if not os.path.exists(kb_c) else kb_n)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
