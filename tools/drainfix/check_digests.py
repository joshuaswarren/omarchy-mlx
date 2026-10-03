#!/usr/bin/env python3
"""DrainFix: verify result JSONs' greedy digests against the DecodeBw pins and summarize."""
import glob
import importlib.util
import json
import os
import statistics
import sys

PINS_FILE = os.environ.get("DF_PINS", "/var/tmp/drainfix/pins.py")
_spec = importlib.util.spec_from_file_location("df_pins", PINS_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PINS = _mod.PINS  # (new_tokens, passes) -> digest; unknown keys reported as "?"


def main() -> int:
    outdir = sys.argv[1]
    bad = 0
    rows = []
    for path in sorted(glob.glob(f"{outdir}/*.json")):
        if path.endswith(("package.txt",)):
            continue
        try:
            data = json.load(open(path))
        except json.JSONDecodeError:
            print(f"JSON-DECODE-FAIL {path}")
            bad += 1
            continue
        label = data.get("label") or os.path.basename(path)[:-5]
        proto = data.get("protocol") or {}
        depth = proto.get("new_tokens", 0)
        passes = proto.get("passes", 1)
        digest = data.get("ordered_records_sha256", "")
        rates = [r["decode_tok_rate"] for r in data.get("per_prompt", []) if r.get("decode_tok_rate")]
        med = statistics.median(rates) if rates else 0.0
        mn, mx = (min(rates), max(rates)) if rates else (0.0, 0.0)
        pin = PINS.get((int(depth), int(passes)))
        ok = (digest == pin) if pin else None
        if ok is False:
            bad += 1
        rows.append((label, int(depth), med, mn, mx, len(rates), "OK" if ok else ("?" if ok is None else "DIGEST-FAIL")))
    if not rows:
        print("NO-RESULT-JSONS")
        return 1
    print(f"{'label':32s} {'d':>4} {'med tok/s':>10} {'min':>8} {'max':>8} {'n':>2}  digest")
    for label, depth, med, mn, mx, n, ok in rows:
        print(f"{label:32s} {depth:4d} {med:10.2f} {mn:8.2f} {mx:8.2f} {n:2d}  {ok}")
    print(f"DIGEST_FAILURES={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
