"""Pre-registered acceptance rule for branch gdn-recur32-mask (masked prefill takes the single-pass recur32 route),
written before the wheel is built or any device run. Args: pfbg log of arms qgpr and qrm (pfbg-ticket.sh format,
@@pg <arm> PFBG {...}), one log per host.
Per host: ratio = batched_s / seq_s, the median over the runs of an arm.
ACCEPT iff, on EVERY host given: qrm ratio <= 1.15 AND qrm ratio <= 0.85 * qgpr ratio (a real gain over the merged fix)
AND every qrm run has first_tokens_equal true OR pfbg2 names the only differing row a near-tie (margin < 0.25).
REJECT if any qrm ratio >= qgpr ratio. Otherwise INCONCLUSIVE. Numerics are checked separately by the fp64 doctest
(GDN recur32 masked prefill skips masked tokens) and by the device doctest dispatch count; both must pass on two chips."""
import json
import re
import statistics
import sys

ratios = {}
equal = {}
for path in sys.argv[1:]:
    for line in open(path):
        m = re.match(r"@@pg (\w+) PFBG (\{.*\})", line)
        if m:
            d = json.loads(m[2])
            ratios.setdefault((path, m[1]), []).append(d["ratio"])
            equal.setdefault((path, m[1]), []).append(d["first_tokens_equal"])
verdicts = []
for path in sys.argv[1:]:
    g, r = ratios.get((path, "qgpr")), ratios.get((path, "qrm"))
    if not g or not r:
        print(path, "VOID (need both arms)")
        verdicts.append("VOID")
        continue
    mg, mr = statistics.median(g), statistics.median(r)
    print(f"{path}: qgpr {mg:.3f} {g}  qrm {mr:.3f} {r}  first_tokens_equal qrm {equal[(path, 'qrm')]} qgpr {equal[(path, 'qgpr')]}")
    ok = mr <= 1.15 and mr <= 0.85 * mg
    verdicts.append("REJECT" if mr >= mg else "OK" if ok else "INCONCLUSIVE")
print("per-host:", verdicts)
print("DECISION:", "ACCEPT (if tokens equal or near-tie per pfbg2)" if verdicts and all(v == "OK" for v in verdicts)
      else "REJECT" if "REJECT" in verdicts else "INCONCLUSIVE")
