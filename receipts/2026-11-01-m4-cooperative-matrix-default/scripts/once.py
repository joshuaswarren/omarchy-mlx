"""Run one probe once on one arm with the same idle/thermal procedure and
log as ab.py. usage: once.py <out.jsonl> <arm> -- <probe.py> [args...]
Extra environment: AB_ENV="K=V ...". Exits with the probe's status, 9 on a new
GPU driver error and 8 when GPU error monitoring is unavailable (ab.py's contract). """
import json
import os
import sys

import ab

out, arm = sys.argv[1], sys.argv[2]
probe = sys.argv[sys.argv.index("--") + 1:]
try:
    ab.gpu_errors()
    rec = ab.run_one(arm, probe, "")
except ab.MonitorUnavailable as e:
    print(f"REFUSED: GPU error monitoring is unavailable ({e}); the probe does not run", file=sys.stderr)
    sys.exit(8)
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(arm, rec["rc"], "\n".join(rec["provenance"] + rec["stdout_tail"][-30:]))
for r in rec["result"]:
    print(json.dumps(r))
if rec["monitor_error"]:
    print(f"STOP: GPU error monitoring failed after the run ({rec['monitor_error']})")
    sys.exit(8)
if rec["new_gpu_kernel_errors"]:
    print("NEW GPU KERNEL ERRORS")
    sys.exit(9)
sys.exit(rec["rc"])
