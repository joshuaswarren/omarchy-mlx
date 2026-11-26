"""Run one probe once on one arm with the same idle/thermal procedure and
log as ab.py. usage: once.py <out.jsonl> <arm> -- <probe.py> [args...]
Extra environment: AB_ENV="K=V ...". """
import json
import os
import sys

import ab

out, arm = sys.argv[1], sys.argv[2]
probe = sys.argv[sys.argv.index("--") + 1:]
rec = ab.run_one(arm, probe, "")
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(arm, rec["rc"], "\n".join(rec["provenance"] + rec["stdout_tail"][-30:]))
for r in rec["result"]:
    print(json.dumps(r))
if rec["new_gpu_kernel_errors"]:
    print("NEW GPU KERNEL ERRORS")
    sys.exit(9)
sys.exit(rec["rc"])
