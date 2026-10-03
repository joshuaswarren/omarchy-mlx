#!/usr/bin/env python3
"""Warm routing latency: 5 warm-up calls, then 100 timed routing calls
against a live Laya worker, cycling through a labelled suite's texts.
Records loadavg before and after and any other python/mlx processes, so
a timing is only published when the run was alone on the GPU."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from dev_sweep_run import QUESTION_DEFAULT, build_payload, call_decision


def others():
    out = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True).stdout
    me = {str(os.getpid()), str(os.getppid())}
    return sum(1 for line in out.splitlines()[1:]
               if ("python" in line or "mlx" in line)
               and line.split(None, 1)[0] not in me
               and "mlx_omarchy_laya.server" not in line
               and "routing_latency" not in line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-url", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = [c["text"] for c in json.loads(Path(args.suite).read_text())["cases"]]
    before = {"loadavg": Path("/proc/loadavg").read_text().strip(), "other_procs": others()}
    for text in texts[:5]:
        call_decision(args.decision_url, build_payload(text, QUESTION_DEFAULT), 5.0)
    timings = []
    for i in range(100):
        _, ms, failed = call_decision(args.decision_url,
                                      build_payload(texts[i % len(texts)], QUESTION_DEFAULT), 5.0)
        timings.append({"ms": ms, "failed": failed})
    after = {"loadavg": Path("/proc/loadavg").read_text().strip(), "other_procs": others()}
    ok = sorted(t["ms"] for t in timings if not t["failed"])
    result = {"before": before, "after": after, "n": len(timings),
              "failed": sum(t["failed"] for t in timings),
              "p50_ms": ok[len(ok) // 2] if ok else None,
              "p95_ms": ok[int(len(ok) * 0.95) - 1] if ok else None,
              "max_ms": ok[-1] if ok else None, "timings": timings}
    tmp = Path(args.out + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, args.out)
    os.sync()
    print(f"latency p50={result['p50_ms']} p95={result['p95_ms']} failed={result['failed']} "
          f"before={before} after={after}")


if __name__ == "__main__":
    main()
