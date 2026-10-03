#!/usr/bin/env python3
"""DrainFix: parse strace -f -y -tt -T -e trace=ioctl logs into a per-token decode census.

strace here has no DRM ioctl name decoding, so ioctls are grouped by raw request code.
The decode window is the last (tokens+warmup)/rate seconds of GPU (renderD*) ioctls.
Caveat: ptrace inflates every duration (compare ctl tok/s); COUNTS are exact.
"""
import collections
import re
import sys

LINE = re.compile(
    r"^\s*(?P<tid>\d+)\s+(?P<clock>[\d:.]+)\s+ioctl\((?P<args>.*)\)\s*=\s*\S+\s*<(?P<dur>[\d.]+)>"
)


def tsec(clock):
    parts = clock.split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def main() -> int:
    path = sys.argv[1]
    tokens = int(sys.argv[2])
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 109.0
    rows = []
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if m:
            args = m["args"]
            is_gpu = "renderD" in args.split(",")[0]
            rows.append((tsec(m["clock"]), args, float(m["dur"]), is_gpu, m["tid"]))
    if not rows:
        print("NO-IOCTL-LINES-PARSED")
        return 1

    def code_of(args):
        fields = args.split(",")
        return fields[1].strip() if len(fields) >= 2 else fields[0].strip()

    print("== all ioctl request codes (whole process) ==")
    for code, cnt in collections.Counter(code_of(a) for _, a, _, _, _ in rows).most_common():
        sample = next(a[:120] for _, a, _, _, _ in rows if code_of(a) == code)
        print(f"{cnt:8d}  {code}   sample: {sample}")

    gpu = [r for r in rows if r[3]]
    t_end = gpu[-1][0]
    warm = 8
    window_s = (tokens + warm) / rate
    span = [r for r in gpu if r[0] >= t_end - window_s]
    n_tokens_eff = tokens + warm
    total_us = sum(r[2] for r in span) * 1e6
    agg = collections.defaultdict(lambda: [0, 0.0])
    for _, args, dur, _, _ in span:
        agg[code_of(args)][0] += 1
        agg[code_of(args)][1] += dur
    print(f"\n== decode window: last {window_s:.3f}s (~{n_tokens_eff} tokens incl warmup), "
          f"ioctls={len(span)}, span_us={total_us:.0f} ==")
    print(f"{'count':>8} {'per/tok':>8} {'sum_us':>12} {'us/token':>10}  code")
    for code, (cnt, dur) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{cnt:8d} {cnt/n_tokens_eff:8.2f} {dur*1e6:12.0f} {dur*1e6/n_tokens_eff:10.1f}  {code}")
    print(f"{'TOTAL':8s} {len(span)/n_tokens_eff:8.2f} {total_us:12.0f} {total_us/n_tokens_eff:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
