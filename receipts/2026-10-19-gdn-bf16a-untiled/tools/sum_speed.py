"""sum_speed.py RUNLOG BASE_ARM OTHER_ARM: median decode tok/s per prompt and arm over 5 rounds, paired wins, spread.

Reads the `time Pn round i ARM rc=0 decode X tok/s over 63 tokens` lines of a run log. Arm letters are single characters."""
import re
import statistics
import sys

path, base, other = sys.argv[1:4]
pattern = re.compile(r'time (P\d) round (\d) (\w) rc=(\d+) decode ([\d.]+) tok/s over 63 tokens')
rows = {}
with open(path, encoding='utf-8') as handle:
    for line in handle:
        m = pattern.search(line)
        if m:
            pn, rnd, arm, rc, tps = m.groups()
            assert rc == '0', line
            rows[(pn, int(rnd), arm)] = float(tps)
assert len(rows) == 20, len(rows)
for pn in ('P1', 'P2'):
    a = [rows[(pn, r, base)] for r in range(1, 6)]
    b = [rows[(pn, r, other)] for r in range(1, 6)]
    ma, mb = statistics.median(a), statistics.median(b)
    wins = sum(y > x for x, y in zip(a, b))
    spread = max((max(a) - min(a)) / ma, (max(b) - min(b)) / mb)
    print(f'{pn}: {base} median {ma:.3f}, {other} median {mb:.3f} tok/s; {other} vs {base} {100 * (mb - ma) / ma:+.2f}%; '
          f'{other} faster in {wins}/5; spread {100 * spread:.2f}%')
