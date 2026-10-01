#!/usr/bin/env python3
"""Body-preserving recount of unified-diff hunk headers.

Keeps every body line; rewrites only the @@ header counts, and recomputes
each hunk's new-file start as old_start + cumulative net added lines within
its file section (old starts are trusted as authored).
"""
import re, sys

for path in sys.argv[1:]:
    lines = open(path).read().splitlines(keepends=True)
    out, i, added = [], 0, 0
    while i < len(lines):
        l = lines[i]
        if l.startswith('--- '):
            added = 0
            out.append(l); i += 1; continue
        if l.startswith('+++ '):
            out.append(l); i += 1; continue
        m = re.match(r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@', l)
        if m:
            old_start = int(m.group(1))
            oc = nc = 0
            j = i + 1
            while j < len(lines) and not lines[j].startswith(('@@ ', '--- ', '+++ ')):
                c = lines[j][0]
                if c == '+': nc += 1
                elif c == '-': oc += 1
                else: oc += 1; nc += 1
                j += 1
            out.append(f'@@ -{old_start},{oc} +{old_start + added},{nc} @@\n')
            out.extend(lines[i + 1:j])          # bodies preserved
            added += nc - oc
            i = j
        else:
            out.append(l); i += 1
    open(path, 'w').write(''.join(out))
    print(path, "recounted")
