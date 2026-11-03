#!/usr/bin/env python3
"""Redaction used for the published v0.7.37 gate logs: python3 redact-logs.py IN_DIR OUT_DIR"""
import os
import re
import sys

RULES = [('/home/joshuawar[r]en', '$HOME'), ('jwm[1]-linux|jwm[1]', 'host-13in'), ('jw1[6]-linux|jw1[6]', 'host-16in'), ('omp-[A-Za-z0-9-]+', 'host-redacted'), ('[A-Za-z0-9-]+\\.loca[l]\\b', 'host-redacted'), ('(?i)as[a]hi', 'a[redacted]')]
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for name in sorted(os.listdir(src)):
    text = open(os.path.join(src, name), errors='replace').read()
    for pat, rep in RULES:
        text = re.sub(pat, rep, text)
    open(os.path.join(dst, name), 'w').write(text)
