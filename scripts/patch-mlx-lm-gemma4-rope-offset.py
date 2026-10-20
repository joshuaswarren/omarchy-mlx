#!/usr/bin/env python3
"""gemma-4: hand an array cache offset to RoPE as is, not through mx.array(cache.offset).

Root cause (13-inch M1 serve-loop trace census, 2026-10-20, gemma-4-E4B, MLX_OMARCHY_TRACE_DISPATCH=1): BatchGenerator ran 24 extra
`JOIN reason=rope_offset_scalar` host joins, 20 extra submissions and 24 extra buffer copies per decoded token against
bench_decode, +3.8 ms (6 percent) per token. rope_trig_gate (overlay/mlx/backend/omarchy/primitives.cpp) reads a one-element
offset directly only when it is a host constant (array::Status::available, no primitive). gemma4_text.py built the offset as
`mx.array(cache.offset)`: for the single-stream cache's Python int that is a host constant, but for a batch cache's offset array
it is a Copy node with a primitive, so every own-cache layer (24 in E4B) joined the stream even with the host-mirrored offset.

An array offset is now passed through; an int still goes through mx.array. The batch caches replace their offset array on every
update and never mutate it, so the layer's reference stays valid. Same values and shape as the copy it replaces.
Needs patch-mlx-lm-kv-host-offset.py and patch-mlx-lm-rotkv-host-offset.py for the offset to be a host constant. A venv without
gemma4_text.py (older mlx-lm) is skipped.
Usage: python3 patch-mlx-lm-gemma4-rope-offset.py /path/to/venv
"""
import ast
import glob
import sys

MARKER = "isinstance(cache.offset, mx.array)"
OLD = "            offset = mx.array(cache.offset) if cache is not None else 0\n"
NEW = (
    "            offset = 0\n"
    "            if cache is not None:\n"
    "                offset = cache.offset if isinstance(cache.offset, mx.array) else mx.array(cache.offset)\n"
)

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models/gemma4_text.py")
if not site:
    print("no gemma4_text.py under " + venv + "; nothing to patch")
    sys.exit(0)
q = site[0]
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if text.count(OLD) != 1:
    sys.exit("unrecognized gemma4_text.py content in " + q + "; refusing to patch")
text = text.replace(OLD, NEW)
ast.parse(text)
open(q, "w").write(text)
print("patched:", q)
