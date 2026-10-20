#!/usr/bin/env python3
"""Give BatchRotatingKVCache the host offset mirror BatchKVCache already has (sliding-window RoPE host joins).

Root cause (13-inch M1 serve-loop census, 2026-10-19/20, gemma-4-E4B): the omarchy RoPE trig gate reads a one-element
offset directly only when it is a host constant (array::Status::available, no primitive); anything else takes
settle + synchronize("rope_offset_scalar"), a host round trip per call. patch-mlx-lm-kv-host-offset.py gave BatchKVCache a
host mirror so its offset is `mx.array(host list)`. BatchRotatingKVCache kept `self.offset = self.offset + S` on the device,
so each sliding-window layer paid a device add (20 ElementwiseI32 dispatches per decoded token in E4B) and a join.

The cache now keeps `offset_list`, a host mirror of `offset`:
  * set in __init__, shifted in-class by `_shift_offset` in _update_in_place and _update_concat, where the stored offset is
    `mx.array(offset_list)` (same int32 values; MLX_OMARCHY_KV_HOST_OFFSET=0 restores the lazy device add);
  * an assignment from outside the mirror (filter, extend, merge, prepare, finalize, a state restore) clears it through the
    `offset` property setter, and the first decode step after that rebuilds it once from the device value (one host read per
    batch composition change; PromptProcessingBatch.split calls filter on every request);
  * once the window rotates, `left_padding` is stored as `mx.array(mirror)` instead of a device subtract.
Needs patch-mlx-lm-kv-host-offset.py (defines _KV_HOST_OFFSET) and patch-mlx-lm-rotkv-maskless.py (defines _rotkv_host_list)
applied first. Anchors on the mlx-lm 0.32 BatchRotatingKVCache; the 0.31.3 series is skipped by apply-mlx-lm-patches.sh.
Usage: python3 patch-mlx-lm-rotkv-host-offset.py /path/to/venv
"""
import ast
import glob
import sys

MARKER = "BatchRotatingKVCache host offset mirror"

INIT_OLD = """    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None

        self.left_padding = mx.array(left_padding)
        self.left_padding_list = _rotkv_host_list(left_padding)
        self.offset = mx.array([-l for l in left_padding])
"""
INIT_NEW = """    # BatchRotatingKVCache host offset mirror: offset_list mirrors offset on the
    # host, in-class updates store mx.array(offset_list) instead of a device add
    # (MLX_OMARCHY_KV_HOST_OFFSET=0 keeps the lazy device array), so the RoPE
    # gate reads it without a host join. An outside assignment clears the mirror.
    @property
    def offset(self):
        return self._offset_array

    @offset.setter
    def offset(self, value):
        self._offset_array = value
        self.offset_list = None

    def _set_offset(self, device, host):
        self._offset_array = mx.array(host) if host is not None and _KV_HOST_OFFSET else device
        self.offset_list = host

    def _shift_offset(self, device, delta):
        m = self.offset_list
        if m is not None:
            m = [o + delta for o in m]
        self._set_offset(device, m)

    def __init__(self, max_size, left_padding: List[int]):
        self.keys = None
        self.values = None

        self.left_padding = mx.array(left_padding)
        self.left_padding_list = _rotkv_host_list(left_padding)
        lp = self.left_padding_list
        self._set_offset(mx.array([-l for l in left_padding]), None if lp is None else [-l for l in lp])
"""

CONCAT_OLD = """        self.offset = self.offset + keys.shape[2]
        self._offset += keys.shape[2]
"""
CONCAT_NEW = """        self._shift_offset(self.offset + keys.shape[2], keys.shape[2])
        self._offset += keys.shape[2]
"""

START_OLD = """        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
"""
START_NEW = """        B, n_kv_heads, S, k_head_dim = keys.shape
        prev = self._offset
        if _KV_HOST_OFFSET and self.offset_list is None:
            self.offset_list = self.offset.tolist()
"""

ROTATED_OLD = """        if self.rotated:
            mirror = self.left_padding_list
            self.left_padding -= S
            if mirror is not None:
                self.left_padding_list = [p - S for p in mirror]
"""
ROTATED_NEW = """        if self.rotated:
            mirror = self.left_padding_list
            if mirror is None and _KV_HOST_OFFSET:
                mirror = self.left_padding.tolist()
            if mirror is not None and _KV_HOST_OFFSET:
                mirror = [p - S for p in mirror]
                self.left_padding = mx.array(mirror)
                self.left_padding_list = mirror
            else:
                self.left_padding -= S
                if mirror is not None:
                    self.left_padding_list = [p - S for p in mirror]
"""

INPLACE_OLD = """        self._offset += S
        self.offset = self.offset + S
"""
INPLACE_NEW = """        self._offset += S
        self._shift_offset(self.offset + S, S)
"""

REPLACEMENTS = (
    (INIT_OLD, INIT_NEW),
    (CONCAT_OLD, CONCAT_NEW),
    (START_OLD, START_NEW),
    (ROTATED_OLD, ROTATED_NEW),
    (INPLACE_OLD, INPLACE_NEW),
)

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
q = site[0] + "/cache.py"
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if "_KV_HOST_OFFSET" not in text:
    sys.exit("cache.py has no _KV_HOST_OFFSET (apply patch-mlx-lm-kv-host-offset.py first)")
if "_rotkv_host_list" not in text:
    sys.exit("cache.py has no _rotkv_host_list (apply patch-mlx-lm-rotkv-maskless.py first)")
for old, new in REPLACEMENTS:
    if text.count(old) != 1:
        sys.exit("unrecognized BatchRotatingKVCache content in " + q + " near: " + old.strip().splitlines()[0]
                 + "; refusing to patch (mlx-lm 0.32 series only)")
    text = text.replace(old, new)
ast.parse(text)
open(q, "w").write(text)
print("patched:", q)
