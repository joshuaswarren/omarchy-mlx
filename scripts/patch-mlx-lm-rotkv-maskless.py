#!/usr/bin/env python3
"""Skip the all-valid BatchRotatingKVCache decode mask (sliding-window SDPA fix).

Root cause (13-inch M1 serve-loop census, 2026-10-18): mlx_lm.server's
BatchGenerator serves sliding-window models through BatchRotatingKVCache,
whose make_mask always returns an array. While the window is not binding the
mask is all-True by construction (one unpadded query at the newest position,
every cached key causal and inside the window), but an array mask fails the
backend's native SDPA decode gate (inputs.size() == 3), so every
sliding-window layer composed its decode attention: for gemma-4-E4B that is
35 layers x ~16 dispatches = +568 dispatches and +5.18 ms per token (19%
slower than the bench_decode route on the same chip; token ids unchanged).

make_mask(1) now returns None when a host mirror of left_padding proves the
mask all-True: no padding, no rotation, no right-padding lengths,
_idx == _offset, and _offset < window_size. Prefill (N > 1), padded batches,
rotated caches, window-bound calls and return_array calls keep the array.

The mirror rides left_padding_list, rebuilt at every in-class mutation site
(init, update paths, prepare, finalize, filter, extend). left_padding becomes
a property whose setter clears the mirror, so ANY assignment from outside
this class (state restores, cache copies) disables the guard. Only
list/tuple paddings are mirrored (an mx array argument would need a host
sync).

Kill switch at runtime: MLX_OMARCHY_ROTKV_MASKLESS=0 (read once at import).
Anchors on the mlx-lm 0.32 BatchRotatingKVCache; the 0.31.3 series is
skipped by apply-mlx-lm-patches.sh.
Usage: python3 patch-mlx-lm-rotkv-maskless.py /path/to/venv
"""
import ast
import glob
import sys

MARKER = "MLX_OMARCHY_ROTKV_MASKLESS"

CLASS_OLD = "class BatchRotatingKVCache(_BaseCache):\n    step = 256\n"
CLASS_NEW = '''# A BatchRotatingKVCache whose rows carry no left padding needs no decode
# mask while the sliding window is not binding: at N == 1 the query is the
# newest position, so every cached key is causal, and while
# _offset < window_size every cached key is inside the window too. make_mask
# returns None there, which keeps the native SDPA decode kernel reachable on
# the batched serve path. MLX_OMARCHY_ROTKV_MASKLESS=0 restores the array
# mask.
_ROTKV_MASKLESS_UNPADDED = os.environ.get("MLX_OMARCHY_ROTKV_MASKLESS", "1") != "0"


def _rotkv_host_list(x):
    return list(x) if isinstance(x, (list, tuple)) else None


class BatchRotatingKVCache(_BaseCache):
    step = 256

    # Assigning left_padding clears the host mirror; in-class sites rebuild
    # it after they assign, so a write from anywhere else (state restores,
    # cache copies) disables the maskless guard (the safe default).
    @property
    def left_padding(self):
        return self._left_padding

    @left_padding.setter
    def left_padding(self, value):
        self._left_padding = value
        self.left_padding_list = None
'''

INIT_OLD = """        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
"""
INIT_NEW = """        self.left_padding = mx.array(left_padding)
        self.left_padding_list = _rotkv_host_list(left_padding)
        self.offset = mx.array([-l for l in left_padding])
"""

CONCAT_ROLL_OLD = """                self.left_padding += roll
                self.offset = self.offset - roll
"""
CONCAT_ROLL_NEW = """                self.left_padding += roll
                self.left_padding_list = None
                self.offset = self.offset - roll
"""

CONCAT_TRIM_OLD = """            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                self.left_padding -= trim_size
"""
CONCAT_TRIM_NEW = """            trim_size = self._idx - self.max_size + 1
            if trim_size > 0:
                mirror = self.left_padding_list
                self.left_padding -= trim_size
                if mirror is not None:
                    self.left_padding_list = [p - trim_size for p in mirror]
"""

INPLACE_TRIM_OLD = """            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            self.left_padding -= trim_size
"""
INPLACE_TRIM_NEW = """            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size
            mirror = self.left_padding_list
            self.left_padding -= trim_size
            if mirror is not None:
                self.left_padding_list = [p - trim_size for p in mirror]
"""

INPLACE_ROTATE_OLD = """        if self.rotated:
            self.left_padding -= S
"""
INPLACE_ROTATE_NEW = """        if self.rotated:
            mirror = self.left_padding_list
            self.left_padding -= S
            if mirror is not None:
                self.left_padding_list = [p - S for p in mirror]
"""

PREPARE_OLD = """            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset = self.offset - left_padding
"""
PREPARE_NEW = """            mirror, added = self.left_padding_list, _rotkv_host_list(left_padding)
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            if mirror is not None and added is not None:
                self.left_padding_list = [a + b for a, b in zip(mirror, added)]
            self.offset = self.offset - left_padding
"""

FINALIZE_OLD = """            self.left_padding += roll
            self.offset = self.offset - roll
            self._lengths = None
"""
FINALIZE_NEW = """            self.left_padding += roll
            self.left_padding_list = None
            self.offset = self.offset - roll
            self._lengths = None
"""

FILTER_OLD = """        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
"""
FILTER_NEW = """        mirror = self.left_padding_list
        if mirror is not None:
            if isinstance(batch_indices, (list, tuple)):
                mirror = [mirror[i] for i in batch_indices]
            else:
                mirror = None
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        self.left_padding_list = mirror
"""

EXTEND_EMPTY_OLD = """            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
"""
EXTEND_EMPTY_NEW = """            mirrors = (self.left_padding_list, other.left_padding_list)
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            if None not in mirrors:
                self.left_padding_list = mirrors[0] + mirrors[1]
            self.offset = mx.concatenate([self.offset, other.offset])
"""

EXTEND_OLD = """        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx
"""
EXTEND_NEW = """        mirror = None
        if self.left_padding_list is not None and other.left_padding_list is not None:
            mirror = [p + max_idx - self._idx for p in self.left_padding_list] + [
                p + max_idx - other._idx for p in other.left_padding_list
            ]
        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self.left_padding_list = mirror
        self._idx = max_idx
"""

MASK_OLD = """    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        left_padding = self.left_padding
"""
MASK_NEW = """    def make_mask(
        self, N: int, window_size: Optional[int] = None, return_array: bool = False
    ):
        if (
            _ROTKV_MASKLESS_UNPADDED
            and N == 1
            and not return_array
            and not self.rotated
            and self._lengths is None
            and self._idx == self._offset
            and self._idx < self.max_size
            and self._offset < (window_size or self.max_size)
            and self.left_padding_list is not None
            and not any(self.left_padding_list)
        ):
            # Unpadded single-token decode with every key causal and inside
            # the window: the stock mask below is all-True, so attention can
            # run without it (native SDPA decode kernel).
            return None
        left_padding = self.left_padding
"""

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
q = site[0] + "/cache.py"
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if "\nimport os\n" not in text:
    sys.exit("cache.py has no `import os` (apply patch-mlx-lm-ssm-maskless.py first)")

start = text.find(CLASS_OLD)
end = text.find("\nclass ", start + len(CLASS_OLD))
if start < 0 or end < 0:
    sys.exit("BatchRotatingKVCache not found in " + q)
body = text[start:end]
for old, new in [(CLASS_OLD, CLASS_NEW), (INIT_OLD, INIT_NEW),
                 (CONCAT_ROLL_OLD, CONCAT_ROLL_NEW), (CONCAT_TRIM_OLD, CONCAT_TRIM_NEW),
                 (INPLACE_TRIM_OLD, INPLACE_TRIM_NEW), (INPLACE_ROTATE_OLD, INPLACE_ROTATE_NEW),
                 (PREPARE_OLD, PREPARE_NEW), (FINALIZE_OLD, FINALIZE_NEW),
                 (FILTER_OLD, FILTER_NEW), (EXTEND_EMPTY_OLD, EXTEND_EMPTY_NEW),
                 (EXTEND_OLD, EXTEND_NEW), (MASK_OLD, MASK_NEW)]:
    if body.count(old) != 1:
        sys.exit("unrecognized BatchRotatingKVCache content in " + q + "; refusing to patch "
                 "(mlx-lm version mismatch?)")
    body = body.replace(old, new, 1)
text = text[:start] + body + text[end:]
ast.parse(text)
open(q, "w").write(text)
print("patched:", q)
