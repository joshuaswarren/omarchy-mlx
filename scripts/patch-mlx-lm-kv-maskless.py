#!/usr/bin/env python3
"""Skip the all-valid BatchKVCache decode mask (batched serve-path SDPA fix).

Root cause (receipts/2026-10-07-batch-decode-2): the mlx-lm BatchGenerator
serves concurrent requests through BatchKVCache, whose make_mask always
returns an array (create_causal_mask with left_padding), even when no row is
padded. An array mask fails the backend's native SDPA decode gate
(inputs.size() == 3), so every full-attention layer of a batched decode step
ran the composed attention chain.

At N == 1 with no left padding the mask is all-True by construction: the one
query sits at the newest position, so every cached key is causal and valid.
make_mask then returns None (no window, no return_array). Prefill (N > 1) and
padded batches keep the real mask.

The host knowledge rides left_padding_list, a Python mirror of the mx array.
left_padding becomes a property whose setter clears the mirror, so ANY
assignment from outside this class (oMLX rollback/rotation paths, state
restores, cache copies) disables the guard; the in-class mutation sites
rebuild the mirror right after they assign. Only list/tuple paddings are
mirrored (an mx array argument would need a host sync).

Kill switch at runtime: MLX_OMARCHY_KV_MASKLESS=0 (read once at import).
Usage: python3 patch-mlx-lm-kv-maskless.py /path/to/venv
"""
import ast
import glob
import sys

MARKER = "MLX_OMARCHY_KV_MASKLESS"

CLASS_OLD = "class BatchKVCache(_BaseCache):\n    step = 256\n"
CLASS_NEW = '''# A BatchKVCache whose rows carry no left padding needs no decode mask: at
# N == 1 the query is the newest position, so every cached key is valid.
# make_mask returns None there, which keeps the native SDPA decode kernel
# reachable on the batched serve path. MLX_OMARCHY_KV_MASKLESS=0 restores
# the array mask.
_KV_MASKLESS_UNPADDED = os.environ.get("MLX_OMARCHY_KV_MASKLESS", "1") != "0"


def _host_list(x):
    return list(x) if isinstance(x, (list, tuple)) else None


class BatchKVCache(_BaseCache):
    step = 256

    # Assigning left_padding clears the host mirror; in-class sites rebuild
    # it after they assign, so a write from anywhere else disables the
    # maskless guard (the safe default).
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
        self._idx = 0

        self._right_padding = None
"""
INIT_NEW = """        self.left_padding = mx.array(left_padding)
        self.left_padding_list = _host_list(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0

        self._right_padding = None
        self._right_padding_list = None
"""

PREPARE_OLD = """            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset = self.offset - left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
"""
PREPARE_NEW = """            mirror, added = self.left_padding_list, _host_list(left_padding)
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            if mirror is not None and added is not None:
                self.left_padding_list = [a + b for a, b in zip(mirror, added)]
            self.offset = self.offset - left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)
            self._right_padding_list = _host_list(right_padding)
"""

FINALIZE_OLD = """            self.offset = self.offset - padding
            self.left_padding += padding
            self._right_padding = None
"""
FINALIZE_NEW = """            self.offset = self.offset - padding
            mirror, added = self.left_padding_list, self._right_padding_list
            self.left_padding += padding
            if mirror is not None and added is not None:
                self.left_padding_list = [a + b for a, b in zip(mirror, added)]
            self._right_padding = None
            self._right_padding_list = None
"""

MASK_OLD = """    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )
"""
MASK_NEW = """    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        if (
            _KV_MASKLESS_UNPADDED
            and N == 1
            and not return_array
            and kwargs.get("window_size") is None
            and self.left_padding_list is not None
            and not any(self.left_padding_list)
        ):
            return None
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )
"""

FILTER_OLD = """        min_left_pad = min(self.left_padding.tolist())
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad
"""
FILTER_NEW = """        mirror = self.left_padding.tolist()
        self.left_padding_list = mirror
        min_left_pad = min(mirror)
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad
            self.left_padding_list = [p - min_left_pad for p in mirror]
"""

EXTEND_EMPTY_OLD = """            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return
"""
EXTEND_EMPTY_NEW = """            mirrors = (self.left_padding_list, other.left_padding_list)
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            if None not in mirrors:
                self.left_padding_list = mirrors[0] + mirrors[1]
            self.offset = mx.concatenate([self.offset, other.offset])
            return
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
    sys.exit("BatchKVCache not found in " + q)
body = text[start:end]
for old, new in [(CLASS_OLD, CLASS_NEW), (INIT_OLD, INIT_NEW), (PREPARE_OLD, PREPARE_NEW),
                 (FINALIZE_OLD, FINALIZE_NEW), (MASK_OLD, MASK_NEW), (FILTER_OLD, FILTER_NEW),
                 (EXTEND_EMPTY_OLD, EXTEND_EMPTY_NEW), (EXTEND_OLD, EXTEND_NEW)]:
    if body.count(old) != 1:
        sys.exit("unrecognized BatchKVCache content in " + q + "; refusing to patch "
                 "(mlx-lm version mismatch?)")
    body = body.replace(old, new, 1)
text = text[:start] + body + text[end:]
ast.parse(text)
open(q, "w").write(text)
print("patched:", q)
