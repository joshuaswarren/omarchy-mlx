#!/usr/bin/env python3
"""Skip trivially all-valid batch SSM masks (serve-path GDN prefill fallback fix).

Root cause (QualityPrefill/QualityPrefill2, receipts/2026-10-02-quality-prefill):
the assistant shim's speech-yield gate pins generation to the mlx-lm
BatchGenerator route. Its PromptProcessingBatch merges fresh per-sequence
caches through ArraysCache.merge, which sets left_padding = mx.array([0]*B) on
the batch ArraysCache even when nothing is padded. ArraysCache.make_mask then
returns a REAL all-True (B, S) validity mask for every SSM layer, and that mask
rides into mx.fast.gated_delta_update, failing the backend's maskless gate for
the fused coopmat GDN prefill kernel (primitives.cpp: gdn_coopmat requires
!has_mask) and forcing the two-pass scan fallback (~11x slower per layer, 42%
of serve-path GPU time on the 27B).

The all-True mask is arithmetically a no-op: the model applies it to qkv as an
identity where-select, and the gated-delta update treats every token as valid.
So make_mask returns None when every host-known left_padding is <= 0 AND
lengths is unset (no right padding happened): the mask is pos >= left_padding,
all True for pos >= 0. advance(N) lowers left_padding by N every step, so an
unpadded batch reads 0 only before its first token and negative after; a
zeros-only test missed every decode step (2026-10-07, B=4 BatchGenerator: 18
GDN fallbacks and host joins per step). Genuinely padded batches keep the real
mask and the masked fallback path. The host knowledge rides a Python
left_padding_list that mirrors the mx array at every mutation site; foreign
state restores leave the mirror unset, which disables the guard (old behavior,
safe default).

Idempotent for mlx-lm 0.31.3 venvs and the 0.32 line. Kill switch at runtime:
MLX_OMARCHY_SSM_MASKLESS=0 (read once at import). Usage:
python3 patch-mlx-lm-ssm-maskless.py /path/to/venv
"""
import glob
import sys

MARKER = "MLX_OMARCHY_SSM_MASKLESS"

FLAG_BLOCK = """# A batch ArraysCache with no padded row carries left_padding <= 0 (0 at
# merge, lowered by advance) and no lengths; its make_mask is then an all-True
# validity mask. Consumers apply it
# as identity (qkv zeroing skips nothing; the gated-delta update treats every
# token as valid), so returning None is arithmetic-preserving and lets the
# maskless fast kernels (fused GDN decode and coopmat prefill) run on the
# batched serve path. Genuinely padded batches keep the real mask.
# MLX_OMARCHY_SSM_MASKLESS=0 restores the masked behavior.
_SSM_MASKLESS_UNPADDED = os.environ.get("MLX_OMARCHY_SSM_MASKLESS", "1") != "0"


class ArraysCache(_BaseCache):"""

GUARD_0313 = """    def make_mask(self, N: int):
        if (
            _SSM_MASKLESS_UNPADDED
            and self.left_padding is not None
            and self.lengths is None
            and self.left_padding_list is not None
            and max(self.left_padding_list) <= 0
        ):
            return None
        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
"""

GUARD_032 = """    def make_mask(self, N: int):
        if (
            _SSM_MASKLESS_UNPADDED
            and self.left_padding is not None
            and self.lengths is None
            and self.left_padding_list is not None
            and max(self.left_padding_list) <= 0
        ):
            return None
        pos = mx.arange(N)
        mask = None
"""

INIT_0313_OLD = """    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            self.left_padding = mx.array(left_padding)
"""
INIT_0313_NEW = """    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.left_padding = None
        instance.lengths = None
        instance.left_padding_list = None
        return instance

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        if left_padding:
            self.left_padding = mx.array(left_padding)
            self.left_padding_list = list(left_padding)
"""

INIT_032_OLD = """    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        self.left_padding = mx.array(left_padding) if left_padding else None
        self.lengths = None
"""
INIT_032_NEW = """    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        self.left_padding = mx.array(left_padding) if left_padding else None
        self.lengths = None
        self.left_padding_list = list(left_padding) if left_padding else None
"""

# 0.32's state setter round-trips left_padding as an mx array; reset the
# mirror so a foreign state restore disables the guard (safe default).
STATE_032_OLD = """    @state.setter
    def state(self, v):
        self.cache, self.left_padding, self.lengths = v
"""
STATE_032_NEW = """    @state.setter
    def state(self, v):
        self.cache, self.left_padding, self.lengths = v
        self.left_padding_list = None
"""

FILTER_OLD = """        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]
"""
FILTER_NEW = """        if self.left_padding is not None:
            self.left_padding = self.left_padding[batch_indices]
            if self.left_padding_list is not None:
                self.left_padding_list = [
                    self.left_padding_list[i] for i in batch_indices
                ]
        if self.lengths is not None:
            self.lengths = self.lengths[batch_indices]
"""

EXTEND_OLD = """        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self.left_padding = cat(self.left_padding, other.left_padding)
        self.lengths = cat(self.lengths, other.lengths)
"""
EXTEND_NEW = """        self.cache = [cat(c, o) for c, o in zip(self.cache, other.cache)]
        self.left_padding = cat(self.left_padding, other.left_padding)
        if self.left_padding is not None or other.left_padding is not None:
            a_list = (
                self.left_padding_list
                if self.left_padding_list is not None
                else [0] * self.batch_size
            )
            b_list = (
                other.left_padding_list
                if other.left_padding_list is not None
                else [0] * other.batch_size
            )
            self.left_padding_list = a_list + b_list
        else:
            self.left_padding_list = None
        self.lengths = cat(self.lengths, other.lengths)
"""

FINALIZE_ADVANCE_OLD = """    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, N):
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N
"""
FINALIZE_ADVANCE_NEW = """    def finalize(self):
        self.lengths = None
        self.left_padding = None
        self.left_padding_list = None

    def advance(self, N):
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N
            if self.left_padding_list is not None:
                self.left_padding_list = [p - N for p in self.left_padding_list]
"""

MERGE_OLD = """        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            return cache
"""
MERGE_NEW = """        if all(c.empty() for c in caches):
            cache.left_padding = mx.array([0] * B)
            cache.left_padding_list = [0] * B
            return cache
"""

MAKE_MASK_0313_OLD = """    def make_mask(self, N: int):
        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
"""
MAKE_MASK_032_OLD = """    def make_mask(self, N: int):
        pos = mx.arange(N)
        mask = None
"""

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
q = site[0] + "/cache.py"
text = open(q).read()
# Trees patched before 2026-10-07 carry the zeros-only guard; upgrade them in place.
ZEROS_GUARD = ("            and self.lengths is None\n"
               "            and self.left_padding_list is not None\n"
               "            and not any(self.left_padding_list)\n")
if MARKER in text:
    if ZEROS_GUARD in text:
        open(q, "w").write(text.replace(ZEROS_GUARD, ZEROS_GUARD.replace(
            "not any(self.left_padding_list)", "max(self.left_padding_list) <= 0"), 1))
        print("upgraded guard:", q)
    else:
        print("already patched:", q)
    sys.exit(0)

is_032 = MAKE_MASK_032_OLD in text
pairs = [
    ("import copy\nfrom collections import deque",
     "import copy\nimport os\nfrom collections import deque"),
    ("class ArraysCache(_BaseCache):", FLAG_BLOCK),
    (INIT_032_OLD if is_032 else INIT_0313_OLD,
     INIT_032_NEW if is_032 else INIT_0313_NEW),
    (FILTER_OLD, FILTER_NEW),
    (EXTEND_OLD, EXTEND_NEW),
    (FINALIZE_ADVANCE_OLD, FINALIZE_ADVANCE_NEW),
    (MERGE_OLD, MERGE_NEW),
    (MAKE_MASK_032_OLD if is_032 else MAKE_MASK_0313_OLD,
     GUARD_032 if is_032 else GUARD_0313),
]
if is_032:
    pairs.append((STATE_032_OLD, STATE_032_NEW))
for old, new in pairs:
    if old not in text:
        sys.exit("unrecognized content in " + q + "; refusing to patch "
                 "(mlx-lm version mismatch?)")
    text = text.replace(old, new, 1)
open(q, "w").write(text)
import ast
ast.parse(text)
print("patched:", q, "(mlx-lm", "0.32 line)" if is_032 else "0.31.3)")
