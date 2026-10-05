#!/usr/bin/env python3
"""Fold the qwen3 (dense) attention q/k RMSNorm into mx.fast.rope_rms_norm,
mirroring scripts/patch-mlx-lm-rope-norm.py (F3) which targets qwen3_next.py.

The fused branch is gated on MLX_OMARCHY_ROPE_NORM_FUSE (default 1 = on, landed 2026-10-04 at +4.18..+4.86% bit-exact; set =0 to run
exact eager chain runs; kill switch =0 at any time). Bit-identical by the
same contract as F3: the fused kernel reproduces the fast RMSNorm reduction
tree and rounds normalized values to bf16 before rotation consumes them,
the same rounding the composed norm -> rope chain does across the kernel
boundary. The backend wrapper refuses non-fuseable legs loudly; this site
only issues bf16, non-traditional, D=128 legs (qwen3-4B head_dim 128, plain
rope, no scaling). Usage: python3 patch-mlx-lm-qwen3-rope-norm.py /venv
"""
import glob
import sys

IMPORT_OLD = "from typing import Any, Dict, Optional, Union\n"
IMPORT_NEW = "from typing import Any, Dict, Optional, Union\nimport os\n"

NORM_OLD = """        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
"""

NORM_NEW = """        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # mlx-omarchy rope-norm patch (qwen3 dense): fold the per-head q/k
        # RMSNorm into the rope dispatch (mx.fast.rope_rms_norm). Kill
        # switch: MLX_OMARCHY_ROPE_NORM_FUSE=0 runs the exact eager chain
        # below (bit-identical by construction); the backend fence refuses
        # non-fuseable legs loudly.
        # SpecPrefill installs RoPE wrappers (_PositionMappedRoPE has no
        # attribute delegation; _OffsetAdjustedRoPE remaps offsets) whose
        # __call__ semantics the fused kernel cannot reproduce: it rotates
        # at the contiguous cache offset, not at the mapped or adjusted
        # positions the wrapper exists to provide. Any wrapped rope must
        # take the composed chain below. Reading self.rope.dims directly on
        # the wrapper raised AttributeError and crashed sparse prefill
        # (A8, 2026-10-05). Genuine stock ropes carry .dims and never
        # carry _original.
        if (
            # qwen3 dense rope-norm fold. Kill switches (any of these
            # disables the fold and falls through to the bit-identical
            # composed chain in the else branch):
            #   * MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE=0  -- this patch only
            #   * MLX_OMARCHY_ROPE_NORM_FUSE=0         -- qwen3 + qwen3_next
            # Array (per-request) cache offsets need no gate here: the op
            # fuses scalar offsets and routes vector offsets through its
            # internal composed fallback (fixed 2026-10-05; the v0.7.27
            # B==1/int-only fence is gone).
            (
                os.environ.get("MLX_OMARCHY_QWEN3_ROPE_NORM_FUSE",
                                os.environ.get("MLX_OMARCHY_ROPE_NORM_FUSE", "1"))
                == "1"
            )
            and queries.dtype == mx.bfloat16
            and hasattr(mx.fast, "rope_rms_norm")
            and getattr(self.rope, "dims", None) is not None
            and not hasattr(self.rope, "_original")
        ):
            offset_pos = cache.offset if cache is not None else 0
            queries = mx.fast.rope_rms_norm(
                queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3),
                self.rope.dims,
                self.q_norm.weight,
                self.q_norm.eps,
                traditional=self.rope.traditional,
                base=self.rope.base,
                scale=self.rope.scale,
                offset=offset_pos,
            )
            keys = mx.fast.rope_rms_norm(
                keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3),
                self.rope.dims,
                self.k_norm.weight,
                self.k_norm.eps,
                traditional=self.rope.traditional,
                base=self.rope.base,
                scale=self.rope.scale,
                offset=offset_pos,
            )
            values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
                0, 2, 1, 3
            )
            keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
                0, 2, 1, 3
            )
            values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
"""


# Migration for already-patched installs: the wrap-fallback gate added
# 2026-10-05 (A8). Files patched before it carry the gate without the two
# wrapper conditions; upgrade them in place (idempotent).
GATE_OLD = (
    '            and queries.dtype == mx.bfloat16\n'
    '            and hasattr(mx.fast, "rope_rms_norm")\n'
    '        ):'
)
GATE_NEW = (
    '            and queries.dtype == mx.bfloat16\n'
    '            and hasattr(mx.fast, "rope_rms_norm")\n'
    '            and getattr(self.rope, "dims", None) is not None\n'
    '            and not hasattr(self.rope, "_original")\n'
    '        ):'
)


def patch_file(path, old, new, marker):
    text = open(path).read()
    if marker in text:
        print("already patched:", path)
        return
    if old not in text:
        sys.exit("unrecognized content in " + path + "; refusing to patch")
    open(path, "w").write(text.replace(old, new, 1))
    print("patched:", path)


venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
site = site[0]

q = site + "/qwen3.py"
text = open(q).read()
if "MLX_OMARCHY_ROPE_NORM_FUSE" in text:
    if GATE_NEW in text:
        print("already patched:", q)
    elif text.count(GATE_OLD) == 1:
        open(q, "w").write(text.replace(GATE_OLD, GATE_NEW, 1))
        print("upgraded wrap-fallback gate:", q)
    else:
        sys.exit(
            "patched qwen3.py without the expected fused gate; refusing to "
            "upgrade " + q
        )
else:
    if IMPORT_NEW not in text:
        if text.count(IMPORT_OLD) != 1:
            sys.exit("unrecognized import block in " + q + "; refusing to patch")
        text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    if NORM_OLD not in text:
        sys.exit("unrecognized attention block in " + q + "; refusing to patch")
    text = text.replace(NORM_OLD, NORM_NEW, 1)
    open(q, "w").write(text)
    print("patched:", q)
