#!/usr/bin/env python3
"""Route the Qwen3.5 / Qwen3.8 GDN decode step to mx.fast.gdn_conv_delta_update
(conv, SiLU, q/k norm and the gated-delta step as one dispatch).

Idempotent patch for mlx-lm 0.31.3 venvs, mirroring
patch-mlx-lm-qwen35-gdn-conv.py. The new branch runs when
MLX_OMARCHY_GDN_CONV_DELTA is 1 (conv + delta) or 2 (the gated norm folded in too), or when the
variable is unset and the backend's per-chip default says so (omarchy::gdn_conv_delta_norm_default_part, read
through the C ABI mlx_omarchy_gdn_conv_delta_default: 0 = the stock route, 1 = conv + delta, 2 = conv + delta and the
gated norm). A wheel without that symbol keeps the stock route. The branch also needs the attribute to exist on
mx.fast, and the step to be a plain bf16 decode (T == 1, no mask, no cache lengths, an existing recurrent state, no
tensor sharding, GPU default device). Every other case takes the composed path unchanged, and so does the op itself
wherever it cannot fuse. Apply it after the conv, norm and raw-gates patches: its anchor exists with or without them.
Usage: python3 patch-mlx-lm-qwen35-gdn-conv-delta.py /path/to/venv
"""
import glob
import sys

# The per-chip default lives in the backend (omarchy::gdn_conv_delta_norm_default_part): the C ABI answers with the mode for this device,
# 0 = the stock route, 1 = conv + delta fused, 2 = conv + delta + gated norm fused. A wheel without the symbol keeps the stock route.
RESOLVER = '''

def _gdn_conv_delta_default_mode():
    """The mode the fused GatedDelta op runs in when MLX_OMARCHY_GDN_CONV_DELTA is unset: "0", "1" or "2"."""
    try:
        import ctypes

        import mlx.core as mx

        base = os.path.dirname(mx.__file__)
        for candidate in (os.path.join(base, "lib", "libmlx.so"), mx.__file__):
            try:
                lib = ctypes.CDLL(candidate)
                lib.mlx_omarchy_gdn_conv_delta_default.restype = ctypes.c_int
                mode = int(lib.mlx_omarchy_gdn_conv_delta_default())
                return {0: "0", 1: "1", 2: "2"}.get(mode, "0")
            except (OSError, AttributeError):
                continue
    except Exception:
        pass
    return "0"


_GDN_CONV_DELTA_DEFAULT = _gdn_conv_delta_default_mode()
'''

IMPORT_OLD = "from dataclasses import dataclass, field\n"
IMPORT_NEW = "import os\nfrom dataclasses import dataclass, field\n" + RESOLVER
# The import block an earlier version of this patch wrote: an upgrade has to add the resolver there.
IMPORT_PATCHED_OLD = "import os\nfrom dataclasses import dataclass, field\n"

BRANCH_ANCHOR = """        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
"""

BRANCH_BODY = """        gdn_fuse = os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA") or _GDN_CONV_DELTA_DEFAULT
        if (
            gdn_fuse in ("1", "2")
            and mask is None
            and S == 1
            and cache is not None
            and cache.lengths is None
            and cache[1] is not None
            and qkv.dtype == mx.bfloat16
            and self.sharding_group is None
            and mx.default_device() == mx.gpu
            and hasattr(mx.fast, "gdn_conv_delta_update")
        ):
            inv_scale = self.head_k_dim**-0.5
            out, state, new_conv_state = mx.fast.gdn_conv_delta_update(
                conv_state,
                qkv,
                self.conv1d.weight,
                a,
                b,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.key_dim,
                inv_scale**2,
                inv_scale,
                1e-6,
                *((z, self.norm.weight, self.norm.eps) if gdn_fuse == "2" else ()),
            )
            cache[0] = new_conv_state
            cache[1] = state
            cache.advance(S)
            if gdn_fuse != "2":
                out = self.norm(out, z)
            return self.out_proj(out.reshape(B, S, -1))
"""

# The branch an earlier version of this patch wrote: an upgrade replaces it with the one above.
BRANCH_PATCHED_OLD = """        gdn_fuse = os.environ.get("MLX_OMARCHY_GDN_CONV_DELTA")
        if (
            gdn_fuse in ("1", "2")
            and mask is None
            and S == 1
            and cache is not None
            and cache.lengths is None
            and cache[1] is not None
            and qkv.dtype == mx.bfloat16
            and self.sharding_group is None
            and mx.default_device() == mx.gpu
            and hasattr(mx.fast, "gdn_conv_delta_update")
        ):
            inv_scale = self.head_k_dim**-0.5
            out, state, new_conv_state = mx.fast.gdn_conv_delta_update(
                conv_state,
                qkv,
                self.conv1d.weight,
                a,
                b,
                self.A_log,
                self.dt_bias,
                cache[1],
                self.key_dim,
                inv_scale**2,
                inv_scale,
                1e-6,
                *((z, self.norm.weight, self.norm.eps) if gdn_fuse == "2" else ()),
            )
            cache[0] = new_conv_state
            cache[1] = state
            cache.advance(S)
            if gdn_fuse != "2":
                out = self.norm(out, z)
            return self.out_proj(out.reshape(B, S, -1))
"""

BRANCH_NEW = BRANCH_BODY + BRANCH_ANCHOR


def patch_file(path, pairs, marker):
    text = open(path).read()
    if marker in text:
        print("already patched:", path)
        return
    for old, _ in pairs:
        if text.count(old) != 1:
            sys.exit("unrecognized content in " + path + "; refusing to patch")
    for old, new in pairs:
        text = text.replace(old, new)
    open(path, "w").write(text)
    print("patched:", path)


def main(argv):
    venv = argv[1] if len(argv) > 1 else "."
    site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
    if not site:
        sys.exit("mlx_lm/models not found under " + venv)
    # qwen3_5.py only: the Qwen3.5 / Qwen3.8 GatedDeltaNet lives here.
    path = site[0] + "/qwen3_5.py"
    text = open(path).read()
    if "_GDN_CONV_DELTA_DEFAULT" in text:
        print("already patched:", path)
        return
    if "gdn_conv_delta_update" in text:
        # A file an earlier version of this patch wrote: replace its import block and its branch.
        pairs = [(IMPORT_PATCHED_OLD, IMPORT_NEW), (BRANCH_PATCHED_OLD, BRANCH_NEW)]
    else:
        pairs = [(IMPORT_OLD, IMPORT_NEW), (BRANCH_ANCHOR, BRANCH_NEW)]
    patch_file(path, pairs, "_GDN_CONV_DELTA_DEFAULT")


if __name__ == "__main__":
    main(sys.argv)
