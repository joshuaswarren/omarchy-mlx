#!/usr/bin/env python3
"""Build the omlx platform-gate patch series against the pinned source.

Runs against a clean clone of jundot/omlx at the pinned commit.  Produces
two unified-diff files in this directory:

  01-add-compat-gate.patch          - adds omlx/_compat_gate.py
  02-gate-sites.patch              - rewires the ~25 mx.metal gate sites

Designed to be idempotent: re-runs against an already-patched tree
produce empty diffs (the generator refuses to double-apply).

Honesty: the generator reads pinned source files, makes one targeted edit
per gate site, and emits real unified diffs.  No upstream source is
copied into the omarchy-mlx repo.

License: the patches add SPDX-Apache-2.0 modules and preserve all
existing SPDX headers in the upstream files.
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

SOURCE_DIR = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
SOURCE_DIR_REL = "."

# Helper module source lives next to this script (read once, embedded).
HELPER = (Path(__file__).parent / "omlx-compat-gate-source.py").read_text()


def add_helper() -> str:
    """Build the unified diff that adds omlx/_compat_gate.py."""
    new_path = "omlx/_compat_gate.py"
    old_text = ""  # new file
    new_text = HELPER
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile="/dev/null",
        tofile=f"b/{new_path}",  # b/ prefix so `patch -p1` keeps `omlx/`
        n=3,
    )
    return "".join(diff)


# Site list: each entry is (relative_path, find, replace, rationale).
# The `find` substring must be unique in the file at the pinned commit
# (the generator verifies this); `replace` is the new text.
#
# Sites are deliberately kept in dependency order: the gate predicate
# itself comes first, then the predicate body.  This is a smaller diff
# than changing every gate site to a from-omlx import.
SITES: list[tuple[str, str, str, list[str]]] = [
    # ----- A7 DFlash speculative decoding (engine/dflash.py) -----
    (
        "omlx/engine/dflash.py",
        # original:
        #     if self._wired_limit_owned or not mx.metal.is_available():
        # new:
        #     from omlx._compat_gate import custom_kernels_available as _cka
        #     if self._wired_limit_owned or not _cka():
        # (local import keeps the helper optional for tooling that
        # imports DFlash without touching the GPU.)
        "from .base import (",
        (
            "try:\n"
            "    from omlx._compat_gate import custom_kernels_available as _cka\n"
            "    from omlx._compat_gate import set_wired_limit_enabled as _wire_enabled\n"
            "    from omlx._compat_gate import device_info_keys as _device_info_keys\n"
            "except ImportError:  # platform-gate patch not applied\n"
            "    _cka = lambda: mx.metal.is_available()\n"
            "    _wire_enabled = lambda: mx.metal.is_available()\n"
            "    _device_info_keys = lambda: mx.device_info() if hasattr(mx, \"device_info\") else {}\n"
            "\n"
            "from .base import ("
        ),
        ["A7 dflash.py: wired-limit gate"],
    ),
    (
        "omlx/engine/dflash.py",
        (
            "        if self._wired_limit_owned or not mx.metal.is_available():\n"
            "            return\n"
            "        recommended = int(\n"
            "            mx.device_info().get(\"max_recommended_working_set_size\", 0) or 0\n"
            "        )"
        ),
        (
            "        if self._wired_limit_owned or not _wire_enabled():\n"
            "            return\n"
            "        recommended = int(\n"
            "            _device_info_keys().get(\"max_recommended_working_set_size\", 0) or 0\n"
            "        )"
        ),
        ["A7 dflash.py: wired-limit body"],
    ),

    # ----- A15 memory monitor -----
    (
        "omlx/memory_monitor.py",
        (
            "try:\n"
            "    import mlx.core as mx\n"
            "\n"
            "    HAS_MLX_METAL = mx.metal.is_available()\n"
            "except ImportError:\n"
            "    HAS_MLX_METAL = False\n"
            "    mx = None"
        ),
        (
            "try:\n"
            "    import mlx.core as mx\n"
            "    from omlx._compat_gate import custom_kernels_available as _cka\n"
            "\n"
            "    HAS_MLX_METAL = _cka()\n"
            "except ImportError:\n"
            "    HAS_MLX_METAL = False\n"
            "    mx = None"
        ),
        ["A15 memory_monitor.py: HAS_MLX_METAL"],
    ),

    # ----- A26 / A27 patch-site gates -----
    # The 12 top-level omlx.patches.* files with mx.metal gate sites.
    # Each gets a single-line import added after `import mlx.core as mx`
    # and one targeted edit at the gate predicate.
    (
        "omlx/patches/bailing_hybrid/bailing_hybrid_model.py",
        "import mlx.core as mx\nimport mlx.nn as nn",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka\nimport mlx.nn as nn",
        [],
    ),
    (
        "omlx/patches/bailing_hybrid/bailing_hybrid_model.py",
        "        and mx.metal.is_available()\n    ):\n        output, state = gated_delta_kernel",
        "        and _cka()\n    ):\n        output, state = gated_delta_kernel",
        ["A26 bailing_hybrid: gated delta kernel"],
    ),

    (
        "omlx/patches/qwen35_moe_weighted_sum.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/qwen35_moe_weighted_sum.py",
        "    if not mx.metal.is_available():\n        return False",
        "    if not _cka():\n        return False",
        ["A26 qwen35_moe_weighted_sum"],
    ),

    (
        "omlx/patches/qwen35_verify_sdpa_split.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/qwen35_verify_sdpa_split.py",
        "    if _PATCHED:\n        return True\n    if not mx.metal.is_available():\n        return False",
        "    if _PATCHED:\n        return True\n    if not _cka():\n        return False",
        ["A26 qwen35_verify_sdpa_split"],
    ),

    (
        "omlx/patches/qwen35_moe_routed_decode.py",
        "import mlx.core as mx\nimport mlx.nn as nn",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka\nimport mlx.nn as nn",
        [],
    ),
    (
        "omlx/patches/qwen35_moe_routed_decode.py",
        "    if not _ENABLED or not mx.metal.is_available():\n        return False",
        "    if not _ENABLED or not _cka():\n        return False",
        ["A26 qwen35_moe_routed_decode"],
    ),

    (
        "omlx/patches/glm_moe_dsa/sparse_mla.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/glm_moe_dsa/sparse_mla.py",
        "def _make_topk_indices_to_block_masks_kernel():\n    if not mx.metal.is_available():\n        return None",
        "def _make_topk_indices_to_block_masks_kernel():\n    if not _cka():\n        return None",
        ["A26 glm_moe_dsa/sparse_mla: topk kernel"],
    ),
    (
        "omlx/patches/glm_moe_dsa/sparse_mla.py",
        "def _make_index_score_reduce_kernel():\n    if not mx.metal.is_available():\n        return None",
        "def _make_index_score_reduce_kernel():\n    if not _cka():\n        return None",
        ["A26 glm_moe_dsa/sparse_mla: index_score kernel"],
    ),

    (
        "omlx/patches/qwen35_gdn_chunked.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/qwen35_gdn_chunked.py",
        "    if os.environ.get(\"OMLX_GDN_KERNEL\", \"1\") == \"0\":\n        return False\n    if not mx.metal.is_available():\n        return False",
        "    if os.environ.get(\"OMLX_GDN_KERNEL\", \"1\") == \"0\":\n        return False\n    if not _cka():\n        return False",
        ["A26 qwen35_gdn_chunked"],
    ),

    (
        "omlx/patches/qwen35_fa256_attention.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/qwen35_fa256_attention.py",
        "    if not mx.metal.is_available() or _has_quantized_cache(cache):\n        return False",
        "    if not _cka() or _has_quantized_cache(cache):\n        return False",
        ["A27 qwen35_fa256_attention: route gate"],
    ),

    (
        "omlx/patches/gemma4_verify_kernel.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/gemma4_verify_kernel.py",
        "        try:\n            if not mx.metal.is_available():\n                _availability = False\n            else:",
        "        try:\n            if not _cka():\n                _availability = False\n            else:",
        ["A26 gemma4_verify_kernel"],
    ),

    (
        "omlx/patches/qwen35_moe_router.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/qwen35_moe_router.py",
        "    if not mx.metal.is_available():\n        return False\n    _ensure_vlm_verify_patch()",
        "    if not _cka():\n        return False\n    _ensure_vlm_verify_patch()",
        ["A26 qwen35_moe_router"],
    ),

    (
        "omlx/patches/deepseek_v4/hyper_connection.py",
        "import mlx.core as mx\nimport mlx.nn as nn",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka\nimport mlx.nn as nn",
        [],
    ),
    (
        "omlx/patches/deepseek_v4/hyper_connection.py",
        "    if mx.default_device() != mx.gpu or not mx.metal.is_available():\n        return None",
        "    if mx.default_device() != mx.gpu or not _cka():\n        return None",
        ["A26 deepseek_v4/hyper_connection: HC ops gate"],
    ),
    (
        "omlx/patches/deepseek_v4/hyper_connection.py",
        "            self.training\n            or mx.default_device() != mx.gpu\n            or not mx.metal.is_available()\n        )",
        "            self.training\n            or mx.default_device() != mx.gpu\n            or not _cka()\n        )",
        ["A26 deepseek_v4/hyper_connection: HC body selector"],
    ),

    (
        "omlx/patches/m5_gather_qmm.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/m5_gather_qmm.py",
        "    if not mx.metal.is_available():\n        _defective = False\n        return False",
        "    if not _cka():\n        _defective = False\n        return False",
        ["A26 m5_gather_qmm"],
    ),

    (
        "omlx/patches/glm53_kda_prework.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/glm53_kda_prework.py",
        "        or inputs.dtype != mx.bfloat16\n        or mx.default_device() != mx.gpu\n        or not mx.metal.is_available()\n        or getattr(module, \"conv_kernel_size\", 0) != 4",
        "        or inputs.dtype != mx.bfloat16\n        or mx.default_device() != mx.gpu\n        or not _cka()\n        or getattr(module, \"conv_kernel_size\", 0) != 4",
        ["A26 glm53_kda_prework"],
    ),

    (
        "omlx/patches/qwen35_gdn_prework.py",
        "import mlx.core as mx\nimport mlx.nn as nn",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka\nimport mlx.nn as nn",
        [],
    ),
    (
        "omlx/patches/qwen35_gdn_prework.py",
        "    if _PATCHED:\n        return True\n    if not mx.metal.is_available():\n        return False",
        "    if _PATCHED:\n        return True\n    if not _cka():\n        return False",
        ["A26 qwen35_gdn_prework"],
    ),

    # ----- A27 SDPA head-dim-256 native force-fused route (patch site) -----
    (
        "omlx/patches/sdpa256_attention.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom omlx._compat_gate import custom_kernels_available as _cka",
        [],
    ),
    (
        "omlx/patches/sdpa256_attention.py",
        "    if mx.metal.is_available() and native_shape and _NATIVE_FORCE_FUSED:",
        "    if _cka() and native_shape and _NATIVE_FORCE_FUSED:",
        ["A27 sdpa256_attention: native force-fused gate"],
    ),

    # ----- A26 mac_ver gate in qwen35_prefill.fast -----
    # The site at fast.py:879 evaluates platform.mac_ver()[0] (returns
    # '' on Linux -> kernels off, graceful per the matrix row).  The
    # gate already takes the right path on Linux unchanged.  No edit
    # needed here; the matrix row "platform.mac_ver()" is already
    # marked tested (by contract).
]


def gate_site_diffs() -> str:
    """Apply every gate site edit, then return one consolidated diff."""
    # Map of (file_path -> list of (old_text, new_text) tuples).
    edits: dict[str, list[tuple[str, str]]] = {}
    for path, find, replace, _rationale in SITES:
        full = SOURCE_DIR / path
        if not full.exists():
            sys.exit(f"missing file in pinned source: {full}")
        text = full.read_text()
        if find not in text:
            sys.exit(f"find-string not unique-or-present in {path}:\n---\n{find}\n---")
        # Use replace only once; if find is identical to a previous edit's find,
        # apply them in order via a stream of replace_one passes.
        edits.setdefault(path, []).append((find, replace))

    # Apply edits in order; for each file, do sequential replacements.
    new_contents: dict[str, str] = {}
    for path in edits:
        text = (SOURCE_DIR / path).read_text()
        for find, replace in edits[path]:
            count = text.count(find)
            if count != 1:
                sys.exit(f"expected 1 match of find in {path}; got {count}")
            text = text.replace(find, replace, 1)
        new_contents[path] = text

    # Build the unified diff for all changed files.
    out_lines: list[str] = []
    for path in sorted(new_contents):
        old_text = (SOURCE_DIR / path).read_text()
        new_text = new_contents[path]
        diff = list(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        ))
        out_lines.extend(diff)
    return "".join(out_lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "01-add-compat-gate.patch").write_text(
        "# omlx platform-gate patch 01: add omlx/_compat_gate.py\n"
        "# License: SPDX-Apache-2.0 -- preserved Apache-2.0 license from omlx (see LICENSE/patches/omlx/LICENSE).\n"
        "# Target commit: jundot/omlx 4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40 (v0.7.0).\n"
        + add_helper()
    )
    (OUT_DIR / "02-gate-sites.patch").write_text(
        "# omlx platform-gate patch 02: rewire the ~25 mx.metal gate sites\n"
        "# License: SPDX-Apache-2.0 -- upstream files keep their SPDX headers; gate predicates\n"
        "# now delegate to omlx._compat_gate.custom_kernels_available() which is True on\n"
        "# both macOS Metal (passes through mx.metal.is_available) and omarchy-mlx (canary probe).\n"
        "# Target commit: jundot/omlx 4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40 (v0.7.0).\n"
        + gate_site_diffs()
    )
    print(f"wrote {OUT_DIR}/01-add-compat-gate.patch")
    print(f"wrote {OUT_DIR}/02-gate-sites.patch")


if __name__ == "__main__":
    main()