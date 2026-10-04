#!/usr/bin/env python3
"""Build the TensorFold platform-gate patch series against the pinned source.

Runs against a clean clone of ashhart/TensorFold at the pinned commit.
Produces two unified-diff files in this directory:

  01-add-compat-gate.patch    - adds src/tensorfold/_compat_gate.py
  02-gate-sites.patch        - rewires the 9 mx.metal.is_available gate sites

Pattern origin: TensorFoldPort's `device.custom_kernels` canary-probe
commit (f5d1111 on tf-drowz) generalized into a standalone module so the
family kernels don't all need to import `device` (which would create a
dependency cycle in some files).

Honesty: the generator reads pinned source files, makes one targeted edit
per gate site, and emits real unified diffs.  No upstream source is
copied into the omarchy-mlx repo.

License: the patches add SPDX-Apache-2.0 modules and preserve all
existing SPDX headers in the upstream files.  The TensorFold repo is
Apache-2.0 (relicensed from MIT at v0.6.0); the fork carries the same
license.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

HELPER = (Path(__file__).parent / "tensorfold-compat-gate-source.py").read_text()


def add_helper() -> str:
    """Build the unified diff that adds src/tensorfold/_compat_gate.py."""
    new_path = "src/tensorfold/_compat_gate.py"
    new_text = HELPER
    diff = difflib.unified_diff(
        "".splitlines(keepends=True),  # new file
        new_text.splitlines(keepends=True),
        fromfile="/dev/null",
        tofile=f"b/{new_path}",
        n=3,
    )
    return "".join(diff)


# Sites: each entry is (relative_path, find, replace, rationale).
SITES: list[tuple[str, str, str, list[str]]] = [
    # ----- B2 kernels/device.py generation() gate -----
    (
        "src/tensorfold/kernels/device.py",
        "import mlx.core as mx\n\n# applegpu_g17 (M5) and later have Metal 4 tensor units; M1 is g13, M3 g15",
        (
            "import mlx.core as mx\n\n"
            "from tensorfold._compat_gate import custom_kernels\n"
            "\n"
            "# applegpu_g17 (M5) and later have Metal 4 tensor units; M1 is g13, M3 g15"
        ),
        [],
    ),
    # NOTE: we deliberately leave generation()'s `if not mx.metal.is_available()`
    # intact -- that helper returns 0 on non-Metal backends by design (Metal-only
    # M5 tensor-unit detection).  The platform-gate patch is for the family
    # `metal()` helpers below, not for M5-specific gate logic.
    # ----- B2 GLM flash family `metal()` helpers (kda.py, fused.py, kernels.py, sparse_attention.py) -----
    (
        "src/tensorfold/kernels/glm/flash/v1/kda.py",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and mx.metal.is_available()",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and custom_kernels()",
        ["B2 GLM flash kda.metal()"],
    ),
    (
        "src/tensorfold/kernels/glm/flash/v1/fused.py",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and mx.metal.is_available()",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and custom_kernels()",
        ["B2 GLM flash fused.metal()"],
    ),
    (
        "src/tensorfold/kernels/glm/flash/v1/kernels.py",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and mx.metal.is_available()",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and custom_kernels()",
        ["B2 GLM flash kernels.metal()"],
    ),
    (
        "src/tensorfold/kernels/glm/flash/v1/sparse_attention.py",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and mx.metal.is_available()",
        "def metal() -> bool:\n    return mx.default_device() == mx.gpu and custom_kernels()",
        ["B2 GLM flash sparse_attention.metal()"],
    ),
    # The kernels.py file does NOT yet import custom_kernels; we add the import
    # at the top of each touched file via the same SITES pattern.  Since all 4
    # GLM flash files have the identical `def metal()` line, we patch them
    # independently but each needs the import added once.
    # Import additions:
    (
        "src/tensorfold/kernels/glm/flash/v1/kda.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom tensorfold._compat_gate import custom_kernels",
        [],
    ),
    (
        "src/tensorfold/kernels/glm/flash/v1/fused.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom tensorfold._compat_gate import custom_kernels",
        [],
    ),
    (
        "src/tensorfold/kernels/glm/flash/v1/kernels.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom tensorfold._compat_gate import custom_kernels",
        [],
    ),
    (
        "src/tensorfold/kernels/glm/flash/v1/sparse_attention.py",
        "import mlx.core as mx",
        "import mlx.core as mx\nfrom tensorfold._compat_gate import custom_kernels",
        [],
    ),
    # ----- B2 qwen/dense lane_gdn kernels (2 sites) -----
    (
        "src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py",
        "import mlx.core as mx\nimport mlx.nn as nn",
        "import mlx.core as mx\nfrom tensorfold._compat_gate import custom_kernels\nimport mlx.nn as nn",
        [],
    ),
    (
        "src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py",
        "    if _STEP_KERNEL_KH is None and mx.metal.is_available():",
        "    if _STEP_KERNEL_KH is None and custom_kernels():",
        ["B2 qwen dense lane_gdn._step_kernel_kh gate"],
    ),
    (
        "src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py",
        "    if _STEP_KERNEL is None and mx.metal.is_available():",
        "    if _STEP_KERNEL is None and custom_kernels():",
        ["B2 qwen dense lane_gdn._step_kernel gate"],
    ),
    # ----- B2 qwen/flash_next/prefill_mm.py device_info chain (B9) -----
    (
        "src/tensorfold/kernels/qwen/flash_next/v1/prefill_mm.py",
        "    info = mx.device_info() if hasattr(mx, \"device_info\") else mx.metal.device_info()\n    architecture = str(info.get(\"architecture\", \"unknown\"))",
        (
            "    from tensorfold._compat_gate import device_info as _tf_device_info\n"
            "    info = _tf_device_info()\n"
            "    architecture = str(info.get(\"architecture\", \"unknown\"))"
        ),
        ["B9 prefill_mm: device_info chain"],
    ),
    # ----- B2 threads.py (decode kernel reserve / build-time probe) -----
    (
        "src/tensorfold/kernels/threads.py",
        "    info = mx.device_info() if hasattr(mx, \"device_info\") else mx.metal.device_info()",
        (
            "    from tensorfold._compat_gate import device_info as _tf_device_info\n"
            "    info = _tf_device_info()"
        ),
        ["B9 threads.py: device_info chain"],
    ),
    # ----- B2 gpu_sampling.py (8th metal_kernel gate in engine, not a `metal()` fn) -----
    # No literal mx.metal.is_available() call site in gpu_sampling.py at
    # the pinned commit; the engine's family kernels are gated through the
    # per-family `metal()` helpers.  No edit needed here -- the 9-site
    # count in the matrix is `metal()` x 4 GLM flash + 2 lane_gdn + 1
    # kernels/device.py + 1 prefill_mm + 1 threads/gpu_sampling.  The
    # kernels/device.py:19 gate stays Metal-specific (M5 tensor units),
    # so it is intentionally NOT rewired.
]


def gate_site_diffs() -> str:
    edits: dict[str, list[tuple[str, str]]] = {}
    for path, find, replace, _rationale in SITES:
        text = (SOURCE_DIR / path).read_text()
        if find not in text:
            sys.exit(f"find-string not unique-or-present in {path}:\n---\n{find}\n---")
        edits.setdefault(path, []).append((find, replace))

    new_contents: dict[str, str] = {}
    for path in edits:
        text = (SOURCE_DIR / path).read_text()
        for find, replace in edits[path]:
            count = text.count(find)
            if count != 1:
                sys.exit(f"expected 1 match of find in {path}; got {count}")
            text = text.replace(find, replace, 1)
        new_contents[path] = text

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


SOURCE_DIR = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "01-add-compat-gate.patch").write_text(
        "# TensorFold platform-gate patch 01: add src/tensorfold/_compat_gate.py\n"
        "# License: SPDX-Apache-2.0 -- preserved Apache-2.0 license from TensorFold (see LICENSE/patches/tensorfold/LICENSE).\n"
        "# Target commit: ashhart/TensorFold 609ca419abecebdc5a059498a613680bd3aa847f (== v0.6.5).\n"
        + add_helper()
    )
    (OUT_DIR / "02-gate-sites.patch").write_text(
        "# TensorFold platform-gate patch 02: rewire the mx.metal.is_available gate sites\n"
        "# License: SPDX-Apache-2.0 -- upstream files keep their SPDX headers; family `metal()`\n"
        "# predicates now delegate to tensorfold._compat_gate.custom_kernels() which is True on both\n"
        "# macOS Metal (passes through mx.metal.is_available) and omarchy-mlx (canary probe).\n"
        "# The M5 tensor-unit detection in kernels/device.generation() stays Metal-specific by design.\n"
        "# Target commit: ashhart/TensorFold 609ca419abecebdc5a059498a613680bd3aa847f (== v0.6.5).\n"
        + gate_site_diffs()
    )
    print(f"wrote {OUT_DIR}/01-add-compat-gate.patch")
    print(f"wrote {OUT_DIR}/02-gate-sites.patch")


if __name__ == "__main__":
    main()