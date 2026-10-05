#!/usr/bin/env bash
# Compile the deployed GDN prefill shader and both HOIST passes with glslc.
set -euo pipefail
ROOT="${1:-.}"
OM="$ROOT/overlay/mlx/backend/omarchy"
SHADERS="$OM/shaders"

python3 - "$OM/CMakeLists.txt" "$SHADERS" <<'PY'
import re
import subprocess
import sys
import tempfile

cmake, shader_dir = sys.argv[1:]
text = open(cmake, encoding="utf-8").read()
expected = {
    "gated_delta_prefill_coopmat_bf16": "gated_delta_prefill_coopmat.comp",
    "gated_delta_prefill_kktqkt": "gated_delta_prefill_kktqkt.comp",
    "gated_delta_prefill_coopmat_hoist_bf16": "gated_delta_prefill_coopmat_hoist.comp",
}
pairs = re.findall(r"omarchy_shader\(\s*([\w]+)\s+shaders/([\w.]+)([^)]*)\)", text)
found = {name: (filename, args) for name, filename, args in pairs}
missing = set(expected) - found.keys()
if missing:
    raise SystemExit(f"CMake shader entries missing: {', '.join(sorted(missing))}")
for name, filename in expected.items():
    actual, args = found[name]
    if actual != filename:
        raise SystemExit(f"{name}: expected {filename}, found {actual}")
    flags = re.findall(r"-D\w+=\d+", args)
    with tempfile.NamedTemporaryFile(suffix=".spv") as out:
        command = ["glslc", "-O", "--target-env=vulkan1.3", *flags,
                   f"{shader_dir}/{filename}", "-o", out.name]
        subprocess.run(command, check=True)
    print(f"compile OK: {name} ({' '.join(flags) or 'no defines'})")
PY
