"""Replicate mlx's metal_kernel write_signature to dump the generated MSL for
each recheck kernel without touching the GPU (kernel construction only; the
caller must not mx.eval).

Matches .work/mlx/mlx/backend/common/metal_kernel.cpp:
  - inputs with size < 8 bind in the `constant` address space, others `device`;
  - ndim == 0 binds as `T&`, otherwise `T*`;
  - `NAME_shape` / `NAME_strides` / `NAME_ndim` metadata buffers are appended
    when the source mentions them;
  - attributes are appended when the source mentions them;
  - templated kernels get `template <...>` before the kernel and the
    `template [[host_name(...)]] [[kernel]] decltype(...)` instantiation after
    the body.

Usage: python3 -m tools.kernel_recheck.dump_msl [NAME...] [--out DIR]
Prints one `NAME <msl-path> <translate argv>` line per kernel; the argv feeds
omarchy_custom_kernel_translate_dump.
"""
import sys
from pathlib import Path

import numpy as np

from . import defs
from .run import build_inputs, source_text

DTYPE_STR = {
    "float32": "float",
    "float16": "float16_t",
    "bfloat16": "bfloat16_t",
    "uint32": "uint",
    "int32": "int",
    "uint8": "uchar",
    "uint16": "ushort",
    "int8": "char",
}

METAL_ATTRS = [
    "dispatch_quadgroups_per_threadgroup",
    "dispatch_simdgroups_per_threadgroup",
    "dispatch_threads_per_threadgroup",
    "grid_origin",
    "grid_size",
    "quadgroup_index_in_threadgroup",
    "quadgroups_per_threadgroup",
    "simdgroup_index_in_threadgroup",
    "simdgroups_per_threadgroup",
    "thread_execution_width",
    "thread_index_in_quadgroup",
    "thread_index_in_simdgroup",
    "thread_index_in_threadgroup",
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_threadgroup",
]


def _template_value(value):
    if isinstance(value, bool):
        return ("bool", "true" if value else "false")
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, str) and value in DTYPE_STR:
        return ("typename", DTYPE_STR[value])
    return ("typename", "float")


def generated_msl(name):
    spec = defs.get_spec(name)
    source = source_text(spec)
    inputs = build_inputs(spec)
    template = spec.get("template", [])
    pieces = []
    if template:
        parts = [f"{_template_value(v)[0]} {k}" for k, v in template]
        pieces.append("template <" + ", ".join(parts) + ">\n")
    func = f"custom_kernel_{name}"
    pieces.append(f"[[kernel]] void {func}(\n")
    index = 0
    params = []
    in_dtypes = defs.IN_DTYPES[name]
    for i, n in enumerate(spec["inputs"]):
        arr = np.asarray(inputs[n])
        location = "constant" if arr.size < 8 else "device"
        ref = "&" if arr.ndim == 0 else "*"
        params.append(
            f"  const {location} {DTYPE_STR[in_dtypes[i]]}{ref} {n} "
            f"[[buffer({index})]]")
        index += 1
        if arr.ndim > 0:
            if f"{n}_shape" in source:
                params.append(f"  const constant int* {n}_shape [[buffer({index})]]")
                index += 1
            if f"{n}_strides" in source:
                params.append(f"  const constant int64_t* {n}_strides [[buffer({index})]]")
                index += 1
            if f"{n}_ndim" in source:
                params.append(f"  const constant int& {n}_ndim [[buffer({index})]]")
                index += 1
    for j, on in enumerate(spec["outputs"]):
        t = DTYPE_STR[spec["out_dtypes"][j]]
        if spec.get("atomic_outputs"):
            t = f"atomic<{t}>"
        params.append(f"  device {t}* {on} [[buffer({index})]]")
        index += 1
    attrs = [a for a in METAL_ATTRS if a in source]
    total = index
    rendered = []
    for k, p in enumerate(params):
        is_last_param = k == len(params) - 1
        sep = "" if (is_last_param and not attrs) else ","
        rendered.append(p + sep)
    for k, a in enumerate(attrs):
        sep = "," if k < len(attrs) - 1 else ""
        rendered.append(f"  uint3 {a} [[{a}]]{sep}")
    pieces.append("\n".join(rendered) + "\n) {\n")
    pieces.append(source)
    pieces.append("\n}\n")
    if template:
        tvals = ", ".join(_template_value(v)[1] for _, v in template)
        tdef = f"<{tvals}>"
        pieces.append(
            f"\ntemplate [[host_name(\"{func}\")]] [[kernel]] decltype({func}{tdef}) "
            f"{func}{tdef};\n")
    return "".join(pieces)


def main():
    args = sys.argv[1:]
    out_dir = Path("generated")
    if "--out" in args:
        i = args.index("--out")
        out_dir = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    names = args or [s["name"] for s in defs.SPECS]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = defs.get_spec(name)
        path = out_dir / f"{name}.msl"
        path.write_text(generated_msl(name))
        grid = spec["grid"]
        tg = spec["threadgroup"]
        outs = len(spec["outputs"])
        print(f"{name} {path} {grid[0]} {grid[1]} {grid[2]} "
              f"{tg[0]} {tg[1]} {tg[2]} {outs}")


if __name__ == "__main__":
    main()
