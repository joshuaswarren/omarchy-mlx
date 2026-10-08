"""Replicate mlx's metal_kernel write_signature to dump the generated MSL for
each recheck kernel without touching the GPU (kernel construction only; the
caller must not mx.eval). Byte-verified against the wheel's verbose=True dump
on 2026-10-08 (bitlinear_matmul).

Matches .work/mlx/mlx/backend/common/metal_kernel.cpp:
  - inputs with size < 8 bind in the `constant` address space, others `device`;
  - ndim == 0 binds as `T&`, otherwise `T*`;
  - `NAME_shape` / `NAME_strides` / `NAME_ndim` metadata buffers are appended
    when the source mentions them;
  - attributes are appended when the source mentions them, and the closing
    `) {` lands on the last attribute's line;
  - the kernel function name carries the template hash and the per-input /
    per-output dtype suffixes ('s' for 0-d, 'c' for small arrays);
  - templated kernels get `template <...>` before the kernel and the
    `template [[host_name(...)]] [[kernel]] decltype(...)` instantiation after
    the body, with bools rendered as 0/1.

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
    "uint64": "uint64_t",
    "int32": "int",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "int8": "int8_t",
    "int16": "int16_t",
    "bool": "bool",
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
    """(param kind, definition rendering, instantiation rendering)."""
    if isinstance(value, bool):
        return ("bool", str(int(value)), str(int(value)))
    if isinstance(value, int):
        return ("int", str(value), str(value))
    if isinstance(value, str) and value in DTYPE_STR:
        return ("typename", DTYPE_STR[value], DTYPE_STR[value])
    return ("typename", "float", "float")


def _make_template_hash(template_def):
    out = []
    i = 0
    while i < len(template_def):
        c = template_def[i]
        if c in "<>":
            out.append("_")
        elif c == "," and i + 1 < len(template_def) and template_def[i + 1] == " ":
            out.append("_")
            i += 1
        else:
            out.append(c)
        i += 1
    return "".join(out)[:-1]


def generated_msl(name):
    spec = defs.get_spec(name)
    source = source_text(spec)
    inputs = build_inputs(spec)
    template = spec.get("template", [])
    tdef = "<" + ", ".join(_template_value(v)[1] for _, v in template) + ">" \
        if template else ""
    kernel_name = f"custom_kernel_{name}" + \
        (("_" + _make_template_hash(tdef)) if template else "")
    in_dtypes = defs.IN_DTYPES[name]
    for i, n in enumerate(spec["inputs"]):
        arr = np.asarray(inputs[n])
        kernel_name += "_" + DTYPE_STR[in_dtypes[i]]
        if arr.ndim == 0:
            kernel_name += "s"
        elif arr.size < 8:
            kernel_name += "c"
    for od in spec["out_dtypes"]:
        kernel_name += "_" + DTYPE_STR[od]

    pieces = []
    if template:
        parts = [f"{_template_value(v)[0]} {k}" for k, v in template]
        pieces.append("template <" + ", ".join(parts) + ">\n")
    pieces.append(f"[[kernel]] void {kernel_name}(\n")
    index = 0
    params = []
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
    rendered = []
    for k, p in enumerate(params):
        last = k == len(params) - 1
        rendered.append(p + ("" if last and not attrs else ","))
    if attrs:
        rendered[-1] = rendered[-1].rstrip(",") + ","
        for k, a in enumerate(attrs):
            end = "]]) {" if k == len(attrs) - 1 else "],"
            rendered.append(f"  uint3 {a} [[{a}{end}")
    else:
        rendered[-1] = rendered[-1] + ") {"
    pieces.append("\n".join(rendered) + "\n")
    pieces.append(source)
    pieces.append("\n}\n")
    if template:
        pieces.append(
            f"\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] "
            f"decltype({kernel_name}{tdef}) {kernel_name}{tdef};\n\n")
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
