# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Keep ANE graph outputs in fp16 (Phase 5 fail-fast leaf).

The H13 backend cannot materialize non-fp16 logical results
(`h13.unsupported-logical-result-conversion`): the graph handed to the
ANE compiler must return fp16 (or 1-byte bool) tensors. Trailing
boundary casts that only widen — fp16→fp32, bool→int32 — are exact and
injective, so the frontend peels them off the graph: the ANE program
returns the pre-cast tensor, and the cast stays GPU-side per §3.2/§32
(decoder and joint run on Vulkan). Re-applying the peeled cast
downstream reproduces the original output bit-for-bit.

Any trailing cast that is NOT an exact widening (e.g. fp32→fp16
narrowing) is refused with a named error: silently dropping it would
change values.

Operates on MIL text: the compiler consumes `model.mil`, and the peel
is a line-level rewrite of the trailing-return region plus the function
footer. All other lines are preserved byte-identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class OutputPeelError(RuntimeError):
    """A trailing boundary cast cannot be peeled; the reason is named."""


# tensor<fp32, [1, 375, 640]> encoder_hidden = cast(dtype = ..., x = ...)...;
_CAST_LINE = re.compile(
    r"^(?P<indent>\s*)tensor<(?P<out_dtype>[a-z0-9]+),\s*"
    r"(?P<out_shape>\[[^\]]*\])>\s*(?P<out>[A-Za-z_][\w]*) = cast\("
    r"dtype = (?P<dtype_const>[A-Za-z_][\w]*), "
    r"x = (?P<src>[A-Za-z_][\w]*)\)"
)
# } -> (encoder_hidden, encoder_mask);
_FOOTER = re.compile(r"^(?P<indent>\s*)\} -> \((?P<returns>[^)]*)\)\s*;\s*$")
# tensor<fp16, [1, 375, 640]> linear_217_cast_fp16 = linear(...
_DEF_LINE = re.compile(
    r"^\s*tensor<(?P<dtype>[a-z0-9]+),\s*(?P<shape>\[[^\]]*\])>\s*"
    r"(?P<name>[A-Za-z_][\w]*) = "
)
# ... val = tensor<string, []>("fp32")];
_STRING_CONST = re.compile(
    r'val = tensor<string, \[\]>\("(?P<value>[^"]*)"\)'
)

# Exact, injective widenings the peel may remove. Anything else (notably
# fp32→fp16 narrowing) must fail loudly.
_PEELABLE = {
    ("fp16", "fp32"),
    ("bool", "int32"),
}


@dataclass(frozen=True)
class PeeledCast:
    """One removed boundary cast, recorded for the GPU epilogue."""

    ane_output: str  # tensor the ANE graph now returns
    ane_dtype: str
    ane_shape: str
    removed_output: str  # original graph output name
    removed_dtype: str
    dtype_const: str


def peel_fp16_outputs(mil_text: str) -> tuple[str, dict]:
    """Remove trailing exact-widening boundary casts from MIL text.

    Returns (peeled_text, epilogue_spec). The epilogue spec tells the
    Vulkan side which casts to re-apply, in order, to reproduce the
    original outputs bit-for-bit.
    """
    lines = mil_text.splitlines(keepends=True)

    footer_index = next(
        (i for i, line in enumerate(lines) if _FOOTER.match(line.rstrip("\n"))),
        None,
    )
    if footer_index is None:
        raise OutputPeelError("no function footer '} -> (...);' found")

    returns = [
        name.strip()
        for name in _FOOTER.match(lines[footer_index].rstrip("\n")).group(
            "returns"
        ).split(",")
    ]
    if any(not name for name in returns):
        raise OutputPeelError("empty name in function return tuple")

    definitions: dict[str, tuple[int, str, str]] = {}
    param_pattern = re.compile(
        r"tensor<(?P<dtype>[a-z0-9]+),\s*(?P<shape>\[[^\]]*\])>\s*"
        r"(?P<name>[A-Za-z_][\w]*)"
    )
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("func ") and "(" in stripped:
            params = stripped.split("(", 1)[1].rsplit(")", 1)[0]
            # finditer, not comma-split: shape tuples contain commas.
            for match in param_pattern.finditer(params):
                definitions[match.group("name")] = (
                    index,
                    match.group("dtype"),
                    match.group("shape"),
                )
    for index, line in enumerate(lines):
        match = _DEF_LINE.match(line)
        if match:
            definitions[match.group("name")] = (
                index,
                match.group("dtype"),
                match.group("shape"),
            )
    string_consts: dict[str, str] = {}
    for line in lines:
        match = _DEF_LINE.match(line)
        if not match:
            continue
        value = _STRING_CONST.search(line)
        if value and match.group("dtype") == "string":
            string_consts[match.group("name")] = value.group("value")

    remove: set[int] = set()
    peeled: list[PeeledCast] = []
    new_returns: dict[str, str] = {}
    for name in returns:
        if name not in definitions:
            raise OutputPeelError(
                f"return '{name}' has no defining line in the graph"
            )
        index, out_dtype, out_shape = definitions[name]
        cast = _CAST_LINE.match(lines[index].rstrip("\n"))
        if cast is None or cast.group("out") != name:
            continue  # not a trailing cast; ANE already returns it as-is
        dtype_name = cast.group("dtype_const")
        if dtype_name not in string_consts:
            raise OutputPeelError(
                f"cast '{name}' dtype const '{dtype_name}' is not a "
                "readable string constant"
            )
        target_dtype = string_consts[dtype_name]
        src = cast.group("src")
        if src not in definitions:
            raise OutputPeelError(
                f"cast '{name}' input '{src}' has no defining line"
            )
        src_index, src_dtype, src_shape = definitions[src]
        if src_index > index:
            raise OutputPeelError(
                f"cast '{name}' input '{src}' is defined after it"
            )
        if out_dtype != target_dtype:
            raise OutputPeelError(
                f"cast '{name}' declares {out_dtype} but its dtype const "
                f"says {target_dtype}"
            )
        if (src_dtype, out_dtype) not in _PEELABLE:
            raise OutputPeelError(
                f"refusing to peel '{name}': {src_dtype}->{out_dtype} is "
                "not an exact widening (only fp16->fp32 and bool->int32 "
                "peel; anything else would change values)"
            )
        if src_shape != out_shape:
            raise OutputPeelError(
                f"cast '{name}' changes shape {src_shape}->{out_shape}; "
                "the peel only handles dtype conversions"
            )
        remove.add(index)
        new_returns[name] = src
        peeled.append(
            PeeledCast(
                ane_output=src,
                ane_dtype=src_dtype,
                ane_shape=src_shape,
                removed_output=name,
                removed_dtype=out_dtype,
                dtype_const=dtype_name,
            )
        )

    if not peeled:
        raise OutputPeelError(
            "no trailing boundary casts found; nothing to peel"
        )

    kept = [
        line for index, line in enumerate(lines) if index not in remove
    ]
    footer = _FOOTER.match(lines[footer_index].rstrip("\n"))
    renamed = ", ".join(new_returns.get(name, name) for name in returns)
    eol = "\n" if lines[footer_index].endswith("\n") else ""
    kept[footer_index - len(remove)] = (
        f"{footer.group('indent')}}} -> ({renamed});{eol}"
    )

    epilogue = {
        "schema": "mlx-omarchy.gpu-boundary-epilogue.v1",
        "casts": [
            {
                "input": entry.ane_output,
                "input_dtype": entry.ane_dtype,
                "input_shape": entry.ane_shape,
                "output": entry.removed_output,
                "output_dtype": entry.removed_dtype,
            }
            for entry in peeled
        ],
    }
    return "".join(kept), epilogue
