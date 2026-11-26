# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Compiler eligibility for Core ML packages (plan section 39).

Classifies every op of a parsed ``.mlpackage`` against the H13
execution surface using three inputs:

1. the H13 lowering surface of the pinned mil-hwx-compiler
   (commit 83a4434, ``plugins/H13/ANEH13Compiler.mm`` dispatch; frozen
   here so eligibility works on any host, provenance recorded);
2. the frontend pre-expansion (``constexpr_lut_to_dense`` → dense
   fp16 consts, receipts/2026-09-13-depalettize);
3. the fp16 0/1 mask lowerings
   (docs/2026-09-13-h13-boolean-mask-fp16.md).

Report classes: SUPPORTED, LOWERABLE-VIA-FRONTEND, NEEDS-COMPILER-OP
(the exact blocking ops are listed), BOUNDARY (host pack/unpack).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .mlpackage import inspect as inspect_package
from .mlpackage import open_mlpackage
from .mask_lowering import plan_mask_lowering
from .pad_elimination import plan_pad_elimination
from .proto import load_model

# H13 lowering surface, mil-hwx-compiler 83a4434810b0981d1764e6231f6a679df6f35241
# (release ane-parity-83a4434), extracted from the dispatch branches in
# plugins/H13/ANEH13Compiler.mm — see receipts/2026-09-13-encoder-coverage.
H13_SUPPORTED_OPS = frozenset({
    "add", "sub", "mul", "maximum", "minimum", "real_div",
    "abs", "exp", "gelu", "leaky_relu", "relu", "rsqrt", "sigmoid",
    "silu", "sqrt", "tanh",
    "softmax", "layer_norm", "reduce_sum", "reduce_max", "reduce_mean",
    "reshape", "squeeze", "expand_dims", "split",
    "matmul", "linear", "conv", "const", "clip",
})

SUPPORTED = "SUPPORTED"
LOWERABLE_VIA_FRONTEND = "LOWERABLE-VIA-FRONTEND"
NEEDS_COMPILER_OP = "NEEDS-COMPILER-OP"
BOUNDARY = "BOUNDARY"
# Section-39 nuance: implemented on a compiler branch, not yet in
# the pinned release. These ops become plain SUPPORTED once the
# branch ships and the ane-compiler.lock pin moves.
SUPPORTED_PENDING_RELEASE = "SUPPORTED-PENDING-COMPILER-RELEASE"

_PENDING_RELEASE_OPS = {
    "less": \
        "registry entry + exact rejection on compiler branch feature/h13-registry-boolean-ops "
        "(b4f4da9); 1-byte bool surface via the existing per-binding "
        "dtype field (compiler yield: no separate elementDtype field); "
        "the pinned release 83a4434 has neither",
    "floor": \
        "registry entry + exact rejection on compiler branch feature/h13-registry-boolean-ops "
        "(b4f4da9); 1-byte bool surface captured by the compiler lane; pinned "
        "release 83a4434 has neither",
    "floor_div": \
        "composes real_div (pinned) with floor (branch); blocked on the floor "
        "encoder release",
}

ELIGIBILITY_SCHEMA = "mlx-omarchy.coreml.eligibility/1"


class EligibilityError(RuntimeError):
    """The package cannot be classified; the reason is named."""


@dataclass(frozen=True)
class OpDisposition:
    op_type: str
    count: int
    disposition: str
    reason: str


def classify_spec(spec, histogram: dict[str, int]) -> list[OpDisposition]:
    """Classify one op type from its count and the package spec."""
    dispositions: list[OpDisposition] = []
    mask_entries: dict[str, list] = {}
    for entry in plan_mask_lowering(spec):
        mask_entries.setdefault(entry.op_type, []).append(entry)
    for op_type, count in sorted(
        histogram.items(), key=lambda item: (-item[1], item[0])
    ):
        if op_type in H13_SUPPORTED_OPS:
            dispositions.append(
                OpDisposition(op_type, count, SUPPORTED, "H13 lowering exists")
            )
            continue
        if op_type == "constexpr_lut_to_dense":
            dispositions.append(
                OpDisposition(
                    op_type,
                    count,
                    LOWERABLE_VIA_FRONTEND,
                    "byte-exact pre-expansion to dense fp16 consts "
                    "(receipts/2026-09-13-depalettize)",
                )
            )
            continue
        if op_type == "pad":
            pad_entries = plan_pad_elimination(spec)
            eliminable = sum(
                1 for e in pad_entries if e.disposition == "ELIMINABLE"
            )
            rejected = sum(
                1 for e in pad_entries if e.disposition == "REJECTED"
            )
            if eliminable + rejected != count:
                raise EligibilityError(
                    f"pad plan accounts {eliminable + rejected} ops, "
                    f"histogram says {count}"
                )
            if eliminable:
                dispositions.append(
                    OpDisposition(
                        op_type,
                        eliminable,
                        LOWERABLE_VIA_FRONTEND,
                        "depthwise identity-conv rewrite carrying the "
                        "amounts as native conv padding — bit-exact "
                        "(1.0 * v = v for every fp16 lane); Apple's own "
                        "ANE tool rejects pad in all forms "
                        "(OracleMint2, 2026-09-13)",
                    )
                )
            if rejected:
                reasons = "; ".join(
                    e.reason
                    for e in pad_entries
                    if e.disposition == "REJECTED"
                )
                dispositions.append(
                    OpDisposition(
                        op_type, rejected, NEEDS_COMPILER_OP, reasons
                    )
                )
            continue
        if op_type in _PENDING_RELEASE_OPS:
            dispositions.append(
                OpDisposition(
                    op_type,
                    count,
                    SUPPORTED_PENDING_RELEASE,
                    _PENDING_RELEASE_OPS[op_type],
                )
            )
            continue
        entries = mask_entries.get(op_type, [])
        if entries:
            per_type = len(entries)
            dispositions_seen = {e.disposition for e in entries}
            if dispositions_seen == {"LOWERABLE"} and per_type == count:
                dispositions.append(
                    OpDisposition(
                        op_type,
                        count,
                        LOWERABLE_VIA_FRONTEND,
                        f"fp16 0/1 mask lowering: {entries[0].reason}",
                    )
                )
                continue
            if dispositions_seen == {"BOUNDARY"} and per_type == count:
                dispositions.append(
                    OpDisposition(
                        op_type, count, BOUNDARY, entries[0].reason
                    )
                )
                continue
            if dispositions_seen == {"NEEDS_COMPILER_OP"} and per_type == count:
                dispositions.append(
                    OpDisposition(
                        op_type, count, NEEDS_COMPILER_OP, entries[0].reason
                    )
                )
                continue
            # Mixed dispositions inside one op type (select: finite vs
            # -inf fills): split by the per-entry outcome.
            lowerable = sum(
                1 for e in entries if e.disposition == "LOWERABLE"
            )
            blocking = sum(
                1 for e in entries if e.disposition == "NEEDS_COMPILER_OP"
            )
            boundary = sum(
                1 for e in entries if e.disposition == "BOUNDARY"
            )
            accounted = lowerable + blocking + boundary
            if accounted != count:
                raise EligibilityError(
                    f"mask plan accounts {accounted} '{op_type}' ops, "
                    f"histogram says {count}"
                )
            if lowerable:
                dispositions.append(
                    OpDisposition(
                        op_type,
                        lowerable,
                        LOWERABLE_VIA_FRONTEND,
                        "fp16 0/1 mask lowering (finite-fill instances)",
                    )
                )
            if blocking:
                # The -inf-fill select encoder is captured on the
                # compiler branch (3-in/out7 bool surface via the
                # existing per-binding dtype field); only the release
                # is pending.
                dispositions.append(
                    OpDisposition(
                        op_type,
                        blocking,
                        SUPPORTED_PENDING_RELEASE,
                        "non-finite select fill: 0 * inf = NaN; encoder "
                        "captured on compiler branch "
                        "feature/h13-registry-boolean-ops (b4f4da9+) as a "
                        "3-in/out7 1-byte bool surface; pinned release "
                        "83a4434 has neither",
                    )
                )
            continue
        dispositions.append(
            OpDisposition(
                op_type,
                count,
                NEEDS_COMPILER_OP,
                "no H13 lowering branch and no frontend lowering",
            )
        )
    return dispositions


def eligibility_report(package: Path) -> dict:
    """The section-39 report for one ``.mlpackage``."""
    inventory = inspect_package(Path(package))
    opened = open_mlpackage(Path(package))
    spec = load_model(opened.model_path.read_bytes())
    histogram = inventory["op_histogram"]
    dispositions = classify_spec(spec, histogram)

    counts = {
        SUPPORTED: 0,
        LOWERABLE_VIA_FRONTEND: 0,
        NEEDS_COMPILER_OP: 0,
        BOUNDARY: 0,
        SUPPORTED_PENDING_RELEASE: 0,
    }
    blocking: list[dict] = []
    for entry in dispositions:
        counts[entry.disposition] += entry.count
        if entry.disposition == NEEDS_COMPILER_OP:
            blocking.append(
                {
                    "op": entry.op_type,
                    "count": entry.count,
                    "reason": entry.reason,
                }
            )
    total = sum(counts.values())
    if total != inventory["op_total"]:
        raise EligibilityError(
            f"classification covers {total} ops, package has "
            f"{inventory['op_total']}"
        )

    # Post-frontend-transform projection: what the compiler would see
    # after the frontend lowerings run (depalettize folds its 194 ops
    # into consts; mask lowerings rewrite onto existing ops).
    post = dict(histogram)
    for entry in dispositions:
        if entry.disposition == LOWERABLE_VIA_FRONTEND:
            post[entry.op_type] = post.get(entry.op_type, 0) - entry.count
            if post[entry.op_type] <= 0:
                del post[entry.op_type]
            if entry.op_type == "constexpr_lut_to_dense":
                post["const"] = post.get("const", 0) + entry.count
            if entry.op_type == "pad":
                post["conv"] = post.get("conv", 0) + entry.count
    return {
        "schema": ELIGIBILITY_SCHEMA,
        "package": str(package),
        "package_name": inventory["model"].get("description", {}).get(
            "metadata", {}
        ).get("userDefined", {}),
        "total_ops": inventory["op_total"],
        "counts": counts,
        "ops": [
            {
                "op": entry.op_type,
                "count": entry.count,
                "disposition": entry.disposition,
                "reason": entry.reason,
            }
            for entry in dispositions
        ],
        "blocking_ops": blocking,
        "post_frontend_histogram": dict(sorted(post.items())),
    }
