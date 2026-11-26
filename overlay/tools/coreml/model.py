# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""User-facing Core ML model API skeleton (plan sections 36-37).

Loads a ``.mlpackage``, reports eligibility (section 39), and enforces
explicit compute-target semantics: an unsupported target fails with a
named error listing the blocking operations. There is no silent
placement (plan section 3.3) — nothing executes unless its target is
both requested and eligible, and 'cpu' is refused outright as a tensor
execution target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .eligibility import (
    BOUNDARY,
    LOWERABLE_VIA_FRONTEND,
    NEEDS_COMPILER_OP,
    eligibility_report,
)
from .mlpackage import inspect as inspect_package

COMPUTE_TARGETS = ("ane", "gpu")


class CoreMLError(RuntimeError):
    """Base error for the Core ML model API."""


class PackageNotEligible(CoreMLError):
    """The package cannot run on the requested compute target."""


class ComputeTargetUnsupported(CoreMLError):
    """The compute target itself is not available in this build."""


@dataclass(frozen=True)
class Eligibility:
    total_ops: int
    counts: dict[str, int]
    blocking_ops: tuple[dict, ...]

    @property
    def eligible(self) -> bool:
        return not self.blocking_ops


class CoreMLModel:
    """One loaded Core ML package with explicit target semantics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise CoreMLError(
                f"{self.path}: not a directory (.mlpackage is a directory)"
            )
        self._inventory = inspect_package(self.path)
        report = eligibility_report(self.path)
        self._eligibility = Eligibility(
            total_ops=report["total_ops"],
            counts=dict(report["counts"]),
            blocking_ops=tuple(report["blocking_ops"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CoreMLModel":
        return cls(path)

    @property
    def description(self) -> dict:
        return self._inventory["model"]["description"]

    def eligibility(self) -> Eligibility:
        return self._eligibility

    def check_compute_target(self, target: str) -> None:
        """Raise unless ``target`` can execute this package.

        ANE requires zero blocking ops after the frontend lowerings;
        the GPU (Vulkan) path is not yet wired for Core ML packages and
        reports that exactly; 'cpu' is refused as a tensor target
        (section 3.3). Nothing is silently re-placed.
        """
        if target == "cpu":
            raise ComputeTargetUnsupported(
                "compute_target='cpu' is refused: CPU tensor execution "
                "is not a fallback (plan section 3.3)"
            )
        if target not in COMPUTE_TARGETS:
            raise ComputeTargetUnsupported(
                f"unknown compute_target {target!r}; expected one of "
                f"{COMPUTE_TARGETS}"
            )
        if target == "gpu":
            raise ComputeTargetUnsupported(
                "compute_target='gpu' is not wired for Core ML packages "
                "yet; the Vulkan Core ML path lands with Phase 6"
            )
        if self._eligibility.blocking_ops:
            listed = ", ".join(
                f"{entry['op']} x{entry['count']}"
                for entry in self._eligibility.blocking_ops
            )
            raise PackageNotEligible(
                f"compute_target='ane' is blocked by {len(self._eligibility.blocking_ops)} "
                f"op classes ({self._eligibility.counts[NEEDS_COMPILER_OP]} ops): "
                f"{listed}; frontend lowerings already cover "
                f"{self._eligibility.counts[LOWERABLE_VIA_FRONTEND]} ops, "
                f"{self._eligibility.counts[BOUNDARY]} are host boundary ops"
            )
