# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Deterministic host control for Parakeet Token-and-Duration decoding.

Tensor execution stays behind ``TdtWorker``. The control loop receives only
scalar token and duration decisions and forwards opaque recurrent-state handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

StateT = TypeVar("StateT")
JointStateT = TypeVar("JointStateT")


class TdtControlError(ValueError):
    """The TDT configuration or a backend decision is invalid."""


@dataclass(frozen=True)
class TdtConfig:
    blank_token_id: int
    durations: tuple[int, ...]
    max_symbols_per_step: int

    def __post_init__(self) -> None:
        _require_nonnegative(self.blank_token_id, "blank token id")
        if not self.durations:
            raise TdtControlError("duration classes must not be empty")
        for index, duration in enumerate(self.durations):
            _require_nonnegative(duration, f"duration class {index}")
        if (
            isinstance(self.max_symbols_per_step, bool)
            or not isinstance(self.max_symbols_per_step, int)
            or self.max_symbols_per_step <= 0
        ):
            raise TdtControlError("max symbols per step must be a positive integer")


@dataclass(frozen=True)
class TdtDecision:
    token_id: int
    duration_index: int

    def __post_init__(self) -> None:
        _require_nonnegative(self.token_id, "token id")
        _require_nonnegative(self.duration_index, "duration index")


@dataclass(frozen=True)
class DecoderStep(Generic[StateT, JointStateT]):
    recurrent_state: StateT
    joint_state: JointStateT


class TdtWorker(Protocol[StateT, JointStateT]):
    """Tensor backend boundary used by the host control loop."""

    def decode(
        self, input_token: int, recurrent_state: StateT
    ) -> DecoderStep[StateT, JointStateT]: ...

    def joint(self, frame_index: int, joint_state: JointStateT) -> TdtDecision: ...


@dataclass(frozen=True)
class TdtOutput(Generic[StateT]):
    token_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]
    durations: tuple[int, ...]
    final_state: StateT


def decode_tdt(
    *,
    valid_frames: int,
    config: TdtConfig,
    worker: TdtWorker[StateT, JointStateT],
    initial_state: StateT,
) -> TdtOutput[StateT]:
    """Run the pinned greedy TDT state machine without tensor computation."""
    _require_nonnegative(valid_frames, "valid frame count")
    frame = 0
    input_token = config.blank_token_id
    recurrent_state = initial_state
    joint_state: JointStateT
    decoder_state_valid = False
    token_ids: list[int] = []
    frame_indices: list[int] = []
    output_durations: list[int] = []

    while frame < valid_frames:
        symbols = 0
        advanced = False
        while symbols < config.max_symbols_per_step:
            if not decoder_state_valid or input_token != config.blank_token_id:
                step = worker.decode(input_token, recurrent_state)
                recurrent_state = step.recurrent_state
                joint_state = step.joint_state
                decoder_state_valid = True

            decision = worker.joint(frame, joint_state)
            if decision.duration_index >= len(config.durations):
                raise TdtControlError(
                    f"duration index {decision.duration_index} is outside "
                    f"{len(config.durations)} classes"
                )
            duration = config.durations[decision.duration_index]

            if decision.token_id == config.blank_token_id:
                frame += max(duration, 1)
                advanced = True
                break

            token_ids.append(decision.token_id)
            frame_indices.append(frame)
            output_durations.append(duration)
            input_token = decision.token_id
            symbols += 1
            if duration > 0:
                frame += duration
                advanced = True
                break

        if not advanced:
            frame += 1

    return TdtOutput(
        token_ids=tuple(token_ids),
        frame_indices=tuple(frame_indices),
        durations=tuple(output_durations),
        final_state=recurrent_state,
    )


def _require_nonnegative(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TdtControlError(f"{label} must be a non-negative integer")
