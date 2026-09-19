"""Dependency-light domain types shared by model, planning, and runtime code.

These types intentionally contain no NumPy or PyTorch objects.  Concrete learning
adapters should convert tensors at their boundary and return immutable values here.
That keeps planning tests fast and prevents model implementation details from
leaking into the runtime API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, TypeAlias

Latent: TypeAlias = tuple[float, ...]
Point2D: TypeAlias = tuple[float, float]


def as_latent(values: Sequence[float], *, expected_dim: int | None = None) -> Latent:
    """Copy a numeric sequence into a validated, immutable latent vector."""

    result = tuple(float(value) for value in values)
    if expected_dim is not None and len(result) != expected_dim:
        raise ValueError(f"expected latent dimension {expected_dim}, got {len(result)}")
    if not result:
        raise ValueError("latent vectors must not be empty")
    if not all(isfinite(value) for value in result):
        raise ValueError("latent vectors must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class ActionBlock:
    """A velocity command held for a fixed amount of simulated time."""

    forward_mps: float
    yaw_rate_rps: float
    duration_s: float = 0.5

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (self.forward_mps, self.yaw_rate_rps, self.duration_s)
        ):
            raise ValueError("action values must be finite")
        if self.duration_s <= 0:
            raise ValueError("action duration must be positive")


@dataclass(frozen=True, slots=True)
class RobotState:
    """Task-level robot state decoded from a latent representation."""

    x_m: float
    y_m: float
    yaw_rad: float

    @property
    def xy(self) -> Point2D:
        return (self.x_m, self.y_m)


@dataclass(frozen=True, slots=True)
class ObjectState:
    """Position of a consistently indexed object (for example, ``light``)."""

    object_id: str
    x_m: float
    y_m: float

    @property
    def xy(self) -> Point2D:
        return (self.x_m, self.y_m)


@dataclass(frozen=True, slots=True)
class PredictedState:
    """Readout for one future action block."""

    step: int
    robot: RobotState
    objects: tuple[ObjectState, ...] = ()
    failure_risk: float = 0.0
    latent: Latent | None = None

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("predicted steps are one-based")
        if not 0.0 <= self.failure_risk <= 1.0:
            raise ValueError("failure risk must be between zero and one")


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Aligned context consumed by a world-model adapter.

    ``observations`` remain opaque so adapters can accept bytes, arrays, image
    handles, or preprocessed tensors.  Commands are the transitions between
    observations, hence a three-frame history normally has two commands.
    """

    observations: tuple[Any, ...]
    command_history: tuple[ActionBlock, ...]

    def validate(self, *, history_frames: int, block_duration_s: float) -> None:
        if len(self.observations) != history_frames:
            raise ValueError(
                f"expected {history_frames} observations, got {len(self.observations)}"
            )
        expected_commands = max(history_frames - 1, 0)
        if len(self.command_history) != expected_commands:
            raise ValueError(
                f"expected {expected_commands} history commands, "
                f"got {len(self.command_history)}"
            )
        for command in self.command_history:
            if abs(command.duration_s - block_duration_s) > 1e-9:
                raise ValueError(
                    "history command duration does not match the model bundle"
                )


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    """Predicted consequences for one complete candidate command sequence."""

    candidate_id: str
    family: str
    actions: tuple[ActionBlock, ...]
    states: tuple[PredictedState, ...]
    bundle_id: str
    metadata: dict[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if len(self.actions) != len(self.states):
            raise ValueError("a rollout must contain one predicted state per action")
        expected_steps = tuple(range(1, len(self.states) + 1))
        actual_steps = tuple(state.step for state in self.states)
        if actual_steps != expected_steps:
            raise ValueError(
                f"rollout steps must be consecutive and one-based: {actual_steps!r}"
            )
