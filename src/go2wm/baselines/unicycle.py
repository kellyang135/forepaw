"""Commanded-velocity unicycle reference with optional first-order lag."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotionState:
    x_m: float
    y_m: float
    yaw_rad: float
    forward_velocity_mps: float = 0.0
    yaw_rate_rps: float = 0.0


@dataclass(frozen=True, slots=True)
class UnicycleModel:
    """A transparent pose baseline; ``lag_time_constant_s=0`` is instantaneous."""

    lag_time_constant_s: float = 0.0

    def __post_init__(self) -> None:
        if self.lag_time_constant_s < 0 or not math.isfinite(self.lag_time_constant_s):
            raise ValueError("lag_time_constant_s must be finite and non-negative")

    def step(
        self,
        state: MotionState,
        *,
        commanded_forward_mps: float,
        commanded_yaw_rate_rps: float,
        duration_s: float,
    ) -> MotionState:
        if duration_s <= 0 or not math.isfinite(duration_s):
            raise ValueError("duration_s must be finite and positive")
        if self.lag_time_constant_s == 0:
            velocity = commanded_forward_mps
            yaw_rate = commanded_yaw_rate_rps
        else:
            gain = 1.0 - math.exp(-duration_s / self.lag_time_constant_s)
            velocity = state.forward_velocity_mps + gain * (
                commanded_forward_mps - state.forward_velocity_mps
            )
            yaw_rate = state.yaw_rate_rps + gain * (
                commanded_yaw_rate_rps - state.yaw_rate_rps
            )

        # Midpoint integration is stable and keeps the reference implementation
        # explicit; it is not presented as a physics engine.
        yaw_delta = yaw_rate * duration_s
        midpoint_yaw = state.yaw_rad + 0.5 * yaw_delta
        distance = velocity * duration_s
        return MotionState(
            x_m=state.x_m + distance * math.cos(midpoint_yaw),
            y_m=state.y_m + distance * math.sin(midpoint_yaw),
            yaw_rad=_wrap_angle(state.yaw_rad + yaw_delta),
            forward_velocity_mps=velocity,
            yaw_rate_rps=yaw_rate,
        )


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

