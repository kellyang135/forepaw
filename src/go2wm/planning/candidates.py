"""Deterministic, structured candidate commands for three-second MPC."""

from __future__ import annotations

from dataclasses import dataclass

from go2wm.model import ActionBlock


@dataclass(frozen=True, slots=True)
class CandidateSequence:
    """A coherent maneuver rather than independently jittered commands."""

    candidate_id: str
    family: str
    actions: tuple[ActionBlock, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.family:
            raise ValueError("candidate_id and family must not be empty")
        if not self.actions:
            raise ValueError("candidate actions must not be empty")


@dataclass(frozen=True, slots=True)
class CandidateLibraryConfig:
    horizon_blocks: int = 6
    block_duration_s: float = 0.5
    max_forward_mps: float = 0.6
    max_abs_yaw_rate_rps: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon_blocks != 6:
            raise ValueError("the initial structured library is defined for six blocks")
        if self.block_duration_s <= 0:
            raise ValueError("block_duration_s must be positive")
        if self.max_forward_mps < 0.55:
            raise ValueError("max_forward_mps must allow the 0.55 m/s library command")
        if self.max_abs_yaw_rate_rps < 0.9:
            raise ValueError("yaw bounds must allow the 0.9 rad/s library command")


def build_candidate_library(
    config: CandidateLibraryConfig | None = None,
) -> tuple[CandidateSequence, ...]:
    """Return exactly 64 deterministic candidates in five maneuver families.

    The mix is intentionally interpretable: 12 direct/push approaches, 16 left
    detours, 16 right detours, 16 heading adjustments, and 4 wait/stop plans.
    The first action is always the one the receding-horizon controller executes.
    """

    config = config or CandidateLibraryConfig()
    duration = config.block_duration_s

    def action(forward: float, yaw: float) -> ActionBlock:
        if abs(forward) > config.max_forward_mps + 1e-12:
            raise ValueError(f"forward command {forward} exceeds configured bounds")
        if abs(yaw) > config.max_abs_yaw_rate_rps + 1e-12:
            raise ValueError(f"yaw command {yaw} exceeds configured bounds")
        return ActionBlock(forward, yaw, duration)

    candidates: list[CandidateSequence] = []

    # Direct approaches: constant headings plus smooth accelerating profiles.
    for speed in (0.25, 0.40, 0.55):
        for yaw_bias in (-0.12, 0.0, 0.12):
            label = _signed_label(yaw_bias)
            candidates.append(
                CandidateSequence(
                    candidate_id=f"push_v{_decimal_label(speed)}_yaw{label}",
                    family="approach_push",
                    actions=tuple(action(speed, yaw_bias) for _ in range(6)),
                )
            )
    for index, profile in enumerate(
        (
            (0.15, 0.25, 0.35, 0.45, 0.45, 0.45),
            (0.20, 0.30, 0.40, 0.50, 0.50, 0.50),
            (0.25, 0.35, 0.45, 0.55, 0.55, 0.55),
        ),
        start=1,
    ):
        candidates.append(
            CandidateSequence(
                candidate_id=f"push_accel_{index}",
                family="approach_push",
                actions=tuple(action(speed, 0.0) for speed in profile),
            )
        )

    # Detours carve an arc away from the obstacle and counter-steer near the end.
    for direction, family in ((1.0, "detour_left"), (-1.0, "detour_right")):
        side = "left" if direction > 0 else "right"
        for speed in (0.30, 0.45):
            for turn_rate in (0.60, 0.90):
                for turn_blocks in (1, 2):
                    for recovery_rate in (0.60, 0.90):
                        actions: list[ActionBlock] = []
                        for step in range(6):
                            if step < turn_blocks:
                                actions.append(action(speed * 0.75, direction * turn_rate))
                            elif step >= 6 - turn_blocks:
                                actions.append(
                                    action(speed * 0.85, -direction * recovery_rate)
                                )
                            else:
                                actions.append(action(speed, 0.0))
                        candidates.append(
                            CandidateSequence(
                                candidate_id=(
                                    f"{side}_v{_decimal_label(speed)}_"
                                    f"turn{_decimal_label(turn_rate)}_"
                                    f"n{turn_blocks}_recover{_decimal_label(recovery_rate)}"
                                ),
                                family=family,
                                actions=tuple(actions),
                            )
                        )

    # Short heading corrections cover cases where a full S-shaped detour is wrong.
    for yaw_rate in (-0.80, -0.40, 0.40, 0.80):
        for speed in (0.20, 0.35):
            for turn_blocks in (1, 2):
                actions = tuple(
                    action(speed * 0.70, yaw_rate) if step < turn_blocks else action(speed, 0.0)
                    for step in range(6)
                )
                candidates.append(
                    CandidateSequence(
                        candidate_id=(
                            f"adjust_yaw{_signed_label(yaw_rate)}_"
                            f"v{_decimal_label(speed)}_n{turn_blocks}"
                        ),
                        family="turn_adjust",
                        actions=actions,
                    )
                )

    # A zero first block gives the safety layer a genuinely non-moving option.
    stop = action(0.0, 0.0)
    candidates.extend(
        (
            CandidateSequence("stop_hold", "stop", (stop,) * 6),
            CandidateSequence(
                "stop_then_creep",
                "stop",
                (stop, stop, *(action(0.20, 0.0) for _ in range(4))),
            ),
            CandidateSequence(
                "stop_then_left",
                "stop",
                (
                    stop,
                    stop,
                    *(action(0.18, 0.45) for _ in range(2)),
                    *(action(0.22, 0.0) for _ in range(2)),
                ),
            ),
            CandidateSequence(
                "stop_then_right",
                "stop",
                (
                    stop,
                    stop,
                    *(action(0.18, -0.45) for _ in range(2)),
                    *(action(0.22, 0.0) for _ in range(2)),
                ),
            ),
        )
    )

    result = tuple(candidates)
    _validate_library(result, config)
    return result


def family_counts(candidates: tuple[CandidateSequence, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.family] = counts.get(candidate.family, 0) + 1
    return counts


def _validate_library(
    candidates: tuple[CandidateSequence, ...], config: CandidateLibraryConfig
) -> None:
    if len(candidates) != 64:
        raise AssertionError(f"candidate library must have 64 entries, got {len(candidates)}")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise AssertionError("candidate IDs must be unique")
    action_signatures = [candidate.actions for candidate in candidates]
    if len(set(action_signatures)) != len(action_signatures):
        raise AssertionError("candidate action sequences must be unique")
    for candidate in candidates:
        if len(candidate.actions) != config.horizon_blocks:
            raise AssertionError("every candidate must span the configured horizon")


def _decimal_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _signed_label(value: float) -> str:
    prefix = "p" if value >= 0 else "m"
    return prefix + _decimal_label(abs(value))
