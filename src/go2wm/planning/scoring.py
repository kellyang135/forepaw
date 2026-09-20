"""Transparent rollout scoring for learned model-predictive control."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import hypot

from go2wm.model import ActionBlock, CandidateRollout, Point2D


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    """Weights tuned on validation episodes and frozen before final evaluation."""

    goal_distance: float = 1.0
    failure_risk: float = 4.0
    stall: float = 2.0
    control_effort: float = 0.10
    command_change: float = 0.10

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.as_tuple()):
            raise ValueError("score weights must be non-negative")

    def as_tuple(self) -> tuple[float, ...]:
        return (
            self.goal_distance,
            self.failure_risk,
            self.stall,
            self.control_effort,
            self.command_change,
        )


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    weights: ScoreWeights = ScoreWeights()
    minimum_progress_m: float = 0.08
    active_forward_threshold_mps: float = 0.10

    def __post_init__(self) -> None:
        if self.minimum_progress_m < 0 or self.active_forward_threshold_mps < 0:
            raise ValueError("scoring thresholds must be non-negative")


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Score parts.

    ``goal_term_m`` is the goal quantity that enters ``total`` (weighted by
    ``ScoreWeights.goal_distance``); ``terminal_goal_distance_m`` is kept for
    reporting.  New fields are appended with defaults so positional callers
    keep working.
    """

    candidate_id: str
    total: float
    terminal_goal_distance_m: float
    maximum_failure_risk: float
    lack_of_progress_m: float
    control_effort: float
    command_change: float
    goal_cost_m: float | None = None
    closest_goal_distance_m: float | None = None

    @property
    def goal_term_m(self) -> float:
        return self.terminal_goal_distance_m if self.goal_cost_m is None else self.goal_cost_m


class RolloutScorer:
    """Score candidate predictions; lower is better.

    The goal term is the mean, over the predicted blocks, of how far the robot
    is outside the goal circle, with arrival absorbing: once a block ends inside
    the circle, it and every later block cost 0, because the task ends there.
    A terminal-only distance let a plan that stops first and creeps in late beat
    plans that arrive now; since only the first block executes, the robot never
    moved (D-028).  This term rewards arriving early and still pulls toward far
    goals.

    There is deliberately no generic contact penalty.  Contact is necessary for
    successful pushing and should affect the score only through predicted progress,
    object motion, or a specifically trained undesirable-outcome risk.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()

    def score(
        self,
        rollout: CandidateRollout,
        *,
        goal_xy: Point2D,
        initial_robot_xy: Point2D,
        previous_action: ActionBlock | None = None,
        goal_radius_m: float = 0.0,
    ) -> ScoreBreakdown:
        if not rollout.states:
            raise ValueError("cannot score an empty rollout")
        if goal_radius_m < 0:
            raise ValueError("goal_radius_m must be non-negative")

        initial_distance = _distance(initial_robot_xy, goal_xy)
        distances = [_distance(state.robot.xy, goal_xy) for state in rollout.states]
        terminal_distance = distances[-1]
        closest_distance = min(distances)
        goal_cost = _goal_cost(distances, goal_radius_m)
        progress = initial_distance - closest_distance
        mean_forward = sum(abs(action.forward_mps) for action in rollout.actions) / len(
            rollout.actions
        )
        active_fraction = min(
            1.0,
            mean_forward / max(self.config.active_forward_threshold_mps, 1e-12),
        )
        lack_of_progress = max(0.0, self.config.minimum_progress_m - progress) * active_fraction
        risk = max(state.failure_risk for state in rollout.states)
        effort = sum(
            action.forward_mps**2 + action.yaw_rate_rps**2 for action in rollout.actions
        ) / len(rollout.actions)
        command_change = _command_change(rollout.actions, previous_action)

        weights = self.config.weights
        total = (
            weights.goal_distance * goal_cost
            + weights.failure_risk * risk
            + weights.stall * lack_of_progress
            + weights.control_effort * effort
            + weights.command_change * command_change
        )
        return ScoreBreakdown(
            candidate_id=rollout.candidate_id,
            total=total,
            goal_cost_m=goal_cost,
            terminal_goal_distance_m=terminal_distance,
            closest_goal_distance_m=closest_distance,
            maximum_failure_risk=risk,
            lack_of_progress_m=lack_of_progress,
            control_effort=effort,
            command_change=command_change,
        )


def _goal_cost(distances: list[float], goal_radius_m: float) -> float:
    total = 0.0
    for distance in distances:
        if distance <= goal_radius_m:
            break
        total += distance - goal_radius_m
    return total / len(distances)


def _distance(left: Point2D, right: Point2D) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _command_change(actions: tuple[ActionBlock, ...], previous_action: ActionBlock | None) -> float:
    if not actions:
        return 0.0
    comparisons: list[tuple[ActionBlock, ActionBlock]] = []
    if previous_action is not None:
        comparisons.append((previous_action, actions[0]))
    comparisons.extend(pairwise(actions))
    if not comparisons:
        return 0.0
    return sum(
        (right.forward_mps - left.forward_mps) ** 2 + (right.yaw_rate_rps - left.yaw_rate_rps) ** 2
        for left, right in comparisons
    ) / len(comparisons)
