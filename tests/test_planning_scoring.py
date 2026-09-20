from __future__ import annotations

import pytest

from go2wm.model import ActionBlock, CandidateRollout, PredictedState, RobotState
from go2wm.planning import RolloutScorer


def rollout(
    candidate_id: str,
    *,
    terminal_x: float,
    risk: float = 0.0,
    action: ActionBlock | None = None,
) -> CandidateRollout:
    action = action or ActionBlock(0.3, 0.0)
    return CandidateRollout(
        candidate_id=candidate_id,
        family="test",
        actions=(action, action),
        states=(
            PredictedState(1, RobotState(terminal_x / 2.0, 0.0, 0.0), failure_risk=risk),
            PredictedState(2, RobotState(terminal_x, 0.0, 0.0), failure_risk=risk),
        ),
        bundle_id="bundle",
    )


def test_goal_progress_wins_when_risk_is_equal() -> None:
    scorer = RolloutScorer()
    near = scorer.score(
        rollout("near", terminal_x=0.8), goal_xy=(1.0, 0.0), initial_robot_xy=(0.0, 0.0)
    )
    far = scorer.score(
        rollout("far", terminal_x=0.2), goal_xy=(1.0, 0.0), initial_robot_xy=(0.0, 0.0)
    )

    assert near.total < far.total
    assert near.lack_of_progress_m == 0.0


def test_large_failure_risk_can_outweigh_shorter_path() -> None:
    scorer = RolloutScorer()
    risky = scorer.score(
        rollout("risky", terminal_x=0.9, risk=0.8),
        goal_xy=(1.0, 0.0),
        initial_robot_xy=(0.0, 0.0),
    )
    safe = scorer.score(
        rollout("safe", terminal_x=0.5, risk=0.0),
        goal_xy=(1.0, 0.0),
        initial_robot_xy=(0.0, 0.0),
    )

    assert safe.total < risky.total


def test_stalled_moving_candidate_is_penalized() -> None:
    score = RolloutScorer().score(
        rollout("stalled", terminal_x=0.0),
        goal_xy=(1.0, 0.0),
        initial_robot_xy=(0.0, 0.0),
    )

    assert score.lack_of_progress_m == pytest.approx(0.08)


def test_contact_is_not_a_score_input() -> None:
    fields = (
        RolloutScorer()
        .score(
            rollout("plain", terminal_x=0.4),
            goal_xy=(1.0, 0.0),
            initial_robot_xy=(0.0, 0.0),
        )
        .__dataclass_fields__
    )

    assert "contact" not in fields


def _path(candidate_id: str, xs: list[float], forward: list[float]) -> CandidateRollout:
    actions = tuple(ActionBlock(v, 0.0) for v in forward)
    states = tuple(PredictedState(i, RobotState(x, 0.0, 0.0)) for i, x in enumerate(xs, start=1))
    return CandidateRollout(candidate_id, "test", actions, states, "bundle")


def test_arriving_now_beats_stopping_then_creeping_in_late() -> None:
    """D-028: only the first block executes, so a late arrival must not win."""

    scorer = RolloutScorer()
    common = {"goal_xy": (0.4, 0.0), "initial_robot_xy": (0.0, 0.0), "goal_radius_m": 0.2}
    # drives through the goal and beyond; the terminal state is far past it
    through = _path("through", [0.15, 0.3, 0.45, 0.6, 0.75, 0.9], [0.3] * 6)
    # stops for two blocks, then creeps and ends right on the goal
    creep = _path("creep", [0.0, 0.0, 0.1, 0.2, 0.3, 0.4], [0, 0, 0.2, 0.2, 0.2, 0.2])
    through_score = scorer.score(through, **common)
    creep_score = scorer.score(creep, **common)
    assert creep_score.terminal_goal_distance_m < through_score.terminal_goal_distance_m
    assert through_score.total < creep_score.total


def test_arrival_is_absorbing_and_radius_matters() -> None:
    scorer = RolloutScorer()
    rollout_ = _path("r", [0.1, 0.25, 0.4, 0.55, 0.7, 0.85], [0.3] * 6)
    inside_at_2 = scorer.score(
        rollout_, goal_xy=(0.4, 0.0), initial_robot_xy=(0.0, 0.0), goal_radius_m=0.2
    )
    assert inside_at_2.goal_cost_m == pytest.approx((0.3 - 0.2) / 6)
    point_goal = scorer.score(rollout_, goal_xy=(0.4, 0.0), initial_robot_xy=(0.0, 0.0))
    assert point_goal.goal_cost_m > inside_at_2.goal_cost_m
    with pytest.raises(ValueError):
        scorer.score(rollout_, goal_xy=(0.4, 0.0), initial_robot_xy=(0.0, 0.0), goal_radius_m=-1)
