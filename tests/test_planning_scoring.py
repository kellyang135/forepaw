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
    fields = RolloutScorer().score(
        rollout("plain", terminal_x=0.4),
        goal_xy=(1.0, 0.0),
        initial_robot_xy=(0.0, 0.0),
    ).__dataclass_fields__

    assert "contact" not in fields
