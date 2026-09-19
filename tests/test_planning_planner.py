from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from go2wm.model import (
    ActionBlock,
    BundleManifest,
    CandidateRollout,
    ModelBundle,
    ModelInput,
    PredictedState,
    RobotState,
    RuntimeRequirements,
)
from go2wm.planning import (
    CandidateSequence,
    PlanningError,
    PlanningRequest,
    RecedingHorizonPlanner,
)


def make_manifest() -> BundleManifest:
    return BundleManifest(
        bundle_id="planner-bundle",
        encoder_version="enc-1",
        predictor_version="pred-1",
        readout_version="read-1",
        normalization_version="norm-1",
        surprise_calibration_version="surprise-1",
        latent_dim=2,
        horizon_blocks=2,
    )


class PlannerBackend:
    def __init__(self, *, risks: dict[str, float] | None = None, omit: str | None = None):
        self._manifest = make_manifest()
        self.risks = risks or {}
        self.omit = omit

    @property
    def manifest(self):
        return self._manifest

    def predict_candidates(self, model_input, candidates):
        del model_input
        output = []
        for candidate in candidates:
            if candidate.candidate_id == self.omit:
                continue
            terminal_x = {"advance": 0.8, "turn": 0.3, "stop_hold": 0.0}[
                candidate.candidate_id
            ]
            risk = self.risks.get(candidate.candidate_id, 0.0)
            output.append(
                CandidateRollout(
                    candidate.candidate_id,
                    candidate.family,
                    candidate.actions,
                    (
                        PredictedState(
                            1,
                            RobotState(terminal_x / 2, 0.0, 0.0),
                            failure_risk=risk,
                            latent=(0.1, 0.2),
                        ),
                        PredictedState(
                            2,
                            RobotState(terminal_x, 0.0, 0.0),
                            failure_risk=risk,
                            latent=(0.2, 0.3),
                        ),
                    ),
                    self.manifest.bundle_id,
                )
            )
        return tuple(output)


def candidates() -> tuple[CandidateSequence, ...]:
    advance = ActionBlock(0.4, 0.0)
    turn = ActionBlock(0.2, 0.5)
    stop = ActionBlock(0.0, 0.0)
    return (
        CandidateSequence("advance", "approach_push", (advance, advance)),
        CandidateSequence("turn", "turn_adjust", (turn, turn)),
        CandidateSequence("stop_hold", "stop", (stop, stop)),
    )


def request() -> PlanningRequest:
    return PlanningRequest(
        model_input=ModelInput(
            observations=("old", "middle", "new"),
            command_history=(ActionBlock(0.1, 0.0), ActionBlock(0.1, 0.0)),
        ),
        goal_xy=(1.0, 0.0),
        current_robot_xy=(0.0, 0.0),
    )


def make_planner(backend: PlannerBackend) -> RecedingHorizonPlanner:
    requirements = RuntimeRequirements(latent_dim=2, horizon_blocks=2)
    return RecedingHorizonPlanner(
        ModelBundle(backend, requirements),
        candidates=candidates(),
        clock=lambda: datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc),
    )


def test_plan_locks_all_rankings_but_authorizes_only_first_block() -> None:
    result = make_planner(PlannerBackend()).plan(request())

    assert result.selected.rollout.candidate_id == "advance"
    assert result.first_action == ActionBlock(0.4, 0.0)
    assert len(result.rankings) == 3
    assert result.bundle_id == "planner-bundle"
    assert result.locked_at_utc == "2026-09-19T12:00:00.000+00:00"
    assert result.predicted_next_latent == (0.1, 0.2)


def test_all_risky_moving_plans_force_stop_even_if_stop_scores_worse() -> None:
    result = make_planner(
        PlannerBackend(risks={"advance": 0.9, "turn": 0.8, "stop_hold": 0.7})
    ).plan(request())

    assert result.selected.rollout.candidate_id == "stop_hold"
    assert result.selection_reason == "all_moving_candidates_exceed_risk_limit"
    assert result.first_action == ActionBlock(0.0, 0.0)


def test_missing_prediction_fails_closed() -> None:
    planner = make_planner(PlannerBackend(omit="turn"))

    with pytest.raises(PlanningError, match=r"missing=\['turn'\]"):
        planner.plan(request())


def test_backend_rollout_with_wrong_bundle_is_rejected() -> None:
    backend = PlannerBackend()
    original = backend.predict_candidates

    def wrong_bundle(model_input, candidate_values):
        return tuple(
            replace(rollout, bundle_id="stale-bundle")
            for rollout in original(model_input, candidate_values)
        )

    backend.predict_candidates = wrong_bundle  # type: ignore[method-assign]

    with pytest.raises(Exception, match="stale-bundle"):
        make_planner(backend).plan(request())
