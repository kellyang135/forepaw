from __future__ import annotations

import pytest

from go2wm.model import (
    ActionBlock,
    BundleCompatibilityError,
    BundleManifest,
    CandidateRollout,
    PredictedState,
    RobotState,
)
from go2wm.planning import PlanResult, RankedCandidate, ScoreBreakdown
from go2wm.runtime import (
    PlanExecutionMonitor,
    SurpriseCalibration,
    SurpriseStopGuard,
    calibrate_surprise,
    latent_discrepancy,
)


def calibration(*, consecutive: int = 1) -> SurpriseCalibration:
    return SurpriseCalibration(
        calibration_version="surprise-1",
        bundle_id="bundle-1",
        latent_dim=2,
        metric="rmse",
        threshold=0.2,
        consecutive_breaches=consecutive,
        normal_sample_count=100,
    )


def test_rmse_and_cosine_metrics_have_expected_scale() -> None:
    assert latent_discrepancy((0.0, 0.0), (0.3, 0.4), metric="rmse") == pytest.approx(
        (0.125) ** 0.5
    )
    assert latent_discrepancy((1.0, 0.0), (0.0, 1.0), metric="cosine") == pytest.approx(
        1.0
    )


def test_calibration_uses_conservative_nearest_rank_quantile() -> None:
    fitted = calibrate_surprise(
        [((0.0,), (value,)) for value in (0.1, 0.2, 0.3, 0.4)],
        bundle_id="bundle-1",
        calibration_version="surprise-1",
        quantile=0.75,
    )

    assert fitted.threshold == pytest.approx(0.3)
    assert fitted.normal_sample_count == 4


def test_alarm_commands_stop_once_and_latches() -> None:
    stops: list[str] = []
    guard = SurpriseStopGuard(calibration(), stop_callback=stops.append)
    guard.arm((0.0, 0.0), bundle_id="bundle-1", plan_locked_at_utc="locked")

    event = guard.observe((1.0, 1.0))
    repeated = guard.observe((0.0, 0.0))

    assert event.status == "alarm"
    assert event.stop_commanded is True
    assert event.plan_locked_at_utc == "locked"
    assert repeated.status == "alarm_latched"
    assert repeated.stop_commanded is False
    assert len(stops) == 1


def test_consecutive_breach_policy_resets_after_normal_observation() -> None:
    stops: list[str] = []
    guard = SurpriseStopGuard(calibration(consecutive=2), stop_callback=stops.append)

    guard.arm((0.0, 0.0), bundle_id="bundle-1")
    assert guard.observe((1.0, 1.0)).status == "breach"
    guard.arm((0.0, 0.0), bundle_id="bundle-1")
    assert guard.observe((0.1, 0.1)).status == "normal"
    guard.arm((0.0, 0.0), bundle_id="bundle-1")
    assert guard.observe((1.0, 1.0)).status == "breach"
    guard.arm((0.0, 0.0), bundle_id="bundle-1")
    assert guard.observe((1.0, 1.0)).status == "alarm"
    assert len(stops) == 1


def test_prediction_is_single_use_and_bundle_mismatch_is_rejected() -> None:
    guard = SurpriseStopGuard(calibration(), stop_callback=lambda reason: None)
    guard.arm((0.0, 0.0), bundle_id="bundle-1")

    assert guard.observe((0.0, 0.0)).status == "normal"
    assert guard.observe((0.0, 0.0)).status == "unarmed"
    with pytest.raises(BundleCompatibilityError):
        guard.arm((0.0, 0.0), bundle_id="stale")


def test_calibration_must_match_entire_model_bundle() -> None:
    manifest = BundleManifest(
        bundle_id="bundle-1",
        encoder_version="enc",
        predictor_version="pred",
        readout_version="read",
        normalization_version="norm",
        surprise_calibration_version="surprise-2",
        latent_dim=2,
    )

    with pytest.raises(BundleCompatibilityError, match="surprise-1"):
        calibration().validate_manifest(manifest)


def test_execution_monitor_arms_from_locked_plan_and_consumes_prediction() -> None:
    action = ActionBlock(0.2, 0.0)
    rollout = CandidateRollout(
        "advance",
        "approach_push",
        (action,),
        (PredictedState(1, RobotState(0.1, 0.0, 0.0), latent=(0.0, 0.0)),),
        "bundle-1",
    )
    score = ScoreBreakdown("advance", 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    ranked = RankedCandidate(1, rollout, score)
    plan = PlanResult(
        status="planned",
        selection_reason="minimum_score",
        bundle_id="bundle-1",
        goal_xy=(1.0, 0.0),
        locked_at_utc="2026-09-19T12:00:00.000+00:00",
        selected=ranked,
        rankings=(ranked,),
    )
    guard = SurpriseStopGuard(calibration(), stop_callback=lambda reason: None)
    monitor = PlanExecutionMonitor(guard)

    armed = monitor.arm_plan(plan)
    event = monitor.observe((0.0, 0.0))

    assert armed.candidate_id == "advance"
    assert event.status == "normal"
    assert event.plan_locked_at_utc == plan.locked_at_utc
    assert monitor.armed_plan is None
