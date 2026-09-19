from __future__ import annotations

import base64
import json
import math

import pytest

from go2wm.contracts import CameraConfig, Goal2D, ObjectState, Pose2D, ResetRequest
from go2wm.model import (
    BundleManifest,
    ComponentWorldModelBackend,
    ModelBundle,
    PredictedState,
    RobotState,
    RuntimeRequirements,
)
from go2wm.planning import RecedingHorizonPlanner
from go2wm.runtime import SurpriseCalibration, SurpriseStopGuard
from go2wm.runtime.loop import ClosedLoopRunner, LoopConfig
from go2wm.sim import DeterministicFakeSimulator, FakeSimulatorConfig
from go2wm.telemetry import SCHEMA, JsonlTelemetrySink, read_log

LATENT_DIM = 4


class PoseEncoder:
    """Test double that reads the simulator pose; real encoders see pixels only."""

    def __init__(self, simulator: DeterministicFakeSimulator) -> None:
        self.simulator = simulator

    def encode(self, observation):
        del observation
        pose = self.simulator.labels().robot_pose
        return (pose.x_m, pose.y_m, math.cos(pose.yaw_rad), math.sin(pose.yaw_rad))


class UnicyclePredictor:
    def rollout(self, history_latents, history_actions, candidate_actions):
        del history_actions
        x, y, c, s = history_latents[-1]
        yaw = math.atan2(s, c)
        out = []
        for action in candidate_actions:
            yaw += action.yaw_rate_rps * action.duration_s
            x += action.forward_mps * math.cos(yaw) * action.duration_s
            y += action.forward_mps * math.sin(yaw) * action.duration_s
            out.append((x, y, math.cos(yaw), math.sin(yaw)))
        return out


class PoseReadout:
    def decode(self, latent, *, step):
        return PredictedState(
            step=step, robot=RobotState(latent[0], latent[1], math.atan2(latent[3], latent[2]))
        )


def build(tmp_path, *, threshold: float, max_blocks: int = 6, log_name: str = "ui.jsonl"):
    simulator = DeterministicFakeSimulator(
        FakeSimulatorConfig(camera=CameraConfig("overhead", 8, 8))
    )
    manifest = BundleManifest(
        bundle_id="loop-test",
        encoder_version="pose",
        predictor_version="unicycle",
        readout_version="pose",
        normalization_version="none",
        surprise_calibration_version="s1",
        latent_dim=LATENT_DIM,
    )
    backend = ComponentWorldModelBackend(
        manifest, PoseEncoder(simulator), UnicyclePredictor(), PoseReadout()
    )
    bundle = ModelBundle(backend, RuntimeRequirements(latent_dim=LATENT_DIM))
    planner = RecedingHorizonPlanner(bundle)
    stops: list[str] = []
    guard = SurpriseStopGuard(
        SurpriseCalibration("s1", "loop-test", LATENT_DIM, "rmse", threshold),
        stop_callback=stops.append,
    )
    sink = JsonlTelemetrySink(tmp_path / log_name)
    runner = ClosedLoopRunner(
        simulator,
        planner,
        guard,
        encode=backend.encoder.encode,
        locate=lambda z: (z[0], z[1]),
        sink=sink,
        config=LoopConfig(max_blocks=max_blocks, run_id="test"),
    )
    reset = ResetRequest(
        episode_id="loop-0",
        scenario_seed=1,
        robot_pose=Pose2D(-1.0, 0.0, 0.0),
        objects=(ObjectState("light", "blue", Pose2D(0.0, 0.9), True),),
    )
    return simulator, runner, sink, stops, reset


def test_plan_is_durably_locked_before_each_motion_command(tmp_path) -> None:
    simulator, runner, sink, _, reset = build(tmp_path, threshold=10.0)
    original = simulator.execute_block
    checked: list[int] = []

    def guarded(action):
        text = sink.path.read_text(encoding="utf-8")
        if '"type":"run_start"' in text:
            locks = text.count('"type":"plan_locked"')
            executed = text.count('"type":"block_executed"')
            assert locks == executed + 1, "motion requested without a durable plan lock"
            checked.append(locks)
        return original(action)

    simulator.execute_block = guarded
    summary = runner.run(reset, Goal2D(1.2, 0.0, 0.15))
    sink.close()
    assert checked and len(checked) == len(summary.blocks)

    records = read_log(sink.path)
    types = [r["type"] for r in records]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert types[1:-1] == ["plan_locked", "block_executed"] * len(summary.blocks)
    assert all(r["schema"] == SCHEMA for r in records)
    assert [r["seq"] for r in records] == list(range(len(records)))

    start = records[0]
    assert len(start["history"]) == 3 and len(start["history_actions"]) == 2
    png = base64.b64decode(start["history"][0]["png"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")

    plan = records[1]
    assert len(plan["rankings"]) == 64
    assert (
        plan["selected_id"] == plan["rankings"][0]["id"]
        or plan["selection_reason"] != "minimum_score"
    )
    assert len(plan["predicted_next_latent"]) == LATENT_DIM
    total = sum(plan["rankings"][0]["parts"].values())
    assert total == pytest.approx(plan["rankings"][0]["total"], abs=1e-3)


def test_surprise_alarm_stops_after_the_offending_block(tmp_path) -> None:
    _, runner, sink, stops, reset = build(tmp_path, threshold=1e-9)
    summary = runner.run(reset, Goal2D(1.2, 0.0, 0.15))
    sink.close()
    assert summary.reason == "surprise_stop"
    assert len(summary.blocks) == 1 and stops
    records = read_log(sink.path)
    assert records[-2]["surprise"]["stop_commanded"] is True
    assert records[-1] == {**records[-1], "reason": "surprise_stop", "blocks": 1}


def test_idle_planner_is_reported_as_stalled(tmp_path) -> None:
    """A goal 0.4 m ahead: the terminal-distance score prefers stop_then_creep (D-028)."""

    _, runner, sink, _, reset = build(tmp_path, threshold=10.0, max_blocks=20)
    summary = runner.run(reset, Goal2D(-0.6, 0.0, 0.2))
    sink.close()
    assert summary.reason == "stalled"
    assert {b.candidate_id for b in summary.blocks[-4:]} <= {
        "stop_then_creep",
        "stop_then_left",
        "stop_then_right",
        "stop_hold",
    }
    assert summary.planning_latency_p50_s is not None


@pytest.mark.xfail(strict=True, reason="D-028: terminal-only goal cost stalls near the goal")
def test_short_goal_is_reached(tmp_path) -> None:
    _, runner, sink, _, reset = build(tmp_path, threshold=10.0, max_blocks=20, log_name="b.jsonl")
    summary = runner.run(reset, Goal2D(-0.6, 0.0, 0.2))
    sink.close()
    assert summary.reason == "goal_reached"


def test_log_never_overwrites_existing_evidence(tmp_path) -> None:
    (tmp_path / "ui.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        JsonlTelemetrySink(tmp_path / "ui.jsonl")


def test_records_are_strict_json(tmp_path) -> None:
    _, runner, sink, _, reset = build(tmp_path, threshold=10.0, max_blocks=2)
    runner.run(reset, Goal2D(1.2, 0.0, 0.15))
    sink.close()
    for line in sink.path.read_text(encoding="utf-8").splitlines():
        json.loads(line, parse_constant=lambda name: pytest.fail(f"non-finite {name}"))
