"""Concrete controller backend shared by the dimOS transport and direct tests.

The dimOS process never publishes simulator motion itself.  It calls this
backend through the local controller service, and the ``ClosedLoopRunner``
remains the sole owner of reset, plan-lock, block execution, and stop commands.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from go2wm.contracts import ActionCommand, Goal2D
from go2wm.integration.adapters import action_to_model
from go2wm.model import ActionBlock, ModelInput
from go2wm.planning import CandidateSequence, RecedingHorizonPlanner, RolloutScorer
from go2wm.runtime import SurpriseStopGuard
from go2wm.runtime.loop import ClosedLoopRunner, LoopConfig
from go2wm.telemetry import JsonlTelemetrySink

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ControllerRuntimeConfig:
    """Immutable deployment inputs for one local controller service."""

    simulator: Literal["fake", "mujoco"] = "fake"
    controller: Literal["mjlab", "rl_sar"] = "mjlab"
    model: Literal["reference", "bundle"] = "reference"
    scenario: Literal["push", "detour", "anomaly"] = "push"
    output_root: Path = Path("runs/dimos")
    bundle_path: Path | None = None
    policy_path: Path | None = None
    robot_xml_path: Path | None = None
    deploy_yaml_path: Path | None = None
    menagerie_cache: Path = Path("/tmp/go2wm-menagerie-cache")
    lewm_repo: Path | None = None
    device: str = "cpu"
    max_blocks: int = 30
    image_size: int = 32
    goal_radius_m: float = 0.25
    surprise_threshold: float | None = None
    stop_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.model == "bundle" and self.bundle_path is None:
            raise ValueError("model='bundle' requires bundle_path")
        if self.simulator == "mujoco" and self.policy_path is None:
            raise ValueError("simulator='mujoco' requires policy_path")
        if (
            self.simulator == "mujoco"
            and self.controller == "mjlab"
            and self.robot_xml_path is None
        ):
            raise ValueError("the MjLab controller requires robot_xml_path")
        if self.max_blocks < 1:
            raise ValueError("max_blocks must be positive")
        if self.goal_radius_m <= 0 or not math.isfinite(self.goal_radius_m):
            raise ValueError("goal_radius_m must be positive and finite")
        if self.stop_timeout_s <= 0:
            raise ValueError("stop_timeout_s must be positive")


@dataclass(slots=True)
class _Runtime:
    runner: ClosedLoopRunner
    reset: Any
    simulator: Any
    sink: JsonlTelemetrySink | None
    telemetry_path: Path | None

    def close(self) -> None:
        if self.sink is not None:
            self.sink.close()
        if hasattr(self.simulator, "close"):
            self.simulator.close()


class RunnerControllerBackend:
    """Fail-closed backend used by the local HTTP controller service.

    ``plan_to`` owns one complete receding-horizon run.  Every internal cycle
    still predicts 64 six-block candidates, durably locks the selected future,
    executes only block zero, observes, and replans.  A concurrent
    ``stop_motion`` request is relayed to the active runner through an event;
    the runner itself applies the zero block, preserving single-writer motion.
    """

    def __init__(self, config: ControllerRuntimeConfig) -> None:
        self.config = config
        self._operation_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_runner: ClosedLoopRunner | None = None
        self._run_number = 0

    def health(self) -> JsonObject:
        with self._active_lock:
            active = self._active_runner is not None
        return {
            "status": "ready",
            "active_run": active,
            "simulator": self.config.simulator,
            "controller": self.config.controller,
            "model": self.config.model,
            "scenario": self.config.scenario,
            "scope": self._scope_label(),
        }

    def imagine(self, candidate_commands: list[list[JsonObject]]) -> JsonObject:
        candidates = _candidate_sequences(candidate_commands)
        with self._operation_lock:
            runtime = self._build_runtime(with_telemetry=False)
            try:
                report = runtime.simulator.reset(runtime.reset)
                if not report.settled:
                    raise RuntimeError("simulator did not settle; no prediction was produced")
                observations = [report.final_observation]
                commands: list[ActionCommand] = []
                for _ in range(2):
                    transition = runtime.simulator.execute_block(
                        ActionCommand.stopped(runtime.simulator.block_duration_s)
                    )
                    observations.append(transition.end_observation)
                    commands.append(transition.action)
                model_input = ModelInput(
                    observations=tuple(observations),
                    command_history=tuple(action_to_model(command) for command in commands),
                )
                started = _monotonic()
                rollouts = runtime.runner.planner.model_bundle.predict_candidates(
                    model_input, candidates
                )
                latency = _monotonic() - started
                return {
                    "status": "imagined",
                    "motion_executed": False,
                    "bundle_id": runtime.runner.planner.model_bundle.manifest.bundle_id,
                    "candidate_count": len(rollouts),
                    "latency_s": latency,
                    "candidates": [_rollout_payload(item) for item in rollouts],
                    "scope": self._scope_label(),
                }
            finally:
                runtime.close()

    def plan_to(self, x_m: float, y_m: float) -> JsonObject:
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError("goal coordinates must be finite")
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("controller is busy; no second motion publisher was started")
        runtime: _Runtime | None = None
        try:
            runtime = self._build_runtime(with_telemetry=True)
            with self._active_lock:
                self._active_runner = runtime.runner
            summary = runtime.runner.run(
                runtime.reset,
                Goal2D(float(x_m), float(y_m), self.config.goal_radius_m),
            )
            blocks = [_block_payload(item) for item in summary.blocks]
            return {
                "status": "completed" if summary.reason != "external_stop" else "stopped",
                "reason": summary.reason,
                "bundle_id": runtime.runner.planner.model_bundle.manifest.bundle_id,
                "goal": {"x_m": float(x_m), "y_m": float(y_m)},
                "blocks_executed": len(blocks),
                "first_action": blocks[0]["first_action"] if blocks else None,
                "first_selected_candidate": blocks[0]["selected_candidate"]
                if blocks
                else None,
                "first_score": blocks[0]["score"] if blocks else None,
                "planning_latency_p50_s": summary.planning_latency_p50_s,
                "sim_time_s": summary.sim_time_s,
                "blocks": blocks,
                "telemetry_path": None
                if runtime.telemetry_path is None
                else str(runtime.telemetry_path),
                "scope": self._scope_label(),
            }
        finally:
            with self._active_lock:
                self._active_runner = None
            if runtime is not None:
                runtime.close()
            self._operation_lock.release()

    def stop(self, reason: str) -> JsonObject:
        with self._active_lock:
            runner = self._active_runner
        if runner is None:
            return {
                "status": "idle",
                "reason": reason,
                "stop_acknowledged": True,
                "detail": "no active motion run",
            }
        acknowledged = runner.request_stop(reason, timeout_s=self.config.stop_timeout_s)
        return {
            "status": "stopped" if acknowledged else "stop_timeout",
            "reason": reason,
            "stop_acknowledged": acknowledged,
            "detail": "zero-velocity block is recorded in telemetry" if acknowledged else None,
        }

    def _build_runtime(self, *, with_telemetry: bool) -> _Runtime:
        # The UI helpers are deliberately reused so dimOS and the live viewer
        # cannot silently diverge in simulator/controller/scenario setup.
        from go2wm.ui.__main__ import _fake_world, _mujoco_world, _scenario

        if self.config.simulator == "mujoco":
            args = SimpleNamespace(
                policy=None if self.config.policy_path is None else str(self.config.policy_path),
                controller=self.config.controller,
                robot_xml=None
                if self.config.robot_xml_path is None
                else str(self.config.robot_xml_path),
                deploy_yaml=None
                if self.config.deploy_yaml_path is None
                else str(self.config.deploy_yaml_path),
                menagerie_cache=str(self.config.menagerie_cache),
            )
            simulator = _mujoco_world(args)
        else:
            simulator = _fake_world(self.config.image_size)

        if self.config.model == "reference":
            from go2wm.ui.reference import LABEL, reference_bundle

            threshold = self.config.surprise_threshold
            if threshold is None:
                threshold = 0.10 if self.config.simulator == "mujoco" else 0.05
            contact = 0.55 if self.config.simulator == "mujoco" else 0.31
            bundle, encoder, readout, calibration = reference_bundle(
                simulator, threshold=threshold, contact_m=contact
            )
            planner = RecedingHorizonPlanner(bundle)
            model_label = LABEL
            fallback = "F3 integration rehearsal (privileged reference model)"
        else:
            from go2wm.learning.bundle_io import load_bundle

            context: dict[str, Any] = {"device": self.config.device}
            if self.config.lewm_repo is not None:
                context["lewm_repo"] = str(self.config.lewm_repo)
            loaded = load_bundle(self.config.bundle_path, context=context)
            planner = RecedingHorizonPlanner(
                loaded.model_bundle,
                candidates=loaded.candidates,
                scorer=RolloutScorer(loaded.scoring),
                safety_policy=loaded.safety,
            )
            encoder = loaded.model_bundle.backend.encoder
            readout = loaded.readout
            calibration = loaded.surprise
            model_label = f"bundle {loaded.bundle_id}"
            fallback = "unassessed published model bundle"

        self._run_number += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"dimos-{self.config.scenario}-{timestamp}-{self._run_number:03d}"
        telemetry_path: Path | None = None
        sink: JsonlTelemetrySink | None = None
        if with_telemetry:
            telemetry_path = self.config.output_root / run_id / "ui.jsonl"
            sink = JsonlTelemetrySink(telemetry_path)
        stops: list[str] = []
        guard = SurpriseStopGuard(calibration, stop_callback=stops.append)
        runner = ClosedLoopRunner(
            simulator,
            planner,
            guard,
            encode=encoder.encode,
            locate=lambda latent: readout.decode(tuple(latent), step=1).robot.xy,
            sink=sink,
            config=LoopConfig(
                max_blocks=self.config.max_blocks,
                run_id=run_id,
                evidence_scope=self._scope_label(),
                fallback_level=fallback,
                model_label=model_label,
                render_hints=(
                    {"robot_scale": 1.0, "box_size_m": 0.40, "robot_body_m": 0.7}
                    if self.config.simulator == "mujoco"
                    else {"robot_scale": 0.45, "box_size_m": 0.28, "robot_body_m": 0.3}
                ),
            ),
        )
        reset, _ = _scenario(self.config.scenario, self.config.simulator)
        return _Runtime(runner, reset, simulator, sink, telemetry_path)

    def _scope_label(self) -> str:
        if self.config.simulator == "mujoco" and self.config.model == "bundle":
            return "dimOS transport to true-Go2 MjLab simulator with a published bundle"
        if self.config.simulator == "mujoco":
            return (
                "dimOS integration rehearsal on true-Go2 MjLab simulation with a privileged "
                "reference model; not learned-model evidence"
            )
        return "dimOS software integration rehearsal on the deterministic fake simulator"


def _candidate_sequences(raw: list[list[JsonObject]]) -> tuple[CandidateSequence, ...]:
    if len(raw) > 64:
        raise ValueError("imagine accepts at most 64 candidates")
    candidates: list[CandidateSequence] = []
    for candidate_index, blocks in enumerate(raw):
        if len(blocks) != 6:
            raise ValueError("each candidate must contain exactly six 0.5-second blocks")
        actions = []
        for block in blocks:
            forward = float(block["forward_velocity_mps"])
            yaw = float(block["yaw_rate_rps"])
            duration = float(block.get("duration_s", 0.5))
            if not 0.0 <= forward <= 0.6:
                raise ValueError("forward_velocity_mps must be within [0.0, 0.6]")
            if not -1.2 <= yaw <= 1.2:
                raise ValueError("yaw_rate_rps must be within [-1.2, 1.2]")
            if duration != 0.5:
                raise ValueError("duration_s must equal 0.5")
            actions.append(ActionBlock(forward, yaw, duration))
        candidates.append(
            CandidateSequence(
                candidate_id=f"external_{candidate_index:02d}",
                family="external",
                actions=tuple(actions),
            )
        )
    return tuple(candidates)


def _rollout_payload(rollout: Any) -> JsonObject:
    return {
        "candidate_id": rollout.candidate_id,
        "family": rollout.family,
        "states": [
            {
                "step": state.step,
                "robot": {
                    "x_m": state.robot.x_m,
                    "y_m": state.robot.y_m,
                    "yaw_rad": state.robot.yaw_rad,
                },
                "objects": [
                    {"id": item.object_id, "x_m": item.x_m, "y_m": item.y_m}
                    for item in state.objects
                ],
                "failure_risk": state.failure_risk,
            }
            for state in rollout.states
        ],
    }


def _block_payload(block: Any) -> JsonObject:
    return {
        "block": block.block,
        "bundle_id": block.bundle_id,
        "selected_candidate": block.candidate_id,
        "selection_reason": block.selection_reason,
        "first_action": {
            "forward_velocity_mps": block.first_action[0],
            "yaw_rate_rps": block.first_action[1],
            "duration_s": 0.5,
        },
        "score": {"total": block.score_total, "parts": block.score_parts},
        "planning_latency_s": block.planning_latency_s,
        "surprise": {
            "status": block.surprise.status,
            "discrepancy": block.surprise.discrepancy,
            "stop_commanded": block.surprise.stop_commanded,
        },
        "robot_prediction_error_m": block.robot_prediction_error_m,
    }


def _monotonic() -> float:
    import time

    return time.perf_counter()
