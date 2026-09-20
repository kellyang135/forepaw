"""Closed-loop receding-horizon runner: plan, lock, execute one block, observe, check.

The runner is the only place that joins the simulator, the planner and the
surprise monitor.  Planning sees only runtime-visible inputs (three observations,
two prior commands, the goal and a readout of the current latent).  Simulator
labels are read after each block solely to log ground truth and to decide
evaluation-side termination (goal reached, fall, out of bounds).
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from go2wm.contracts import ActionCommand, Goal2D, ResetRequest, RGBObservation
from go2wm.integration.adapters import action_to_model, action_to_simulator
from go2wm.model import ModelInput, Point2D
from go2wm.planning import PlanningRequest, RecedingHorizonPlanner
from go2wm.sim import SimulatorAdapter
from go2wm.telemetry import JsonlTelemetrySink

from .monitor import PlanExecutionMonitor
from .surprise import SurpriseEvent, SurpriseStopGuard

Encoder = Callable[[RGBObservation], Sequence[float]]
Locator = Callable[[Sequence[float]], Point2D]
Perturbation = Callable[[int, SimulatorAdapter], None]


class LoopError(RuntimeError):
    """Raised when the runner cannot start or continue safely."""


@dataclass(frozen=True, slots=True)
class LoopConfig:
    max_blocks: int = 40
    history_frames: int = 3
    idle_blocks_before_stall: int = 4
    run_id: str = "closed-loop"
    evidence_scope: str = "unspecified"
    fallback_level: str = "unspecified"
    model_label: str = "unspecified"
    render_hints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_blocks < 1 or self.history_frames < 2 or self.idle_blocks_before_stall < 1:
            raise ValueError("invalid loop configuration")


@dataclass(frozen=True, slots=True)
class BlockOutcome:
    block: int
    candidate_id: str
    first_action: tuple[float, float]
    surprise: SurpriseEvent
    planning_latency_s: float
    robot_prediction_error_m: float
    bundle_id: str
    selection_reason: str
    score_total: float
    score_parts: dict[str, float]


@dataclass(frozen=True, slots=True)
class LoopSummary:
    reason: str
    blocks: tuple[BlockOutcome, ...]
    sim_time_s: float

    @property
    def planning_latency_p50_s(self) -> float | None:
        if not self.blocks:
            return None
        ordered = sorted(item.planning_latency_s for item in self.blocks)
        return ordered[len(ordered) // 2]


class ClosedLoopRunner:
    """Execute only the first block of each locked plan, then replan."""

    def __init__(
        self,
        simulator: SimulatorAdapter,
        planner: RecedingHorizonPlanner,
        guard: SurpriseStopGuard,
        *,
        encode: Encoder,
        locate: Locator,
        sink: JsonlTelemetrySink | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.planner = planner
        self.guard = guard
        self.monitor = PlanExecutionMonitor(guard)
        self.encode = encode
        self.locate = locate
        self.sink = sink
        self.config = config or LoopConfig()
        manifest = planner.model_bundle.manifest
        if manifest.history_frames != self.config.history_frames:
            raise LoopError("loop history does not match the model bundle")
        if abs(manifest.block_duration_s - simulator.block_duration_s) > 1e-9:
            raise LoopError("simulator block duration does not match the model bundle")
        self._stop_requested = threading.Event()
        self._stop_acknowledged = threading.Event()
        self._running = threading.Event()
        self._stop_reason = "external_stop"
        self._stop_lock = threading.Lock()

    @property
    def running(self) -> bool:
        """Whether a closed-loop run currently owns the simulator."""

        return self._running.is_set()

    def request_stop(self, reason: str = "skill_requested", *, timeout_s: float = 5.0) -> bool:
        """Request a fail-closed stop and wait for the runner to apply zero velocity.

        The simulator adapter is synchronous, so an in-flight 0.5-second block
        finishes before the zero block is applied.  No second caller writes to
        the simulator; the runner remains the sole motion publisher.
        """

        if not reason.strip():
            raise ValueError("stop reason must not be empty")
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        if not self._running.is_set():
            return False
        with self._stop_lock:
            self._stop_reason = reason
        self._stop_requested.set()
        return self._stop_acknowledged.wait(timeout_s)

    def run(
        self,
        reset: ResetRequest,
        goal: Goal2D,
        *,
        perturb: Perturbation | None = None,
    ) -> LoopSummary:
        if self._running.is_set():
            raise LoopError("closed-loop runner already has an active run")
        self._stop_requested.clear()
        self._stop_acknowledged.clear()
        self._running.set()
        report = self.simulator.reset(reset)
        if not report.settled:
            self._running.clear()
            raise LoopError(f"episode {reset.episode_id!r} did not settle after reset")
        observations: list[RGBObservation] = [report.final_observation]
        commands: list[ActionCommand] = []
        for _ in range(self.config.history_frames - 1):
            warmup = self.simulator.execute_block(
                ActionCommand.stopped(self.simulator.block_duration_s)
            )
            commands.append(warmup.action)
            observations.append(warmup.end_observation)

        weights = self.planner.scorer.config.weights
        manifest = self.planner.model_bundle.manifest
        if self.sink is not None:
            self.sink.run_start(
                run_id=self.config.run_id,
                episode_id=reset.episode_id,
                scene=self.simulator.scene_config,
                camera_id=self.simulator.camera_config.camera_id,
                block_duration_s=self.simulator.block_duration_s,
                goal=goal,
                bundle_id=manifest.bundle_id,
                latent_dim=manifest.latent_dim,
                horizon_blocks=manifest.horizon_blocks,
                weights=weights,
                surprise_threshold=self.guard.calibration.threshold,
                surprise_metric=self.guard.calibration.metric,
                initial_labels=self.simulator.labels(),
                history=observations[-self.config.history_frames :],
                history_actions=commands[-(self.config.history_frames - 1) :],
                evidence_scope=self.config.evidence_scope,
                fallback_level=self.config.fallback_level,
                model_label=self.config.model_label,
                render_hints=self.config.render_hints,
            )

        outcomes: list[BlockOutcome] = []
        previous = None
        idle = 0
        reason = "max_blocks"
        sim_time = observations[-1].sim_time_s
        for block in range(self.config.max_blocks):
            if self._stop_requested.is_set():
                stopped = self._apply_requested_stop(block)
                sim_time = stopped.end_observation.sim_time_s
                reason = "external_stop"
                break
            n = self.config.history_frames
            model_input = ModelInput(
                observations=tuple(observations[-n:]),
                command_history=tuple(action_to_model(c) for c in commands[-(n - 1) :]),
            )
            started = time.perf_counter()
            current_xy = self.locate(self.encode(observations[-1]))
            plan = self.planner.plan(
                PlanningRequest(
                    model_input=model_input,
                    goal_xy=(goal.x_m, goal.y_m),
                    current_robot_xy=current_xy,
                    previous_action=previous,
                    goal_radius_m=goal.radius_m,
                )
            )
            latency = time.perf_counter() - started
            self.monitor.arm_plan(plan)
            if self.sink is not None:
                self.sink.plan_locked(
                    block=block,
                    plan=plan,
                    weights=weights,
                    planning_latency_s=latency,
                    current_estimate_xy=current_xy,
                )
            if perturb is not None:
                perturb(block, self.simulator)

            requested = action_to_simulator(plan.first_action)
            transition = self.simulator.execute_block(requested)
            observations.append(transition.end_observation)
            commands.append(transition.action)
            sim_time = transition.end_observation.sim_time_s
            observed_latent = self.encode(transition.end_observation)
            event = self.monitor.observe(observed_latent)
            predicted_next = plan.selected.rollout.states[0]
            end = transition.end_labels.robot_pose
            error = math.hypot(
                predicted_next.robot.x_m - end.x_m, predicted_next.robot.y_m - end.y_m
            )
            if self.sink is not None:
                self.sink.block_executed(
                    block=block,
                    transition=transition,
                    requested=requested,
                    surprise=event,
                    observed_latent=observed_latent,
                    predicted_next=predicted_next,
                )
            outcomes.append(
                BlockOutcome(
                    block=block,
                    candidate_id=plan.selected.rollout.candidate_id,
                    first_action=(plan.first_action.forward_mps, plan.first_action.yaw_rate_rps),
                    surprise=event,
                    planning_latency_s=latency,
                    robot_prediction_error_m=error,
                    bundle_id=plan.bundle_id,
                    selection_reason=plan.selection_reason,
                    score_total=plan.selected.score.total,
                    score_parts={
                        "goal": weights.goal_distance * plan.selected.score.goal_term_m,
                        "risk": weights.failure_risk
                        * plan.selected.score.maximum_failure_risk,
                        "stall": weights.stall * plan.selected.score.lack_of_progress_m,
                        "effort": weights.control_effort
                        * plan.selected.score.control_effort,
                        "change": weights.command_change * plan.selected.score.command_change,
                    },
                )
            )
            previous = plan.first_action

            moved = plan.first_action.forward_mps != 0.0 or plan.first_action.yaw_rate_rps != 0.0
            idle = 0 if moved or plan.selection_reason != "minimum_score" else idle + 1
            if self._stop_requested.is_set():
                stopped = self._apply_requested_stop(block + 1)
                sim_time = stopped.end_observation.sim_time_s
                reason = "external_stop"
                break
            if event.stop_commanded or self.guard.alarm_latched:
                reason = "surprise_stop"
                break
            if transition.events.fell:
                reason = "fall"
                break
            if transition.events.out_of_bounds:
                reason = "out_of_bounds"
                break
            if math.hypot(end.x_m - goal.x_m, end.y_m - goal.y_m) <= goal.radius_m:
                reason = "goal_reached"
                break
            if idle >= self.config.idle_blocks_before_stall:
                reason = "stalled"
                break

        if self.sink is not None:
            self.sink.run_end(reason=reason, blocks=len(outcomes), sim_time_s=sim_time)
        self._running.clear()
        if self._stop_requested.is_set():
            self._stop_acknowledged.set()
        return LoopSummary(reason=reason, blocks=tuple(outcomes), sim_time_s=sim_time)

    def _apply_requested_stop(self, block: int) -> Any:
        with self._stop_lock:
            reason = self._stop_reason
        transition = self.simulator.execute_block(
            ActionCommand.stopped(self.simulator.block_duration_s)
        )
        if self.sink is not None:
            self.sink.stop_executed(block=block, reason=reason, transition=transition)
        self._stop_acknowledged.set()
        return transition
