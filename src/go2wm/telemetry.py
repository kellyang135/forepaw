"""Append-only JSONL telemetry for the closed-loop runner and the UI viewer.

Every record is one JSON object per line with ``schema`` and ``type`` fields.
``plan_locked`` is written and fsynced before the corresponding motion command
(D-010), so the log is also the durable prediction lock.  Records separate
runtime-visible values (plans, latents, observations) from privileged simulator
labels, which appear only under ``ground_truth`` keys for display and evaluation.
"""

from __future__ import annotations

import base64
import json
import math
import os
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from go2wm.contracts import (
    ActionCommand,
    BlockTransition,
    Goal2D,
    ObjectState,
    RGBObservation,
    SceneConfig,
    StateLabels,
)
from go2wm.model import ActionBlock, PredictedState
from go2wm.planning import PlanResult, ScoreWeights
from go2wm.runtime.surprise import SurpriseEvent

SCHEMA = "go2wm.ui.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def encode_png(observation: RGBObservation) -> str:
    """Encode a packed RGB8 observation as a base64 PNG without third-party packages."""

    width, height = observation.width_px, observation.height_px
    stride = width * 3
    raw = b"".join(
        b"\x00" + observation.rgb[row * stride : (row + 1) * stride] for row in range(height)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (
            struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def _num(value: float, digits: int = 5) -> float | None:
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _latent(values: Sequence[float] | None) -> list[float | None] | None:
    return None if values is None else [_num(v) for v in values]


def _action(action: ActionBlock | ActionCommand) -> dict[str, float | None]:
    if isinstance(action, ActionCommand):
        forward, yaw = action.forward_velocity_mps, action.yaw_rate_rps
    else:
        forward, yaw = action.forward_mps, action.yaw_rate_rps
    return {
        "forward_mps": _num(forward),
        "yaw_rate_rps": _num(yaw),
        "duration_s": _num(action.duration_s),
    }


def _object_label(item: ObjectState) -> dict[str, Any]:
    return {
        "id": item.object_id,
        "appearance": item.appearance_class,
        "x": _num(item.pose.x_m),
        "y": _num(item.pose.y_m),
        "yaw": _num(item.pose.yaw_rad),
        "movable": item.movable,
    }


def _labels(labels: StateLabels) -> dict[str, Any]:
    pose = labels.robot_pose
    return {
        "sim_time_s": _num(labels.sim_time_s),
        "robot": {"x": _num(pose.x_m), "y": _num(pose.y_m), "yaw": _num(pose.yaw_rad)},
        "objects": [_object_label(item) for item in labels.objects],
        "fallen": labels.fallen,
    }


def _predicted(state: PredictedState) -> dict[str, Any]:
    return {
        "step": state.step,
        "x": _num(state.robot.x_m),
        "y": _num(state.robot.y_m),
        "yaw": _num(state.robot.yaw_rad),
        "risk": _num(state.failure_risk),
        "objects": [{"id": o.object_id, "x": _num(o.x_m), "y": _num(o.y_m)} for o in state.objects],
    }


@dataclass(slots=True)
class JsonlTelemetrySink:
    """Durable, append-only writer. One instance per run."""

    path: Path
    include_frames: bool = True
    _handle: IO[str] | None = field(default=None, init=False, repr=False)
    _seq: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite telemetry log {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")

    def write(self, record_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._handle is None:
            raise RuntimeError("telemetry sink is closed")
        record = {"schema": SCHEMA, "type": record_type, "seq": self._seq, "wall_utc": utc_now()}
        record.update(payload)
        line = json.dumps(record, allow_nan=False, separators=(",", ":"))
        self._handle.write(line + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._seq += 1
        return record

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> JsonlTelemetrySink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ records

    def run_start(
        self,
        *,
        run_id: str,
        episode_id: str,
        scene: SceneConfig,
        camera_id: str,
        block_duration_s: float,
        goal: Goal2D,
        bundle_id: str,
        latent_dim: int,
        horizon_blocks: int,
        weights: ScoreWeights,
        surprise_threshold: float,
        surprise_metric: str,
        initial_labels: StateLabels,
        history: Sequence[RGBObservation],
        history_actions: Sequence[ActionCommand],
        evidence_scope: str,
        fallback_level: str,
        model_label: str,
        render_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.write(
            "run_start",
            {
                "run_id": run_id,
                "episode_id": episode_id,
                "scene": {
                    "scene_id": scene.scene_id,
                    "arena_width_m": scene.arena_width_m,
                    "arena_height_m": scene.arena_height_m,
                    "frame": "centered",
                },
                "camera": {
                    "camera_id": camera_id,
                    "width_px": history[-1].width_px,
                    "height_px": history[-1].height_px,
                },
                "block_duration_s": block_duration_s,
                "goal": {"x": _num(goal.x_m), "y": _num(goal.y_m), "r": _num(goal.radius_m)},
                "bundle": {
                    "bundle_id": bundle_id,
                    "latent_dim": latent_dim,
                    "horizon_blocks": horizon_blocks,
                    "model_label": model_label,
                },
                "score_weights": {
                    "goal": weights.goal_distance,
                    "risk": weights.failure_risk,
                    "stall": weights.stall,
                    "effort": weights.control_effort,
                    "change": weights.command_change,
                },
                "surprise": {"threshold": _num(surprise_threshold, 8), "metric": surprise_metric},
                "history": [
                    {
                        "sim_time_s": _num(obs.sim_time_s),
                        "png": encode_png(obs) if self.include_frames else None,
                    }
                    for obs in history
                ],
                "history_actions": [_action(a) for a in history_actions],
                "ground_truth": _labels(initial_labels),
                "evidence_scope": evidence_scope,
                "fallback_level": fallback_level,
                "render_hints": dict(render_hints or {}),
            },
        )

    def plan_locked(
        self,
        *,
        block: int,
        plan: PlanResult,
        weights: ScoreWeights,
        planning_latency_s: float,
        current_estimate_xy: tuple[float, float],
    ) -> dict[str, Any]:
        rankings = []
        for item in plan.rankings:
            score = item.score
            rankings.append(
                {
                    "rank": item.rank,
                    "id": item.rollout.candidate_id,
                    "family": item.rollout.family,
                    "total": _num(score.total),
                    "parts": {
                        "goal": _num(weights.goal_distance * score.goal_term_m),
                        "risk": _num(weights.failure_risk * score.maximum_failure_risk),
                        "stall": _num(weights.stall * score.lack_of_progress_m),
                        "effort": _num(weights.control_effort * score.control_effort),
                        "change": _num(weights.command_change * score.command_change),
                    },
                    "max_risk": _num(score.maximum_failure_risk),
                    "path": [[_num(s.robot.x_m), _num(s.robot.y_m)] for s in item.rollout.states],
                }
            )
        selected = plan.selected.rollout
        return self.write(
            "plan_locked",
            {
                "block": block,
                "status": plan.status,
                "locked_at_utc": plan.locked_at_utc,
                "bundle_id": plan.bundle_id,
                "selection_reason": plan.selection_reason,
                "selected_id": selected.candidate_id,
                "selected_family": selected.family,
                "first_action": _action(plan.first_action),
                "selected_actions": [_action(a) for a in selected.actions],
                "selected_states": [_predicted(s) for s in selected.states],
                "predicted_next_latent": _latent(plan.predicted_next_latent),
                "current_estimate": {
                    "x": _num(current_estimate_xy[0]),
                    "y": _num(current_estimate_xy[1]),
                },
                "planning_latency_s": _num(planning_latency_s, 6),
                "rankings": rankings,
            },
        )

    def block_executed(
        self,
        *,
        block: int,
        transition: BlockTransition,
        requested: ActionCommand,
        surprise: SurpriseEvent,
        observed_latent: Sequence[float],
        predicted_next: PredictedState,
    ) -> dict[str, Any]:
        end = transition.end_labels.robot_pose
        error = math.hypot(predicted_next.robot.x_m - end.x_m, predicted_next.robot.y_m - end.y_m)
        events = transition.events
        return self.write(
            "block_executed",
            {
                "block": block,
                "sim_start_s": _num(transition.start_observation.sim_time_s),
                "sim_end_s": _num(transition.end_observation.sim_time_s),
                "requested_action": _action(requested),
                "applied_action": _action(transition.action),
                "observation": {
                    "sim_time_s": _num(transition.end_observation.sim_time_s),
                    "png": encode_png(transition.end_observation) if self.include_frames else None,
                },
                "observed_latent": _latent(observed_latent),
                "surprise": {
                    "status": surprise.status,
                    "discrepancy": None
                    if surprise.discrepancy is None
                    else _num(surprise.discrepancy, 8),
                    "threshold": _num(surprise.threshold, 8),
                    "stop_commanded": surprise.stop_commanded,
                },
                "ground_truth": {
                    "start": _labels(transition.start_labels),
                    "end": _labels(transition.end_labels),
                    "events": {
                        "contacted": list(events.contacted_object_ids),
                        "fell": events.fell,
                        "out_of_bounds": events.out_of_bounds,
                    },
                    "robot_prediction_error_m": _num(error),
                },
            },
        )

    def stop_executed(
        self,
        *,
        block: int,
        reason: str,
        transition: BlockTransition,
    ) -> dict[str, Any]:
        """Record the zero-velocity block issued for an external stop request."""

        end = transition.end_labels
        return self.write(
            "stop_executed",
            {
                "block": block,
                "reason": reason,
                "sim_start_s": _num(transition.start_observation.sim_time_s),
                "sim_end_s": _num(transition.end_observation.sim_time_s),
                "requested_action": _action(transition.requested_action),
                "applied_action": _action(transition.action),
                "observed_velocity": {
                    "forward_mps": _num(end.forward_velocity_mps),
                    "yaw_rate_rps": _num(end.yaw_rate_rps),
                },
                "ground_truth": {"end": _labels(end)},
            },
        )

    def run_end(self, *, reason: str, blocks: int, sim_time_s: float) -> dict[str, Any]:
        return self.write(
            "run_end", {"reason": reason, "blocks": blocks, "sim_time_s": _num(sim_time_s)}
        )


def read_log(path: str | Path) -> list[dict[str, Any]]:
    """Parse a telemetry log, rejecting records from another schema."""

    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != SCHEMA:
            raise ValueError(f"line {number}: unexpected schema {record.get('schema')!r}")
        records.append(record)
    return records
