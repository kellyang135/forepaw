"""Horizon-resolved prediction metrics, action ablation, and G2/G3 gate checks.

Every model number is paired with references computed on the same windows:

* ``encoded_real``: readout applied to the encoder's latent of the true future
  frame (how well the readout can do if prediction were perfect);
* ``persistence``: nothing moves from the true start state;
* ``kinematic``: commanded-velocity unicycle from the true start pose, objects
  stay put;
* ``shuffled``: the same model rolled out with another window's commands.

References that start from privileged true state are evaluation-only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from go2wm.baselines import MotionState, UnicycleModel
from go2wm.contracts import StateLabels
from go2wm.evaluation import action_conditioning_improvement
from go2wm.model import ActionConditionedPredictor, ObservationEncoder

from .readouts import LinearLatentReadout, ReadoutErrors
from .windows import PredictionWindow

DEFAULT_HORIZONS = (1, 2, 4, 6)
METHODS = ("model", "shuffled", "encoded_real", "persistence", "kinematic")


@dataclass(frozen=True, slots=True)
class _StepErrors:
    robot: dict[str, list[float]]
    objects: dict[str, list[float]]
    latent_matched: list[float]
    latent_shuffled: list[float]


def _new_step() -> _StepErrors:
    return _StepErrors({m: [] for m in METHODS}, {m: [] for m in METHODS}, [], [])


def _shuffle_indices(count: int, seed: int) -> np.ndarray:
    """A permutation with no fixed points so every window gets foreign commands."""

    if count < 2:
        raise ValueError("shuffled-action ablation needs at least two windows")
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    return np.roll(order, 1)[np.argsort(order)]


def _robot_obj_error(
    predicted: np.ndarray, labels: StateLabels, readout: LinearLatentReadout
) -> tuple[float, float]:
    truth = readout.spec.encode(labels)
    robot = math.hypot(predicted[0] - truth[0], predicted[1] - truth[1])
    count = len(readout.spec.object_ids)
    if not count:
        return robot, 0.0
    errors = [
        math.hypot(predicted[4 + 2 * i] - truth[4 + 2 * i], predicted[5 + 2 * i] - truth[5 + 2 * i])
        for i in range(count)
    ]
    return robot, float(np.mean(errors))


def _state_vector(
    x: float, y: float, yaw: float, labels: StateLabels, spec_ids: Sequence[str]
) -> np.ndarray:
    values = [x, y, math.sin(yaw), math.cos(yaw)]
    by_id = {item.object_id: item.pose for item in labels.objects}
    for object_id in spec_ids:
        values.extend((by_id[object_id].x_m, by_id[object_id].y_m))
    values.append(0.0)
    return np.asarray(values)


def evaluate_predictions(
    windows: Sequence[PredictionWindow],
    *,
    encoder: ObservationEncoder,
    predictor: ActionConditionedPredictor,
    readout: LinearLatentReadout,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    shuffle_seed: int = 0,
    kinematic_lag_s: float = 0.0,
) -> dict[str, Any]:
    if not windows:
        raise ValueError("no evaluation windows")
    max_h = max(horizons)
    if any(window.horizon < max_h for window in windows):
        raise ValueError("every window must cover the longest requested horizon")
    shuffled = _shuffle_indices(len(windows), shuffle_seed)
    unicycle = UnicycleModel(kinematic_lag_s)
    groups = {
        "all": {h: _new_step() for h in horizons},
        "free": {h: _new_step() for h in horizons},
        "interaction": {h: _new_step() for h in horizons},
    }
    nonfinite = 0
    ids = readout.spec.object_ids

    for index, window in enumerate(windows):
        history = tuple(tuple(encoder.encode(obs)) for obs in window.model_input.observations)
        commands = window.model_input.command_history
        try:
            matched = np.asarray(
                predictor.rollout(history, commands, window.future_actions[:max_h])
            )
            foreign = windows[int(shuffled[index])].future_actions[:max_h]
            other = np.asarray(predictor.rollout(history, commands, foreign))
        except (FloatingPointError, ValueError):
            nonfinite += 1
            continue
        if not (np.isfinite(matched).all() and np.isfinite(other).all()):
            nonfinite += 1
            continue
        real = np.stack(
            [np.asarray(encoder.encode(obs)) for obs in window.future_observations[:max_h]]
        )
        decoded = {
            "model": readout.predict_targets(matched),
            "shuffled": readout.predict_targets(other),
            "encoded_real": readout.predict_targets(real),
        }
        start = window.start_labels.robot_pose
        state = MotionState(start.x_m, start.y_m, start.yaw_rad)
        kinematic_rows = []
        for action in window.future_actions[:max_h]:
            state = unicycle.step(
                state,
                commanded_forward_mps=action.forward_mps,
                commanded_yaw_rate_rps=action.yaw_rate_rps,
                duration_s=action.duration_s,
            )
            kinematic_rows.append(
                _state_vector(state.x_m, state.y_m, state.yaw_rad, window.start_labels, ids)
            )
        decoded["kinematic"] = np.stack(kinematic_rows)
        persistence = _state_vector(start.x_m, start.y_m, start.yaw_rad, window.start_labels, ids)
        decoded["persistence"] = np.tile(persistence, (max_h, 1))

        interaction_so_far = False
        for step in range(1, max_h + 1):
            interaction_so_far = interaction_so_far or window.interaction_types[step - 1] != "free"
            if step not in horizons:
                continue
            labels = window.future_labels[step - 1]
            group_names = ("all", "interaction" if interaction_so_far else "free")
            latent_m = float(np.sqrt(np.mean((matched[step - 1] - real[step - 1]) ** 2)))
            latent_s = float(np.sqrt(np.mean((other[step - 1] - real[step - 1]) ** 2)))
            for name in group_names:
                bucket = groups[name][step]
                for method, values in decoded.items():
                    robot, objects = _robot_obj_error(values[step - 1], labels, readout)
                    bucket.robot[method].append(robot)
                    bucket.objects[method].append(objects)
                bucket.latent_matched.append(latent_m)
                bucket.latent_shuffled.append(latent_s)

    report: dict[str, Any] = {
        "window_count": len(windows),
        "nonfinite_rollouts": nonfinite,
        "horizons_blocks": list(horizons),
        "horizons_s": [h * windows[0].future_actions[0].duration_s for h in horizons],
        "groups": {},
    }
    for name, steps in groups.items():
        report["groups"][name] = {}
        for step, bucket in steps.items():
            count = len(bucket.latent_matched)
            if not count:
                report["groups"][name][str(step)] = {"count": 0}
                continue
            report["groups"][name][str(step)] = {
                "count": count,
                "robot_position_median_m": {m: _median(v) for m, v in bucket.robot.items()},
                "object_position_median_m": {m: _median(v) for m, v in bucket.objects.items()},
                "latent_rmse_median": {
                    "matched": _median(bucket.latent_matched),
                    "shuffled": _median(bucket.latent_shuffled),
                },
            }
    return report


def _median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


@dataclass(frozen=True, slots=True)
class GateCheck:
    gate: str
    criterion: str
    passed: bool
    measured: dict[str, float | int | str]


def g2_checks(
    real_readout: ReadoutErrors,
    constant_mean: ReadoutErrors,
    *,
    leak_ok: bool,
    reload_deterministic: bool,
) -> tuple[GateCheck, ...]:
    return (
        GateCheck(
            "G2",
            "validation real-frame robot readout beats constant-mean baseline",
            real_readout.robot_position_median_m < constant_mean.robot_position_median_m,
            {
                "readout_m": real_readout.robot_position_median_m,
                "constant_mean_m": constant_mean.robot_position_median_m,
            },
        ),
        GateCheck(
            "G2",
            "validation real-frame object readout beats constant-mean baseline",
            real_readout.object_position_median_m < constant_mean.object_position_median_m,
            {
                "readout_m": real_readout.object_position_median_m,
                "constant_mean_m": constant_mean.object_position_median_m,
            },
        ),
        GateCheck("G2", "split leak check passes", leak_ok, {"leak_ok": str(leak_ok)}),
        GateCheck(
            "G2",
            "bundle reload reproduces reference output",
            reload_deterministic,
            {"reload_deterministic": str(reload_deterministic)},
        ),
    )


def g3_checks(
    report: dict[str, Any],
    *,
    min_shuffle_improvement: float = 0.15,
    horizon: int = 6,
) -> tuple[GateCheck, ...]:
    one = report["groups"]["all"].get("1", {})
    far = report["groups"]["all"].get(str(horizon), {})
    checks: list[GateCheck] = []
    if one.get("count"):
        robot = one["robot_position_median_m"]
        objects = one["object_position_median_m"]
        beats = robot["model"] < robot["persistence"] or objects["model"] < objects["persistence"]
        checks.append(
            GateCheck(
                "G3",
                "one-step prediction beats persistence on robot or object position",
                beats,
                {
                    "robot_model_m": robot["model"],
                    "robot_persistence_m": robot["persistence"],
                    "object_model_m": objects["model"],
                    "object_persistence_m": objects["persistence"],
                },
            )
        )
    if far.get("count"):
        matched = far["robot_position_median_m"]["model"]
        shuffled = far["robot_position_median_m"]["shuffled"]
        improvement = (
            action_conditioning_improvement(matched_error=matched, shuffled_error=shuffled)
            if shuffled > 0
            else 0.0
        )
        checks.append(
            GateCheck(
                "G3",
                f"matched commands beat shuffled by >= {min_shuffle_improvement:.0%} "
                f"(robot position, {horizon} blocks)",
                improvement >= min_shuffle_improvement,
                {"matched_m": matched, "shuffled_m": shuffled, "improvement": improvement},
            )
        )
    checks.append(
        GateCheck(
            "G3",
            f"rollouts stay finite through {horizon} blocks",
            report["nonfinite_rollouts"] == 0,
            {"nonfinite_rollouts": report["nonfinite_rollouts"]},
        )
    )
    return tuple(checks)
