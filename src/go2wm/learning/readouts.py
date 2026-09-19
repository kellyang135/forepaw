"""Linear probes from latents to task state, fitted on training latents only.

Targets per observation: robot ``[x, y, sin(yaw), cos(yaw)]``, then ``[x, y]``
for each object in a fixed id order, then a fall indicator.  Latents and
targets are standardized with training statistics; the probe is closed-form
ridge regression, so a fit is deterministic and has no optimizer state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from go2wm.contracts import StateLabels
from go2wm.model import Latent, ObjectState, PredictedState, RobotState


@dataclass(frozen=True, slots=True)
class TargetSpec:
    object_ids: tuple[str, ...]

    @property
    def names(self) -> tuple[str, ...]:
        names = ["robot_x", "robot_y", "robot_sin_yaw", "robot_cos_yaw"]
        for object_id in self.object_ids:
            names.extend((f"{object_id}_x", f"{object_id}_y"))
        names.append("fall")
        return tuple(names)

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def fall_index(self) -> int:
        return self.dim - 1

    @classmethod
    def from_labels(cls, labels: StateLabels) -> TargetSpec:
        return cls(tuple(item.object_id for item in labels.objects))

    def encode(self, labels: StateLabels) -> np.ndarray:
        pose = labels.robot_pose
        values = [pose.x_m, pose.y_m, math.sin(pose.yaw_rad), math.cos(pose.yaw_rad)]
        by_id = {item.object_id: item.pose for item in labels.objects}
        missing = [oid for oid in self.object_ids if oid not in by_id]
        if missing:
            raise ValueError(f"labels lack objects {missing}")
        for object_id in self.object_ids:
            values.extend((by_id[object_id].x_m, by_id[object_id].y_m))
        values.append(1.0 if labels.fallen else 0.0)
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, *, floor: float = 1e-6) -> Standardizer:
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("standardizer needs a 2-D array with at least two rows")
        mean = values.mean(axis=0)
        scale = np.maximum(values.std(axis=0), floor)
        return cls(mean, scale)

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def invert(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Standardizer:
        return cls(
            np.asarray(raw["mean"], dtype=np.float64), np.asarray(raw["scale"], dtype=np.float64)
        )


def ridge_fit(features: np.ndarray, targets: np.ndarray, *, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalized bias; returns a ``(d + 1, k)`` matrix."""

    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("features and targets must be aligned 2-D arrays")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not (np.isfinite(features).all() and np.isfinite(targets).all()):
        raise ValueError("ridge inputs must be finite")
    design = np.hstack([features, np.ones((len(features), 1))])
    penalty = alpha * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    return np.linalg.solve(gram, design.T @ targets)


def ridge_predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return features @ weights[:-1] + weights[-1]


@dataclass(frozen=True, slots=True)
class LinearLatentReadout:
    """Implements ``go2wm.model.LatentReadout`` for the planner."""

    spec: TargetSpec
    latent_norm: Standardizer
    target_norm: Standardizer
    weights: np.ndarray
    alpha: float
    version: str = "linear-ridge-v1"

    def predict_targets(self, latents: np.ndarray) -> np.ndarray:
        latents = np.atleast_2d(np.asarray(latents, dtype=np.float64))
        return self.target_norm.invert(ridge_predict(self.weights, self.latent_norm.apply(latents)))

    def decode(self, latent: Latent, *, step: int) -> PredictedState:
        values = self.predict_targets(np.asarray(latent))[0]
        objects = tuple(
            ObjectState(object_id, float(values[4 + 2 * i]), float(values[5 + 2 * i]))
            for i, object_id in enumerate(self.spec.object_ids)
        )
        return PredictedState(
            step=step,
            robot=RobotState(float(values[0]), float(values[1]), math.atan2(values[2], values[3])),
            objects=objects,
            failure_risk=float(min(1.0, max(0.0, values[self.spec.fall_index]))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "object_ids": list(self.spec.object_ids),
            "target_names": list(self.spec.names),
            "latent_norm": self.latent_norm.to_dict(),
            "target_norm": self.target_norm.to_dict(),
            "weights": self.weights.tolist(),
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LinearLatentReadout:
        spec = TargetSpec(tuple(raw["object_ids"]))
        if list(spec.names) != list(raw["target_names"]):
            raise ValueError("readout target layout does not match this code version")
        return cls(
            spec=spec,
            latent_norm=Standardizer.from_dict(raw["latent_norm"]),
            target_norm=Standardizer.from_dict(raw["target_norm"]),
            weights=np.asarray(raw["weights"], dtype=np.float64),
            alpha=float(raw["alpha"]),
            version=raw["version"],
        )


def fit_linear_readout(
    latents: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[StateLabels],
    *,
    alpha: float = 1.0,
    spec: TargetSpec | None = None,
) -> LinearLatentReadout:
    if len(latents) != len(labels) or not labels:
        raise ValueError("readout fit needs one label per latent")
    spec = spec or TargetSpec.from_labels(labels[0])
    x = np.asarray(latents, dtype=np.float64)
    y = np.stack([spec.encode(item) for item in labels])
    latent_norm = Standardizer.fit(x)
    target_norm = Standardizer.fit(y)
    weights = ridge_fit(latent_norm.apply(x), target_norm.apply(y), alpha=alpha)
    return LinearLatentReadout(spec, latent_norm, target_norm, weights, alpha)


@dataclass(frozen=True, slots=True)
class ReadoutErrors:
    count: int
    robot_position_median_m: float
    robot_position_mean_m: float
    robot_yaw_median_rad: float
    object_position_median_m: float
    object_position_mean_m: float


def readout_errors(
    readout: LinearLatentReadout,
    latents: np.ndarray,
    labels: Sequence[StateLabels],
) -> ReadoutErrors:
    predicted = readout.predict_targets(latents)
    truth = np.stack([readout.spec.encode(item) for item in labels])
    return _errors_from_targets(predicted, truth, len(readout.spec.object_ids))


def constant_mean_errors(
    train_labels: Sequence[StateLabels],
    eval_labels: Sequence[StateLabels],
    spec: TargetSpec,
) -> ReadoutErrors:
    """The G2 reference: predict the training mean for every evaluation frame."""

    mean = np.stack([spec.encode(item) for item in train_labels]).mean(axis=0)
    truth = np.stack([spec.encode(item) for item in eval_labels])
    return _errors_from_targets(np.tile(mean, (len(truth), 1)), truth, len(spec.object_ids))


def _errors_from_targets(
    predicted: np.ndarray, truth: np.ndarray, object_count: int
) -> ReadoutErrors:
    robot = np.hypot(predicted[:, 0] - truth[:, 0], predicted[:, 1] - truth[:, 1])
    yaw_pred = np.arctan2(predicted[:, 2], predicted[:, 3])
    yaw_true = np.arctan2(truth[:, 2], truth[:, 3])
    yaw = np.abs((yaw_pred - yaw_true + np.pi) % (2 * np.pi) - np.pi)
    if object_count:
        dx = predicted[:, 4 : 4 + 2 * object_count : 2] - truth[:, 4 : 4 + 2 * object_count : 2]
        dy = predicted[:, 5 : 5 + 2 * object_count : 2] - truth[:, 5 : 5 + 2 * object_count : 2]
        objects = np.hypot(dx, dy).ravel()
    else:
        objects = np.zeros(1)
    return ReadoutErrors(
        count=len(truth),
        robot_position_median_m=float(np.median(robot)),
        robot_position_mean_m=float(robot.mean()),
        robot_yaw_median_rad=float(np.median(yaw)),
        object_position_median_m=float(np.median(objects)),
        object_position_mean_m=float(objects.mean()),
    )
