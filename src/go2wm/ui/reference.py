"""Privileged kinematic reference backend for UI rehearsal on the fake simulator.

NOT A LEARNED MODEL. The encoder reads simulator poses directly, which D-004
forbids for any deployed planner. It exists only so the closed loop, the
telemetry log and the viewer can be exercised with visible motion, pushes and
a surprise stop before a trained bundle exists. Every log it produces is
labeled accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from go2wm.baselines.unicycle import MotionState, UnicycleModel
from go2wm.model import (
    BundleManifest,
    ComponentWorldModelBackend,
    ModelBundle,
    ObjectState,
    PredictedState,
    RobotState,
    RuntimeRequirements,
)
from go2wm.runtime import SurpriseCalibration

OBJECTS = ("light", "resistant")
LATENT_DIM = 4 + 2 * len(OBJECTS)
CONTACT_M = 0.31
MOVABLE_APPEARANCE = {"blue": True, "red": False}
LABEL = (
    "privileged kinematic reference (reads simulator pose; NOT a learned model, UI rehearsal only)"
)


class SimulatorPoseEncoder:
    def __init__(self, simulator) -> None:
        self.simulator = simulator

    def encode(self, observation) -> tuple[float, ...]:
        del observation
        labels = self.simulator.labels()
        pose = labels.robot_pose
        latent = [pose.x_m, pose.y_m, math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)]
        by_id = {item.object_id: item.pose for item in labels.objects}
        for object_id in OBJECTS:
            latent.extend((by_id[object_id].x_m, by_id[object_id].y_m))
        return tuple(latent)


@dataclass(frozen=True, slots=True)
class AppearancePushPredictor:
    """Unicycle with lag; boxes move if their appearance is believed movable."""

    appearance: tuple[str, ...] = ("blue", "red")
    model: UnicycleModel = field(default_factory=lambda: UnicycleModel(lag_time_constant_s=0.08))

    def rollout(self, history_latents, history_actions, candidate_actions):
        latent = list(history_latents[-1])
        previous = history_actions[-1] if history_actions else None
        state = MotionState(
            latent[0],
            latent[1],
            math.atan2(latent[3], latent[2]),
            previous.forward_mps if previous else 0.0,
            previous.yaw_rate_rps if previous else 0.0,
        )
        boxes = [[latent[4 + 2 * i], latent[5 + 2 * i]] for i in range(len(OBJECTS))]
        out = []
        for action in candidate_actions:
            state = self.model.step(
                state,
                commanded_forward_mps=action.forward_mps,
                commanded_yaw_rate_rps=action.yaw_rate_rps,
                duration_s=action.duration_s,
            )
            x, y = state.x_m, state.y_m
            blocked = 0.0
            for i, box in enumerate(boxes):
                dx, dy = box[0] - x, box[1] - y
                distance = math.hypot(dx, dy) or 1e-9
                if distance >= CONTACT_M:
                    continue
                overlap = CONTACT_M - distance
                if MOVABLE_APPEARANCE.get(self.appearance[i], False):
                    box[0] += dx / distance * overlap
                    box[1] += dy / distance * overlap
                else:
                    x -= dx / distance * overlap
                    y -= dy / distance * overlap
                    blocked = 1.0
            state = MotionState(
                x, y, state.yaw_rad, state.forward_velocity_mps * (1 - blocked), state.yaw_rate_rps
            )
            flat = [x, y, math.cos(state.yaw_rad), math.sin(state.yaw_rad)]
            for box in boxes:
                flat.extend(box)
            flat.append(blocked)
            out.append(tuple(flat))
        return out


class _RiskStrippingPredictor:
    """Carries the blocked flag to the readout, then drops it from the latent."""

    def __init__(self, inner: AppearancePushPredictor) -> None:
        self.inner = inner
        self.risk: dict[tuple[float, ...], float] = {}

    def rollout(self, history_latents, history_actions, candidate_actions):
        out = []
        for flat in self.inner.rollout(history_latents, history_actions, candidate_actions):
            latent = tuple(flat[:LATENT_DIM])
            self.risk[latent] = 0.75 if flat[LATENT_DIM] else 0.04
            out.append(latent)
        return out


class ReferenceReadout:
    def __init__(self, predictor: _RiskStrippingPredictor) -> None:
        self.predictor = predictor

    def decode(self, latent, *, step: int) -> PredictedState:
        latent = tuple(latent)
        objects = tuple(
            ObjectState(object_id, latent[4 + 2 * i], latent[5 + 2 * i])
            for i, object_id in enumerate(OBJECTS)
        )
        return PredictedState(
            step=step,
            robot=RobotState(latent[0], latent[1], math.atan2(latent[3], latent[2])),
            objects=objects,
            failure_risk=self.predictor.risk.get(latent, 0.04),
        )


def reference_bundle(simulator, *, threshold: float = 0.05):
    """Return (ModelBundle, encoder, readout, SurpriseCalibration) for the fake scene."""

    manifest = BundleManifest(
        bundle_id="reference-kinematic-v1",
        encoder_version="sim-pose-privileged",
        predictor_version="unicycle-lag-appearance-push",
        readout_version="identity",
        normalization_version="none",
        surprise_calibration_version="hand-set-v1",
        latent_dim=LATENT_DIM,
    )
    predictor = _RiskStrippingPredictor(AppearancePushPredictor())
    readout = ReferenceReadout(predictor)
    encoder = SimulatorPoseEncoder(simulator)
    backend = ComponentWorldModelBackend(manifest, encoder, predictor, readout)
    bundle = ModelBundle(backend, RuntimeRequirements(latent_dim=LATENT_DIM))
    calibration = SurpriseCalibration(
        "hand-set-v1", manifest.bundle_id, LATENT_DIM, "rmse", threshold
    )
    return bundle, encoder, readout, calibration
