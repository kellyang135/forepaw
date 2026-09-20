"""MuJoCo Go2 ``SimulatorAdapter`` with a pluggable 50 Hz gait (DRAFT for SIM review).

One 0.5 s block = 25 policy updates x 4 physics steps of 5 ms. The requested
command is clipped once to the configured bounds; lateral velocity is always 0.
Every physics step yields a ``PhysicsSample`` with robot-box contacts (normal
impulse), fall, and out-of-bounds flags. Simulated time is counted in integer
physics steps so block timestamps are exact.

Status: PROPOSED. Nothing here is G0 evidence until SIM runs the trials and
records an artifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from go2wm.contracts import (
    ActionCommand,
    BlockEvents,
    BlockTransition,
    CameraConfig,
    ContactSample,
    ObjectState,
    PhysicsSample,
    Pose2D,
    ResetReport,
    ResetRequest,
    RGBObservation,
    SceneConfig,
    StateLabels,
)

from .locomotion import LocomotionController, LocomotionSpec, Policy, quat_to_rotation
from .task_scene import PARKED_XY, TaskSceneConfig, build_task_scene


@dataclass(frozen=True, slots=True)
class MujocoGo2Config:
    block_duration_s: float = 0.5
    # PROPOSED bounds from configs/experiment.toml; P-001 replaces them after G0.
    min_forward_velocity_mps: float = 0.0
    max_forward_velocity_mps: float = 0.6
    max_abs_yaw_rate_rps: float = 1.2
    settle_min_s: float = 1.0
    settle_max_s: float = 5.0
    settle_window_s: float = 0.5
    settle_speed_tolerance_mps: float = 0.03
    settle_yaw_rate_tolerance_rps: float = 0.05
    # Reset-only heading hold (simulator bookkeeping, never a runtime input).
    settle_heading_gain: float = 2.0
    settle_heading_max_rate_rps: float = 0.5
    settle_heading_tolerance_rad: float = 0.05
    fall_min_base_height_m: float = 0.15
    fall_max_tilt_deg: float = 60.0
    min_robot_box_clearance_m: float = 0.60
    # Seeded by ResetRequest.scenario_seed so repeats differ but stay reproducible.
    reset_joint_noise_rad: float = 0.05
    scene: TaskSceneConfig = field(default_factory=TaskSceneConfig)

    def __post_init__(self) -> None:
        if self.min_forward_velocity_mps > self.max_forward_velocity_mps:
            raise ValueError("forward-velocity bounds are inverted")
        if self.max_abs_yaw_rate_rps <= 0:
            raise ValueError("yaw-rate bound must be positive")


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def quat_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


class MujocoGo2Simulator:
    """Implements ``go2wm.sim.SimulatorAdapter`` for the Go2 push/detour scene."""

    def __init__(
        self,
        controller_spec: LocomotionSpec,
        policy: Policy,
        robot_xml: Path,
        config: MujocoGo2Config | None = None,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.config = config or MujocoGo2Config()
        self.controller_spec = controller_spec
        self.built = build_task_scene(controller_spec, robot_xml, self.config.scene)
        self.model = self.built.model
        self.data = mujoco.MjData(self.model)
        self.controller = LocomotionController(controller_spec, policy, self.model, self.data)
        self._dt = controller_spec.physics_dt_s
        steps = self.config.block_duration_s / self._dt
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("block duration must be a whole number of physics steps")
        policy_steps = self.config.block_duration_s / controller_spec.control_dt_s
        if not math.isclose(policy_steps, round(policy_steps), abs_tol=1e-9):
            raise ValueError("block duration must be a whole number of policy periods")
        self._policy_steps_per_block = round(policy_steps)
        self._robot_geoms = frozenset(
            g
            for g in range(self.model.ngeom)
            if self.model.body_rootid[self.model.geom_bodyid[g]] == self.built.robot_root_body
        )
        self._box_geom_to_slot = {g: i for i, g in enumerate(self.built.box_geom_ids)}
        base_mass = self.model.body_mass[list(self.built.box_body_ids)].copy()
        self._box_unit_inertia = (
            self.model.body_inertia[list(self.built.box_body_ids)] / base_mass[:, None]
        )
        self._renderer: Any = None
        self._episode_id: str | None = None
        self._step_index = 0
        self._frame_index = 0
        self._cached_observation: RGBObservation | None = None
        self._slot_objects: list[ObjectState | None] = [None] * len(self.built.box_body_ids)
        self._object_order: list[int] = []
        self._fallen = False
        self._settled = False
        self._camera = CameraConfig(
            self.config.scene.camera_id,
            self.config.scene.image_width_px,
            self.config.scene.image_height_px,
            vertical_fov_deg=self.config.scene.camera_fovy_deg,
        )
        self._scene = SceneConfig(
            self.built.scene_id, self.config.scene.arena_width_m, self.config.scene.arena_height_m
        )

    # ------------------------------------------------------------------ protocol
    @property
    def block_duration_s(self) -> float:
        return self.config.block_duration_s

    @property
    def physics_dt_s(self) -> float:
        return self._dt

    @property
    def camera_config(self) -> CameraConfig:
        return self._camera

    @property
    def scene_config(self) -> SceneConfig:
        return self._scene

    @property
    def controller_id(self) -> str:
        return self.controller_spec.controller_id

    def reset(self, request: ResetRequest) -> ResetReport:
        mj = self._mj
        self._validate_request(request)
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [
            request.robot_pose.x_m,
            request.robot_pose.y_m,
            self.controller_spec.init_base_height_m,
        ]
        self.data.qpos[3:7] = quat_from_yaw(request.robot_pose.yaw_rad)
        rng = np.random.default_rng(request.scenario_seed)
        noise = self.config.reset_joint_noise_rad
        self.data.qpos[self.controller.qadr] = self.controller.default + rng.uniform(
            -noise, noise, 12
        )
        self._place_objects(request.objects)
        mj.mj_forward(self.model, self.data)
        self.controller.reset()
        self._episode_id = request.episode_id
        self._step_index = 0
        self._frame_index = 0
        self._cached_observation = None
        self._fallen = False

        window = round(self.config.settle_window_s / self.controller_spec.control_dt_s)
        min_steps = round(self.config.settle_min_s / self.controller_spec.control_dt_s)
        max_steps = round(self.config.settle_max_s / self.controller_spec.control_dt_s)
        calm = 0
        self._settled = False
        target_yaw = request.robot_pose.yaw_rad
        for policy_step in range(1, max_steps + 1):
            heading_error = _wrap(target_yaw - yaw_from_quat(self.data.qpos[3:7]))
            hold = max(
                -self.config.settle_heading_max_rate_rps,
                min(
                    self.config.settle_heading_max_rate_rps,
                    self.config.settle_heading_gain * heading_error,
                ),
            )
            self.controller.policy_step(0.0, hold)
            for _ in range(self.controller_spec.substeps):
                self._physics_step()
            speed = float(np.linalg.norm(self.data.qvel[0:2]))
            yaw_rate = abs(self._world_yaw_rate())
            quiet = (
                speed <= self.config.settle_speed_tolerance_mps
                and yaw_rate <= self.config.settle_yaw_rate_tolerance_rps
                and abs(heading_error) <= self.config.settle_heading_tolerance_rad
            )
            calm = calm + 1 if quiet else 0
            if self._fallen:
                break
            if policy_step >= min_steps and calm >= window:
                self._settled = True
                break
        observation = self.observe()
        labels = self.labels()
        return ResetReport(
            episode_id=request.episode_id,
            scenario_seed=request.scenario_seed,
            settled=self._settled,
            settle_steps=self._step_index,
            settle_duration_s=self._step_index * self._dt,
            final_observation=observation,
            final_labels=labels,
        )

    def observe(self) -> RGBObservation:
        self._require_reset()
        if self._cached_observation is None:
            if self._renderer is None:
                self._renderer = self._mj.Renderer(
                    self.model, height=self._camera.height_px, width=self._camera.width_px
                )
            self._renderer.update_scene(self.data, camera=self._camera.camera_id)
            pixels = self._renderer.render()
            self._cached_observation = RGBObservation(
                observation_id=f"{self._episode_id}:frame:{self._frame_index:06d}",
                episode_id=self._episode_id or "",
                sim_time_s=self._sim_time(),
                camera_id=self._camera.camera_id,
                width_px=self._camera.width_px,
                height_px=self._camera.height_px,
                rgb=np.ascontiguousarray(pixels, dtype=np.uint8).tobytes(),
            )
            self._frame_index += 1
        return self._cached_observation

    def labels(self) -> StateLabels:
        self._require_reset()
        rotation = quat_to_rotation(self.data.qpos[3:7])
        body_velocity = rotation.T @ np.asarray(self.data.qvel[0:3])
        objects = tuple(self._object_state(slot) for slot in self._object_order)
        return StateLabels(
            sim_time_s=self._sim_time(),
            robot_pose=Pose2D(
                float(self.data.qpos[0]),
                float(self.data.qpos[1]),
                yaw_from_quat(self.data.qpos[3:7]),
            ),
            objects=objects,
            forward_velocity_mps=float(body_velocity[0]),
            yaw_rate_rps=self._world_yaw_rate(),
            fallen=self._fallen,
        )

    def execute_block(self, action: ActionCommand) -> BlockTransition:
        self._require_reset()
        if not self._settled:
            raise RuntimeError("cannot execute an episode from an unsettled reset")
        if not math.isclose(action.duration_s, self.config.block_duration_s, abs_tol=1e-9):
            raise ValueError(f"action duration must equal {self.config.block_duration_s} seconds")
        requested = action
        applied = ActionCommand(
            forward_velocity_mps=min(
                self.config.max_forward_velocity_mps,
                max(self.config.min_forward_velocity_mps, action.forward_velocity_mps),
            ),
            yaw_rate_rps=min(
                self.config.max_abs_yaw_rate_rps,
                max(-self.config.max_abs_yaw_rate_rps, action.yaw_rate_rps),
            ),
            duration_s=action.duration_s,
        )
        start_observation = self.observe()
        start_labels = self.labels()
        samples: list[PhysicsSample] = []
        for _ in range(self._policy_steps_per_block):
            self.controller.policy_step(applied.forward_velocity_mps, applied.yaw_rate_rps)
            for _ in range(self.controller_spec.substeps):
                samples.append(self._physics_step())
        packed = tuple(samples)
        return BlockTransition(
            start_observation=start_observation,
            end_observation=self.observe(),
            action=applied,
            start_labels=start_labels,
            end_labels=self.labels(),
            physics_samples=packed,
            events=BlockEvents.aggregate(packed),
            requested_action=requested,
            physics_dt_s=self._dt,
        )

    def set_object_mass(self, object_id: str, mass_kg: float) -> None:
        """Change one box's mass mid-episode without telling any model.

        For surprise rehearsals and evaluation perturbations only; the box keeps
        its colour, so the change is invisible in the image.
        """

        if mass_kg <= 0:
            raise ValueError("mass must be positive")
        for slot, item in enumerate(self._slot_objects):
            if item is not None and item.object_id == object_id:
                body = self.built.box_body_ids[slot]
                self.model.body_mass[body] = mass_kg
                self.model.body_inertia[body] = self._box_unit_inertia[slot] * mass_kg
                return
        raise KeyError(object_id)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ internals
    def _sim_time(self) -> float:
        return self._step_index * self._dt

    def _world_yaw_rate(self) -> float:
        rotation = quat_to_rotation(self.data.qpos[3:7])
        return float((rotation @ np.asarray(self.data.qvel[3:6]))[2])

    def _physics_step(self) -> PhysicsSample:
        mj = self._mj
        self.controller.apply_substep()
        mj.mj_step(self.model, self.data)
        self._step_index += 1
        self._cached_observation = None
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            raise RuntimeError(f"non-finite physics state at step {self._step_index}")
        impulses: dict[str, float] = {}
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = (int(contact.geom1), int(contact.geom2))
            for box_geom, other in (pair, pair[::-1]):
                slot = self._box_geom_to_slot.get(box_geom)
                if slot is None or other not in self._robot_geoms:
                    continue
                state = self._slot_objects[slot]
                if state is None:
                    continue
                mj.mj_contactForce(self.model, self.data, index, force)
                impulses[state.object_id] = (
                    impulses.get(state.object_id, 0.0) + max(0.0, float(force[0])) * self._dt
                )
        rotation = quat_to_rotation(self.data.qpos[3:7])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))
        if (
            float(self.data.qpos[2]) < self.config.fall_min_base_height_m
            or tilt > self.config.fall_max_tilt_deg
        ):
            self._fallen = True
        half_w = self.config.scene.arena_width_m / 2
        half_h = self.config.scene.arena_height_m / 2
        out = abs(float(self.data.qpos[0])) > half_w or abs(float(self.data.qpos[1])) > half_h
        return PhysicsSample(
            sim_time_s=self._sim_time(),
            contacts=tuple(ContactSample(k, v) for k, v in sorted(impulses.items())),
            fallen=self._fallen,
            out_of_bounds=out,
        )

    def _validate_request(self, request: ResetRequest) -> None:
        scene = self.config.scene
        if len(request.objects) > len(self.built.box_body_ids):
            raise ValueError(f"scene has {len(self.built.box_body_ids)} box slots")
        for item in request.objects:
            box_class = scene.box_classes.get(item.appearance_class)
            if box_class is None:
                raise ValueError(f"unknown appearance class {item.appearance_class!r}")
            if box_class.movable != item.movable:
                raise ValueError(
                    f"{item.object_id}: class {item.appearance_class!r} is "
                    f"{'movable' if box_class.movable else 'resistant'} in this scene"
                )
            distance = math.hypot(
                item.pose.x_m - request.robot_pose.x_m, item.pose.y_m - request.robot_pose.y_m
            )
            if distance < self.config.min_robot_box_clearance_m:
                raise ValueError(f"{item.object_id} starts {distance:.2f} m from the robot")
        poses = [item.pose for item in request.objects]
        min_gap = 2 * math.sqrt(2) * scene.box_half_extent_m + 0.02
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                if math.hypot(poses[i].x_m - poses[j].x_m, poses[i].y_m - poses[j].y_m) < min_gap:
                    raise ValueError("boxes overlap at reset")

    def _place_objects(self, objects: tuple[ObjectState, ...]) -> None:
        scene = self.config.scene
        half = scene.box_half_extent_m
        self._slot_objects = [None] * len(self.built.box_body_ids)
        self._object_order = []
        for slot, (body, geom, qadr) in enumerate(
            zip(
                self.built.box_body_ids,
                self.built.box_geom_ids,
                self.built.box_joint_qposadr,
                strict=True,
            )
        ):
            if slot < len(objects):
                item = objects[slot]
                box_class = scene.box_classes[item.appearance_class]
                self.model.body_mass[body] = box_class.mass_kg
                self.model.body_inertia[body] = self._box_unit_inertia[slot] * box_class.mass_kg
                self.model.geom_rgba[geom] = box_class.rgba
                self.data.qpos[qadr : qadr + 3] = [item.pose.x_m, item.pose.y_m, half]
                self.data.qpos[qadr + 3 : qadr + 7] = quat_from_yaw(item.pose.yaw_rad)
                self._slot_objects[slot] = item
                self._object_order.append(slot)
            else:
                self.data.qpos[qadr : qadr + 3] = [PARKED_XY[0] + 2 * slot, PARKED_XY[1], half]
                self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _object_state(self, slot: int) -> ObjectState:
        item = self._slot_objects[slot]
        assert item is not None
        qadr = self.built.box_joint_qposadr[slot]
        return ObjectState(
            object_id=item.object_id,
            appearance_class=item.appearance_class,
            pose=Pose2D(
                float(self.data.qpos[qadr]),
                float(self.data.qpos[qadr + 1]),
                yaw_from_quat(self.data.qpos[qadr + 3 : qadr + 7]),
            ),
            movable=item.movable,
        )

    def _require_reset(self) -> None:
        if self._episode_id is None:
            raise RuntimeError("simulator must be reset before use")
