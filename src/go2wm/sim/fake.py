"""Deterministic, dependency-free simulator for pipeline verification.

This is not a dynamics benchmark.  It exists to make timing, alignment, reset,
event, persistence, and episode-boundary bugs testable before MuJoCo is wired in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


@dataclass(frozen=True, slots=True)
class FakeSimulatorConfig:
    block_duration_s: float = 0.5
    physics_dt_s: float = 0.01
    camera: CameraConfig = field(
        default_factory=lambda: CameraConfig("overhead", 32, 32)
    )
    scene: SceneConfig = field(
        default_factory=lambda: SceneConfig("fake-arena-v1", 3.0, 3.0)
    )
    robot_radius_m: float = 0.15
    object_radius_m: float = 0.16
    velocity_time_constant_s: float = 0.08
    movable_push_gain: float = 1.25
    residual_forward_velocity_mps: float = 0.04
    residual_yaw_rate_rps: float = 0.03
    settle_decay: float = 0.55
    settle_velocity_tolerance: float = 1e-3
    settle_stable_steps: int = 3
    max_settle_steps: int = 64
    fall_yaw_rate_threshold_rps: float = 2.5
    min_forward_velocity_mps: float = -1.0
    max_forward_velocity_mps: float = 1.0
    max_abs_yaw_rate_rps: float = 3.0

    def __post_init__(self) -> None:
        if self.block_duration_s <= 0 or self.physics_dt_s <= 0:
            raise ValueError("block duration and physics dt must be positive")
        steps = self.block_duration_s / self.physics_dt_s
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("block duration must be an integer number of physics steps")
        if not 0 <= self.settle_decay <= 1:
            raise ValueError("settle_decay must be in [0, 1]")
        if self.settle_stable_steps <= 0 or self.max_settle_steps <= 0:
            raise ValueError("settle step counts must be positive")
        if self.min_forward_velocity_mps >= self.max_forward_velocity_mps:
            raise ValueError("fake forward-velocity bounds are invalid")
        if self.max_abs_yaw_rate_rps <= 0:
            raise ValueError("fake yaw-rate bound must be positive")


class DeterministicFakeSimulator:
    """Small unicycle simulator with simple movable/blocked contact behavior."""

    def __init__(self, config: FakeSimulatorConfig | None = None) -> None:
        self.config = config or FakeSimulatorConfig()
        self._episode_id: str | None = None
        self._scenario_seed = 0
        self._time_s = 0.0
        self._robot_pose = Pose2D(0.0, 0.0, 0.0)
        self._objects: list[ObjectState] = []
        self._forward_velocity_mps = 0.0
        self._yaw_rate_rps = 0.0
        self._fallen = False
        self._frame_index = 0
        self._cached_observation: RGBObservation | None = None
        self._settled = False

    @property
    def block_duration_s(self) -> float:
        return self.config.block_duration_s

    @property
    def physics_dt_s(self) -> float:
        return self.config.physics_dt_s

    @property
    def camera_config(self) -> CameraConfig:
        return self.config.camera

    @property
    def scene_config(self) -> SceneConfig:
        return self.config.scene

    def reset(self, request: ResetRequest) -> ResetReport:
        self._episode_id = request.episode_id
        self._scenario_seed = request.scenario_seed
        self._time_s = 0.0
        self._robot_pose = request.robot_pose
        self._objects = list(request.objects)
        self._forward_velocity_mps = self.config.residual_forward_velocity_mps
        self._yaw_rate_rps = self.config.residual_yaw_rate_rps
        self._fallen = False
        self._frame_index = 0
        self._cached_observation = None

        stable_steps = 0
        settle_steps = 0
        for _ in range(self.config.max_settle_steps):
            settle_steps += 1
            self._forward_velocity_mps *= self.config.settle_decay
            self._yaw_rate_rps *= self.config.settle_decay
            self._integrate_free_motion(self.config.physics_dt_s)
            self._time_s += self.config.physics_dt_s
            if (
                abs(self._forward_velocity_mps) <= self.config.settle_velocity_tolerance
                and abs(self._yaw_rate_rps) <= self.config.settle_velocity_tolerance
            ):
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= self.config.settle_stable_steps:
                break

        self._settled = stable_steps >= self.config.settle_stable_steps
        observation = self.observe()
        labels = self.labels()
        return ResetReport(
            episode_id=request.episode_id,
            scenario_seed=request.scenario_seed,
            settled=self._settled,
            settle_steps=settle_steps,
            settle_duration_s=settle_steps * self.config.physics_dt_s,
            final_observation=observation,
            final_labels=labels,
        )

    def observe(self) -> RGBObservation:
        self._require_reset()
        if self._cached_observation is None:
            self._cached_observation = RGBObservation(
                observation_id=f"{self._episode_id}:frame:{self._frame_index:06d}",
                episode_id=self._episode_id or "",
                sim_time_s=self._time_s,
                camera_id=self.config.camera.camera_id,
                width_px=self.config.camera.width_px,
                height_px=self.config.camera.height_px,
                rgb=self._render_rgb(),
            )
            self._frame_index += 1
        return self._cached_observation

    def labels(self) -> StateLabels:
        self._require_reset()
        return StateLabels(
            sim_time_s=self._time_s,
            robot_pose=self._robot_pose,
            objects=tuple(self._objects),
            forward_velocity_mps=self._forward_velocity_mps,
            yaw_rate_rps=self._yaw_rate_rps,
            fallen=self._fallen,
        )

    def execute_block(self, action: ActionCommand) -> BlockTransition:
        self._require_reset()
        if not self._settled:
            raise RuntimeError("cannot execute an episode from an unsettled reset")
        if not math.isclose(
            action.duration_s, self.config.block_duration_s, abs_tol=1e-9
        ):
            raise ValueError(
                f"action duration must equal {self.config.block_duration_s} seconds"
            )
        requested_action = action
        action = ActionCommand(
            forward_velocity_mps=max(
                self.config.min_forward_velocity_mps,
                min(self.config.max_forward_velocity_mps, action.forward_velocity_mps),
            ),
            yaw_rate_rps=max(
                -self.config.max_abs_yaw_rate_rps,
                min(self.config.max_abs_yaw_rate_rps, action.yaw_rate_rps),
            ),
            duration_s=action.duration_s,
        )
        start_observation = self.observe()
        start_labels = self.labels()
        sample_count = round(action.duration_s / self.config.physics_dt_s)
        samples: list[PhysicsSample] = []
        for _ in range(sample_count):
            samples.append(self._physics_step(action))
        end_observation = self.observe()
        end_labels = self.labels()
        packed_samples = tuple(samples)
        return BlockTransition(
            start_observation=start_observation,
            end_observation=end_observation,
            action=action,
            start_labels=start_labels,
            end_labels=end_labels,
            physics_samples=packed_samples,
            events=BlockEvents.aggregate(packed_samples),
            requested_action=requested_action,
            physics_dt_s=self.config.physics_dt_s,
        )

    def _physics_step(self, action: ActionCommand) -> PhysicsSample:
        dt = self.config.physics_dt_s
        alpha = min(1.0, dt / self.config.velocity_time_constant_s)
        self._forward_velocity_mps += alpha * (
            action.forward_velocity_mps - self._forward_velocity_mps
        )
        self._yaw_rate_rps += alpha * (action.yaw_rate_rps - self._yaw_rate_rps)

        old_pose = self._robot_pose
        new_yaw = old_pose.yaw_rad + self._yaw_rate_rps * dt
        dx = self._forward_velocity_mps * math.cos(new_yaw) * dt
        dy = self._forward_velocity_mps * math.sin(new_yaw) * dt
        candidate_pose = Pose2D(old_pose.x_m + dx, old_pose.y_m + dy, new_yaw)

        contacts: list[ContactSample] = []
        updated_objects: list[ObjectState] = []
        blocked = False
        contact_distance = self.config.robot_radius_m + self.config.object_radius_m
        for item in self._objects:
            distance = math.hypot(
                candidate_pose.x_m - item.pose.x_m,
                candidate_pose.y_m - item.pose.y_m,
            )
            if distance <= contact_distance:
                impulse = abs(self._forward_velocity_mps) * dt
                contacts.append(ContactSample(item.object_id, impulse))
                if item.movable:
                    item = ObjectState(
                        object_id=item.object_id,
                        appearance_class=item.appearance_class,
                        pose=Pose2D(
                            item.pose.x_m + self.config.movable_push_gain * dx,
                            item.pose.y_m + self.config.movable_push_gain * dy,
                            item.pose.yaw_rad,
                        ),
                        movable=item.movable,
                    )
                else:
                    blocked = True
            updated_objects.append(item)

        self._objects = updated_objects
        self._robot_pose = Pose2D(
            old_pose.x_m if blocked else candidate_pose.x_m,
            old_pose.y_m if blocked else candidate_pose.y_m,
            candidate_pose.yaw_rad,
        )
        if abs(action.yaw_rate_rps) >= self.config.fall_yaw_rate_threshold_rps:
            self._fallen = True
        self._time_s += dt
        self._cached_observation = None
        out_of_bounds = (
            abs(self._robot_pose.x_m) > self.config.scene.arena_width_m / 2
            or abs(self._robot_pose.y_m) > self.config.scene.arena_height_m / 2
        )
        return PhysicsSample(
            sim_time_s=self._time_s,
            contacts=tuple(contacts),
            fallen=self._fallen,
            out_of_bounds=out_of_bounds,
        )

    def _integrate_free_motion(self, dt: float) -> None:
        yaw = self._robot_pose.yaw_rad + self._yaw_rate_rps * dt
        self._robot_pose = Pose2D(
            self._robot_pose.x_m + self._forward_velocity_mps * math.cos(yaw) * dt,
            self._robot_pose.y_m + self._forward_velocity_mps * math.sin(yaw) * dt,
            yaw,
        )
        self._cached_observation = None

    def _render_rgb(self) -> bytes:
        width = self.config.camera.width_px
        height = self.config.camera.height_px
        pixels = bytearray((25, 28, 32) * (width * height))

        def to_pixel(pose: Pose2D) -> tuple[int, int]:
            u = round((pose.x_m / self.config.scene.arena_width_m + 0.5) * (width - 1))
            v = round((0.5 - pose.y_m / self.config.scene.arena_height_m) * (height - 1))
            return max(0, min(width - 1, u)), max(0, min(height - 1, v))

        def draw_point(u: int, v: int, color: tuple[int, int, int], radius: int = 1) -> None:
            for py in range(max(0, v - radius), min(height, v + radius + 1)):
                for px in range(max(0, u - radius), min(width, u + radius + 1)):
                    offset = 3 * (py * width + px)
                    pixels[offset : offset + 3] = bytes(color)

        for item in self._objects:
            color = (220, 55, 45) if item.appearance_class.lower() == "red" else (45, 90, 220)
            draw_point(*to_pixel(item.pose), color, radius=1)
        robot_u, robot_v = to_pixel(self._robot_pose)
        draw_point(robot_u, robot_v, (50, 205, 90), radius=1)
        marker = Pose2D(
            self._robot_pose.x_m + 0.12 * math.cos(self._robot_pose.yaw_rad),
            self._robot_pose.y_m + 0.12 * math.sin(self._robot_pose.yaw_rad),
        )
        draw_point(*to_pixel(marker), (255, 220, 40), radius=0)
        return bytes(pixels)

    def _require_reset(self) -> None:
        if self._episode_id is None:
            raise RuntimeError("simulator must be reset before use")
