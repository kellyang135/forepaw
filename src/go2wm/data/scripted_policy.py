"""Scenario sampling and a boundary-aware scripted command policy for data collection.

Collection only. The policy reads privileged simulator labels (robot and box poses)
to aim pushes, plan detours and stay inside the arena; D-004 allows this while
collecting and forbids it at runtime. Nothing here is imported by planning code.

Episode modes and why they exist:

- ``push``: approach and shove the movable (blue) box, then wander.
- ``resist``: drive into the resistant (red) box, stall against it, turn away.
- ``detour``: reach a goal behind the red box by arcing past one side of it.
- ``free``: smooth piecewise-constant commands, including stops and turns in place.

Every mode adds command noise and occasional random segments so the dataset
covers the full action range instead of a few scripted trajectories.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from go2wm.contracts import ActionCommand, Goal2D, ObjectState, Pose2D, ResetRequest, StateLabels

MODES = ("push", "resist", "detour", "free")
DEFAULT_MODE_WEIGHTS = {"push": 0.25, "resist": 0.15, "detour": 0.25, "free": 0.35}


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True, slots=True)
class CommandBounds:
    min_forward_mps: float = 0.0
    max_forward_mps: float = 0.6
    max_abs_yaw_rps: float = 1.2
    block_duration_s: float = 0.5

    def clip(self, forward: float, yaw: float) -> ActionCommand:
        forward = min(self.max_forward_mps, max(self.min_forward_mps, forward))
        yaw = min(self.max_abs_yaw_rps, max(-self.max_abs_yaw_rps, yaw))
        return ActionCommand(round(forward, 4), round(yaw, 4), self.block_duration_s)


@dataclass(frozen=True, slots=True)
class ArenaGeometry:
    """Centered arena frame, matching the fake and MuJoCo simulators."""

    width_m: float
    height_m: float
    box_half_extent_m: float = 0.20
    robot_box_clearance_m: float = 0.60
    spawn_margin_m: float = 0.45
    wall_margin_m: float = 0.45

    def inside(self, x: float, y: float, margin: float) -> bool:
        return abs(x) <= self.width_m / 2 - margin and abs(y) <= self.height_m / 2 - margin


@dataclass(frozen=True, slots=True)
class Scenario:
    reset: ResetRequest
    goal: Goal2D
    mode: str


class ScenarioSampler:
    """Deterministic scenario per seed: robot, one blue and one red box, and a goal."""

    def __init__(
        self,
        arena: ArenaGeometry,
        *,
        movable_class: str = "blue",
        resistant_class: str = "red",
        mode_weights: dict[str, float] | None = None,
        goal_radius_m: float = 0.25,
    ) -> None:
        self.arena = arena
        self.movable_class = movable_class
        self.resistant_class = resistant_class
        weights = mode_weights or DEFAULT_MODE_WEIGHTS
        unknown = set(weights) - set(MODES)
        if unknown or not any(weights.values()):
            raise ValueError(f"invalid mode weights {weights}")
        self.mode_weights = weights
        self.goal_radius_m = goal_radius_m

    def sample(self, seed: int, *, episode_prefix: str = "ep") -> Scenario:
        rng = random.Random(seed * 7919 + 17)
        mode = rng.choices(list(self.mode_weights), weights=list(self.mode_weights.values()))[0]
        for _ in range(500):
            scenario = self._try(rng, seed, mode, episode_prefix)
            if scenario is not None:
                return scenario
        raise RuntimeError(f"could not place a {mode} scenario for seed {seed}")

    def _point(self, rng: random.Random, margin: float) -> tuple[float, float]:
        hw, hh = self.arena.width_m / 2 - margin, self.arena.height_m / 2 - margin
        return rng.uniform(-hw, hw), rng.uniform(-hh, hh)

    def _try(self, rng: random.Random, seed: int, mode: str, prefix: str) -> Scenario | None:
        arena = self.arena
        margin = arena.spawn_margin_m
        blue = self._point(rng, margin + arena.box_half_extent_m)
        red = self._point(rng, margin + arena.box_half_extent_m)
        target = blue if mode == "push" else red if mode in ("resist", "detour") else None
        if target is not None:
            bearing = rng.uniform(-math.pi, math.pi)
            distance = rng.uniform(0.9, 1.4)
            rx, ry = (
                target[0] - distance * math.cos(bearing),
                target[1] - distance * math.sin(bearing),
            )
            yaw = wrap(bearing + rng.uniform(-0.35, 0.35))
        else:
            rx, ry = self._point(rng, margin)
            yaw = rng.uniform(-math.pi, math.pi)
        if not arena.inside(rx, ry, margin):
            return None
        min_box_gap = 2 * math.sqrt(2) * arena.box_half_extent_m + 0.15
        if math.dist(blue, red) < min_box_gap:
            return None
        if (
            min(math.dist((rx, ry), blue), math.dist((rx, ry), red))
            < arena.robot_box_clearance_m + 0.05
        ):
            return None
        if mode == "detour":
            # Goal sits beyond the red box, roughly on the robot-to-box line.
            dx, dy = red[0] - rx, red[1] - ry
            norm = math.hypot(dx, dy)
            reach = rng.uniform(0.7, 1.1)
            goal_xy = (red[0] + dx / norm * reach, red[1] + dy / norm * reach)
        else:
            goal_xy = self._point(rng, margin)
        if (
            not arena.inside(*goal_xy, margin)
            or min(math.dist(goal_xy, blue), math.dist(goal_xy, red)) < 0.5
        ):
            return None
        reset = ResetRequest(
            episode_id=f"{prefix}-{seed:05d}",
            scenario_seed=seed,
            robot_pose=Pose2D(rx, ry, yaw),
            objects=(
                ObjectState("light", self.movable_class, Pose2D(*blue), True),
                ObjectState("resistant", self.resistant_class, Pose2D(*red), False),
            ),
        )
        return Scenario(reset, Goal2D(goal_xy[0], goal_xy[1], self.goal_radius_m), mode)


@dataclass(slots=True)
class ScriptedPolicy:
    """Callable ``policy(block_index, labels) -> ActionCommand`` for ``collect_with_policy``."""

    scenario: Scenario
    arena: ArenaGeometry
    bounds: CommandBounds = field(default_factory=CommandBounds)
    seed: int = 0
    noise_forward_mps: float = 0.05
    noise_yaw_rps: float = 0.12
    random_segment_probability: float = 0.12
    _rng: random.Random = field(init=False)
    _phase: str = field(init=False)
    _phase_blocks: int = field(init=False, default=0)
    _held: ActionCommand | None = field(init=False, default=None)
    _hold_left: int = field(init=False, default=0)
    _detour_side: float = field(init=False, default=1.0)
    _start_box: tuple[float, float] | None = field(init=False, default=None)
    _speed: float = field(init=False, default=0.4)
    _turn_dir: float = field(init=False, default=1.0)
    _stall_blocks: int = field(init=False, default=3)
    _push_distance: float = field(init=False, default=0.7)
    log: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed * 104729 + 3)
        self._phase = {
            "push": "approach",
            "resist": "approach",
            "detour": "detour",
            "free": "free",
        }[self.scenario.mode]
        self._detour_side = self._rng.choice((-1.0, 1.0))
        self._stall_blocks = self._rng.randint(2, 4)
        self._push_distance = self._rng.uniform(0.5, 1.0)

    # ------------------------------------------------------------------ main entry

    def __call__(self, block: int, labels: StateLabels) -> ActionCommand:
        pose = labels.robot_pose
        guard = self._wall_guard(pose)
        if guard is not None:
            self.log.append("wall_guard")
            self._held, self._hold_left = None, 0
            return guard
        if self._phase != "free" and self._rng.random() < self.random_segment_probability:
            self.log.append("random")
            return self._random_command()
        handler = getattr(self, f"_phase_{self._phase}")
        command = handler(block, labels)
        self._phase_blocks += 1
        self.log.append(self._phase)
        return command

    # ------------------------------------------------------------------ safety

    def _wall_guard(self, pose: Pose2D) -> ActionCommand | None:
        """Turn toward the arena center when the next second of motion would reach the wall."""

        lookahead = 0.55
        nx = pose.x_m + lookahead * math.cos(pose.yaw_rad)
        ny = pose.y_m + lookahead * math.sin(pose.yaw_rad)
        if self.arena.inside(nx, ny, self.arena.wall_margin_m):
            return None
        to_center = math.atan2(-pose.y_m, -pose.x_m)
        error = wrap(to_center - pose.yaw_rad)
        near = not self.arena.inside(pose.x_m, pose.y_m, self.arena.wall_margin_m * 0.6)
        forward = 0.0 if near or abs(error) > 1.2 else 0.15
        return self.bounds.clip(forward, math.copysign(0.9, error))

    # ------------------------------------------------------------------ phases

    def _steer(
        self, pose: Pose2D, tx: float, ty: float, speed: float, gain: float = 1.6
    ) -> ActionCommand:
        error = wrap(math.atan2(ty - pose.y_m, tx - pose.x_m) - pose.yaw_rad)
        forward = speed * max(0.0, math.cos(error)) if abs(error) < 1.3 else 0.05
        return self._noisy(forward, gain * error)

    def _phase_approach(self, block: int, labels: StateLabels) -> ActionCommand:
        box_id = "light" if self.scenario.mode == "push" else "resistant"
        box = labels.object(box_id).pose
        if self._start_box is None:
            self._start_box = (box.x_m, box.y_m)
            self._speed = self._rng.uniform(0.3, 0.55)
        moved = math.dist(self._start_box, (box.x_m, box.y_m))
        contact_range = math.dist(
            (labels.robot_pose.x_m, labels.robot_pose.y_m), (box.x_m, box.y_m)
        )
        if self.scenario.mode == "push" and (
            moved > self._push_distance or self._phase_blocks > 16
        ):
            return self._enter("free", block, labels)
        if self.scenario.mode == "resist" and contact_range < 0.55:
            self._phase, self._phase_blocks = "stall", 0
            return self._phase_stall(block, labels)
        if self._phase_blocks > 16:
            return self._enter("free", block, labels)
        return self._steer(labels.robot_pose, box.x_m, box.y_m, self._speed)

    def _phase_stall(self, block: int, labels: StateLabels) -> ActionCommand:
        box = labels.object("resistant").pose
        if self._phase_blocks >= self._stall_blocks:
            self._phase, self._phase_blocks = "turn_away", 0
            self._turn_dir = self._rng.choice((-1.0, 1.0))
            return self._phase_turn_away(block, labels)
        return self._steer(labels.robot_pose, box.x_m, box.y_m, self._rng.uniform(0.3, 0.5))

    def _phase_turn_away(self, block: int, labels: StateLabels) -> ActionCommand:
        if self._phase_blocks >= 3:
            return self._enter("free", block, labels)
        return self._noisy(0.1, 0.9 * self._turn_dir)

    def _phase_detour(self, block: int, labels: StateLabels) -> ActionCommand:
        pose = labels.robot_pose
        red = labels.object("resistant").pose
        goal = self.scenario.goal
        if (
            math.dist((pose.x_m, pose.y_m), (goal.x_m, goal.y_m)) < goal.radius_m
            or self._phase_blocks > 24
        ):
            return self._enter("free", block, labels)
        # Waypoint beside the box, perpendicular to the box-to-goal direction.
        gx, gy = goal.x_m - red.x_m, goal.y_m - red.y_m
        norm = math.hypot(gx, gy) or 1.0
        side = (-gy / norm * self._detour_side, gx / norm * self._detour_side)
        waypoint = (red.x_m + side[0] * 0.75, red.y_m + side[1] * 0.75)
        past_box = (pose.x_m - red.x_m) * gx + (pose.y_m - red.y_m) * gy > 0
        target = (goal.x_m, goal.y_m) if past_box else waypoint
        return self._steer(pose, target[0], target[1], 0.4)

    def _phase_free(self, block: int, labels: StateLabels) -> ActionCommand:
        if self._hold_left <= 0 or self._held is None:
            self._held = self._random_command()
            self._hold_left = self._rng.randint(1, 4)
        self._hold_left -= 1
        held = self._held
        if held.forward_velocity_mps == 0.0 and held.yaw_rate_rps == 0.0:
            return held  # exact stops keep the stop family represented in the data
        return self._noisy(held.forward_velocity_mps, held.yaw_rate_rps, scale=0.5)

    def _enter(self, phase: str, block: int, labels: StateLabels) -> ActionCommand:
        self._phase, self._phase_blocks = phase, 0
        return getattr(self, f"_phase_{phase}")(block, labels)

    # ------------------------------------------------------------------ commands

    def _random_command(self) -> ActionCommand:
        roll = self._rng.random()
        b = self.bounds
        if roll < 0.15:
            return b.clip(0.0, 0.0)
        if roll < 0.35:
            return b.clip(
                self._rng.uniform(0.0, 0.1),
                self._rng.choice((-1, 1)) * self._rng.uniform(0.5, b.max_abs_yaw_rps),
            )
        return b.clip(self._rng.uniform(0.1, b.max_forward_mps), self._rng.gauss(0.0, 0.45))

    def _noisy(self, forward: float, yaw: float, scale: float = 1.0) -> ActionCommand:
        return self.bounds.clip(
            forward + self._rng.gauss(0.0, self.noise_forward_mps * scale),
            yaw + self._rng.gauss(0.0, self.noise_yaw_rps * scale),
        )
