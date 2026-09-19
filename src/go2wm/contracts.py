"""Shared, dependency-light contracts for the Go2 world-model pipeline.

The dataclasses in this module deliberately distinguish runtime observations from
privileged simulator labels.  A deployed controller may consume
``RGBObservation`` and ``ActionCommand``; ``StateLabels`` and event summaries are
for collection, training, and evaluation only.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from itertools import pairwise

TIME_TOLERANCE_S = 1e-9
DEFAULT_BLOCK_DURATION_S = 0.5


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_nonempty(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_utc_timestamp(name: str, value: str) -> None:
    _require_nonempty(name, value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        _require_finite("x_m", self.x_m)
        _require_finite("y_m", self.y_m)
        _require_finite("yaw_rad", self.yaw_rad)


@dataclass(frozen=True, slots=True)
class Goal2D:
    x_m: float
    y_m: float
    radius_m: float

    def __post_init__(self) -> None:
        _require_finite("goal.x_m", self.x_m)
        _require_finite("goal.y_m", self.y_m)
        _require_finite("goal.radius_m", self.radius_m)
        if self.radius_m <= 0:
            raise ValueError("goal.radius_m must be positive")


@dataclass(frozen=True, slots=True)
class ActionCommand:
    """Command actually applied for one simulator block."""

    forward_velocity_mps: float
    yaw_rate_rps: float
    duration_s: float = DEFAULT_BLOCK_DURATION_S

    def __post_init__(self) -> None:
        _require_finite("forward_velocity_mps", self.forward_velocity_mps)
        _require_finite("yaw_rate_rps", self.yaw_rate_rps)
        _require_finite("duration_s", self.duration_s)
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")

    @classmethod
    def stopped(cls, duration_s: float = DEFAULT_BLOCK_DURATION_S) -> ActionCommand:
        return cls(0.0, 0.0, duration_s)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    width_px: int
    height_px: int
    rgb_order: str = "RGB"
    vertical_fov_deg: float = 60.0

    def __post_init__(self) -> None:
        _require_nonempty("camera_id", self.camera_id)
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.rgb_order != "RGB":
            raise ValueError("only packed RGB observations are supported")
        _require_finite("vertical_fov_deg", self.vertical_fov_deg)
        if not 0 < self.vertical_fov_deg < 180:
            raise ValueError("vertical_fov_deg must be between 0 and 180")


@dataclass(frozen=True, slots=True)
class SceneConfig:
    scene_id: str
    arena_width_m: float
    arena_height_m: float

    def __post_init__(self) -> None:
        _require_nonempty("scene_id", self.scene_id)
        _require_finite("arena_width_m", self.arena_width_m)
        _require_finite("arena_height_m", self.arena_height_m)
        if self.arena_width_m <= 0 or self.arena_height_m <= 0:
            raise ValueError("arena dimensions must be positive")


@dataclass(frozen=True, slots=True)
class RGBObservation:
    """A rendered runtime observation.

    Pixels are packed row-major RGB8 bytes.  This intentionally avoids a NumPy
    dependency in the simulator/data boundary.
    """

    observation_id: str
    episode_id: str
    sim_time_s: float
    camera_id: str
    width_px: int
    height_px: int
    rgb: bytes = field(repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("episode_id", self.episode_id),
            ("camera_id", self.camera_id),
        ):
            _require_nonempty(name, value)
        _require_finite("sim_time_s", self.sim_time_s)
        if self.sim_time_s < 0:
            raise ValueError("sim_time_s must be non-negative")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("observation dimensions must be positive")
        expected = self.width_px * self.height_px * 3
        if len(self.rgb) != expected:
            raise ValueError(f"expected {expected} RGB bytes, got {len(self.rgb)}")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.rgb).hexdigest()


@dataclass(frozen=True, slots=True)
class ObjectState:
    """Privileged object label; never a runtime planner input."""

    object_id: str
    appearance_class: str
    pose: Pose2D
    movable: bool

    def __post_init__(self) -> None:
        _require_nonempty("object_id", self.object_id)
        _require_nonempty("appearance_class", self.appearance_class)


@dataclass(frozen=True, slots=True)
class StateLabels:
    """Privileged simulator state aligned to one rendered observation."""

    sim_time_s: float
    robot_pose: Pose2D
    objects: tuple[ObjectState, ...]
    forward_velocity_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    fallen: bool = False

    def __post_init__(self) -> None:
        _require_finite("labels.sim_time_s", self.sim_time_s)
        _require_finite("labels.forward_velocity_mps", self.forward_velocity_mps)
        _require_finite("labels.yaw_rate_rps", self.yaw_rate_rps)
        ids = [item.object_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")

    def object(self, object_id: str) -> ObjectState:
        for item in self.objects:
            if item.object_id == object_id:
                return item
        raise KeyError(object_id)


@dataclass(frozen=True, slots=True)
class ContactSample:
    object_id: str
    normal_impulse_ns: float

    def __post_init__(self) -> None:
        _require_nonempty("contact.object_id", self.object_id)
        _require_finite("normal_impulse_ns", self.normal_impulse_ns)
        if self.normal_impulse_ns < 0:
            raise ValueError("normal_impulse_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class PhysicsSample:
    """High-frequency information captured after one physics substep."""

    sim_time_s: float
    contacts: tuple[ContactSample, ...] = ()
    fallen: bool = False
    out_of_bounds: bool = False

    def __post_init__(self) -> None:
        _require_finite("physics_sample.sim_time_s", self.sim_time_s)


@dataclass(frozen=True, slots=True)
class BlockEvents:
    """Lossless-enough summary of events anywhere inside an action block."""

    contacted_object_ids: tuple[str, ...]
    contact_sample_count: int
    first_contact_time_s: float | None
    last_contact_time_s: float | None
    max_normal_impulse_ns: float
    fell: bool
    fall_sample_count: int
    first_fall_time_s: float | None
    out_of_bounds: bool
    out_of_bounds_sample_count: int
    first_out_of_bounds_time_s: float | None

    @classmethod
    def aggregate(cls, samples: Iterable[PhysicsSample]) -> BlockEvents:
        ids: set[str] = set()
        contact_count = 0
        first_contact: float | None = None
        last_contact: float | None = None
        max_impulse = 0.0
        fall_count = 0
        first_fall: float | None = None
        out_of_bounds_count = 0
        first_out_of_bounds: float | None = None
        for sample in samples:
            if sample.contacts:
                first_contact = (
                    sample.sim_time_s if first_contact is None else first_contact
                )
                last_contact = sample.sim_time_s
            for contact in sample.contacts:
                ids.add(contact.object_id)
                contact_count += 1
                max_impulse = max(max_impulse, contact.normal_impulse_ns)
            if sample.fallen:
                fall_count += 1
                first_fall = sample.sim_time_s if first_fall is None else first_fall
            if sample.out_of_bounds:
                out_of_bounds_count += 1
                first_out_of_bounds = (
                    sample.sim_time_s
                    if first_out_of_bounds is None
                    else first_out_of_bounds
                )
        return cls(
            contacted_object_ids=tuple(sorted(ids)),
            contact_sample_count=contact_count,
            first_contact_time_s=first_contact,
            last_contact_time_s=last_contact,
            max_normal_impulse_ns=max_impulse,
            fell=fall_count > 0,
            fall_sample_count=fall_count,
            first_fall_time_s=first_fall,
            out_of_bounds=out_of_bounds_count > 0,
            out_of_bounds_sample_count=out_of_bounds_count,
            first_out_of_bounds_time_s=first_out_of_bounds,
        )


@dataclass(frozen=True, slots=True)
class BlockTransition:
    """Atomic result of one command block, including both boundary frames."""

    start_observation: RGBObservation
    end_observation: RGBObservation
    action: ActionCommand
    start_labels: StateLabels
    end_labels: StateLabels
    physics_samples: tuple[PhysicsSample, ...]
    events: BlockEvents
    requested_action: ActionCommand
    physics_dt_s: float

    def __post_init__(self) -> None:
        if self.start_observation.episode_id != self.end_observation.episode_id:
            raise ValueError("a block cannot cross an episode boundary")
        if self.start_observation.camera_id != self.end_observation.camera_id:
            raise ValueError("a block cannot switch cameras")
        if (
            self.start_observation.width_px,
            self.start_observation.height_px,
        ) != (self.end_observation.width_px, self.end_observation.height_px):
            raise ValueError("a block cannot change observation dimensions")
        start = self.start_observation.sim_time_s
        end = self.end_observation.sim_time_s
        if not math.isclose(start, self.start_labels.sim_time_s, abs_tol=TIME_TOLERANCE_S):
            raise ValueError("start observation and labels are not time-aligned")
        if not math.isclose(end, self.end_labels.sim_time_s, abs_tol=TIME_TOLERANCE_S):
            raise ValueError("end observation and labels are not time-aligned")
        if not math.isclose(end - start, self.action.duration_s, abs_tol=TIME_TOLERANCE_S):
            raise ValueError("block timestamps do not match action duration")
        if not math.isclose(
            self.requested_action.duration_s,
            self.action.duration_s,
            abs_tol=TIME_TOLERANCE_S,
        ):
            raise ValueError("requested and applied action durations differ")
        _require_finite("physics_dt_s", self.physics_dt_s)
        if self.physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be positive")
        if not self.physics_samples:
            raise ValueError("a block must include high-frequency physics samples")
        if not math.isclose(
            len(self.physics_samples) * self.physics_dt_s,
            self.action.duration_s,
            abs_tol=TIME_TOLERANCE_S,
        ):
            raise ValueError("physics step count/timestep do not match action duration")
        sample_times = tuple(sample.sim_time_s for sample in self.physics_samples)
        if sample_times[0] <= start + TIME_TOLERANCE_S:
            raise ValueError("first physics sample must occur after the block start")
        if any(
            right <= left + TIME_TOLERANCE_S
            for left, right in pairwise(sample_times)
        ):
            raise ValueError("physics sample timestamps must be strictly increasing")
        if not math.isclose(sample_times[-1], end, abs_tol=TIME_TOLERANCE_S):
            raise ValueError("last physics sample must align with the block end")
        for index, sample_time in enumerate(sample_times, start=1):
            expected = start + index * self.physics_dt_s
            if not math.isclose(sample_time, expected, abs_tol=TIME_TOLERANCE_S):
                raise ValueError("physics sample timestamp does not match simulator timestep")
        start_object_ids = tuple(item.object_id for item in self.start_labels.objects)
        end_object_ids = tuple(item.object_id for item in self.end_labels.objects)
        if start_object_ids != end_object_ids:
            raise ValueError("object identity/order changed within an action block")
        if self.events != BlockEvents.aggregate(self.physics_samples):
            raise ValueError("block event summary does not match physics samples")


@dataclass(frozen=True, slots=True)
class ResetRequest:
    episode_id: str
    scenario_seed: int
    robot_pose: Pose2D
    objects: tuple[ObjectState, ...]

    def __post_init__(self) -> None:
        _require_nonempty("episode_id", self.episode_id)
        if not isinstance(self.scenario_seed, int):
            raise TypeError("scenario_seed must be an int")
        ids = [item.object_id for item in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("reset object ids must be unique")


@dataclass(frozen=True, slots=True)
class ResetReport:
    episode_id: str
    scenario_seed: int
    settled: bool
    settle_steps: int
    settle_duration_s: float
    final_observation: RGBObservation
    final_labels: StateLabels

    def __post_init__(self) -> None:
        if self.episode_id != self.final_observation.episode_id:
            raise ValueError("reset report observation belongs to another episode")
        if self.settle_steps < 0 or self.settle_duration_s < 0:
            raise ValueError("settle counts and duration must be non-negative")
        if not math.isclose(
            self.final_observation.sim_time_s,
            self.final_labels.sim_time_s,
            abs_tol=TIME_TOLERANCE_S,
        ):
            raise ValueError("reset observation and labels are not time-aligned")


@dataclass(frozen=True, slots=True)
class EpisodeBlock:
    episode_id: str
    block_index: int
    transition: BlockTransition
    collected_at_utc: str
    valid: bool
    termination_reason: str | None

    def __post_init__(self) -> None:
        _require_nonempty("episode_id", self.episode_id)
        if self.block_index < 0:
            raise ValueError("block_index must be non-negative")
        if self.transition.start_observation.episode_id != self.episode_id:
            raise ValueError("block transition belongs to another episode")
        _require_utc_timestamp("collected_at_utc", self.collected_at_utc)
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        if self.termination_reason is not None:
            _require_nonempty("termination_reason", self.termination_reason)
        if not self.valid and self.termination_reason is None:
            raise ValueError("an invalid block requires a termination reason")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    run_id: str
    split: DatasetSplit
    scenario_seed: int
    scene_id: str
    camera_id: str
    goal: Goal2D
    reset_report: ResetReport
    initial_history: tuple[RGBObservation, ...]
    history_actions: tuple[ActionCommand, ...]
    blocks: tuple[EpisodeBlock, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty("episode_id", self.episode_id)
        _require_nonempty("run_id", self.run_id)
        _require_nonempty("scene_id", self.scene_id)
        _require_nonempty("camera_id", self.camera_id)
        if self.reset_report.episode_id != self.episode_id:
            raise ValueError("reset report belongs to another episode")
        if not self.initial_history:
            raise ValueError("episode requires a fresh observation history")
        if len(self.history_actions) != len(self.initial_history) - 1:
            raise ValueError("N history observations require N-1 connecting actions")
        for observation in self.initial_history:
            if observation.episode_id != self.episode_id:
                raise ValueError("history observation belongs to another episode")
            if observation.camera_id != self.camera_id:
                raise ValueError("history observation uses another camera")
        for expected_index, block in enumerate(self.blocks):
            if block.episode_id != self.episode_id:
                raise ValueError("block belongs to another episode")
            if block.block_index != expected_index:
                raise ValueError("episode block indices must be contiguous from zero")
            if block.termination_reason is not None and expected_index != len(self.blocks) - 1:
                raise ValueError("an episode cannot contain blocks after termination")


def observations_match(left: RGBObservation, right: RGBObservation) -> bool:
    """Compare boundary observations without requiring the same frame id."""

    return (
        left.episode_id == right.episode_id
        and left.camera_id == right.camera_id
        and left.width_px == right.width_px
        and left.height_px == right.height_px
        and math.isclose(left.sim_time_s, right.sim_time_s, abs_tol=TIME_TOLERANCE_S)
        and left.rgb == right.rgb
    )
