"""Strict consumer for episodes written by ``go2wm.data.storage.write_episode``.

The loader rebuilds full ``EpisodeRecord`` values so every constructor-level
contract check runs again on the consumer side.  Anything suspicious raises
``DatasetLoadError``; nothing is silently skipped or repaired.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from go2wm.contracts import (
    ActionCommand,
    BlockEvents,
    BlockTransition,
    ContactSample,
    DatasetSplit,
    EpisodeBlock,
    EpisodeRecord,
    Goal2D,
    ObjectState,
    PhysicsSample,
    Pose2D,
    ResetReport,
    RGBObservation,
    StateLabels,
)
from go2wm.data.manifest import ManifestValidationError, validate_episode

SUPPORTED_EPISODE_SCHEMAS = frozenset({"go2wm.episode.v2"})
SUPPORTED_BLOCK_SCHEMAS = frozenset({"go2wm.block.v2"})


class DatasetLoadError(ValueError):
    """Raised when on-disk data violates the frozen data contract."""


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    root: Path
    episodes: tuple[EpisodeRecord, ...]
    rejected: tuple[tuple[str, str], ...]

    def by_split(self, split: DatasetSplit) -> tuple[EpisodeRecord, ...]:
        return tuple(episode for episode in self.episodes if episode.split == split)


def load_episode(
    directory: str | Path,
    *,
    block_duration_s: float = 0.5,
    history_observations: int = 3,
) -> EpisodeRecord:
    path = Path(directory)
    try:
        return _load_episode(path, block_duration_s, history_observations)
    except DatasetLoadError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        ManifestValidationError,
    ) as error:
        raise DatasetLoadError(f"{path}: {type(error).__name__}: {error}") from error


def load_dataset(
    root: str | Path,
    *,
    strict: bool = True,
    block_duration_s: float = 0.5,
    history_observations: int = 3,
) -> LoadedDataset:
    """Load every episode directory below ``root``.

    ``strict=True`` stops on the first invalid episode.  ``strict=False`` keeps
    going but returns every rejection with its reason so it can be reported;
    it never drops an episode silently.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise DatasetLoadError(f"dataset root {root_path} is not a directory")
    episodes: list[EpisodeRecord] = []
    rejected: list[tuple[str, str]] = []
    for directory in sorted(p for p in root_path.iterdir() if p.is_dir()):
        if directory.name.startswith("."):
            continue
        if not (directory / "episode.json").exists():
            continue
        try:
            episodes.append(
                load_episode(
                    directory,
                    block_duration_s=block_duration_s,
                    history_observations=history_observations,
                )
            )
        except DatasetLoadError as error:
            if strict:
                raise
            rejected.append((directory.name, str(error)))
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise DatasetLoadError("duplicate episode ids across directories")
    return LoadedDataset(root_path, tuple(episodes), tuple(rejected))


def _load_episode(path: Path, block_duration_s: float, history_observations: int) -> EpisodeRecord:
    meta = _read_json(path / "episode.json")
    schema = meta.get("schema_version")
    if schema not in SUPPORTED_EPISODE_SCHEMAS:
        raise DatasetLoadError(f"{path}: unsupported episode schema {schema!r}")
    episode_id = _str(meta, "episode_id")
    if path.name != episode_id:
        raise DatasetLoadError(f"{path}: directory name does not match episode_id")

    frames = _load_frames(path, episode_id)
    block_rows = _read_jsonl(path / "blocks.jsonl")
    physics_rows = _read_jsonl(path / "physics_samples.jsonl")
    samples_by_block = _group_physics(physics_rows, episode_id)

    declared_count = meta.get("block_count")
    if declared_count != len(block_rows):
        raise DatasetLoadError(
            f"{path}: episode.json declares {declared_count} blocks, "
            f"blocks.jsonl has {len(block_rows)}"
        )
    indexes = [row.get("block_index") for row in block_rows]
    if indexes != list(range(len(block_rows))):
        raise DatasetLoadError(f"{path}: block indexes are not contiguous from zero: {indexes}")
    if set(samples_by_block) - set(indexes):
        raise DatasetLoadError(f"{path}: physics samples reference unknown blocks")

    blocks: list[EpisodeBlock] = []
    for row in block_rows:
        index = row["block_index"]
        if row.get("schema_version") not in SUPPORTED_BLOCK_SCHEMAS:
            raise DatasetLoadError(f"{path}: block {index} has an unsupported schema")
        if row.get("run_id") != meta.get("run_id"):
            raise DatasetLoadError(f"{path}: block {index} run_id mismatch")
        if row.get("episode_id") != episode_id:
            raise DatasetLoadError(f"{path}: block {index} belongs to another episode")
        if row.get("action_units") != {
            "forward_velocity": "m/s",
            "yaw_rate": "rad/s",
            "duration": "s",
        }:
            raise DatasetLoadError(f"{path}: block {index} action units mismatch")
        samples = samples_by_block.get(index, ())
        if len(samples) != row.get("physics_sample_count"):
            raise DatasetLoadError(f"{path}: block {index} physics sample count mismatch")
        transition = BlockTransition(
            start_observation=_frame(frames, row["start_observation_id"], path),
            end_observation=_frame(frames, row["end_observation_id"], path),
            action=_action(row["applied_action"]),
            start_labels=_labels(row["start_labels"]),
            end_labels=_labels(row["end_labels"]),
            physics_samples=samples,
            events=_events(row["events"]),
            requested_action=_action(row["requested_action"]),
            physics_dt_s=_finite(row["physics_dt_s"], "physics_dt_s"),
        )
        if not math.isclose(
            _finite(row["sim_start_time_s"], "sim_start_time_s"),
            transition.start_observation.sim_time_s,
            abs_tol=1e-9,
        ) or not math.isclose(
            _finite(row["sim_end_time_s"], "sim_end_time_s"),
            transition.end_observation.sim_time_s,
            abs_tol=1e-9,
        ):
            raise DatasetLoadError(f"{path}: block {index} duplicate timestamps disagree")
        termination_reason = row.get("termination_reason")
        if termination_reason is not None and (
            not isinstance(termination_reason, str) or not termination_reason.strip()
        ):
            raise DatasetLoadError(f"{path}: block {index} has invalid termination_reason")
        blocks.append(
            EpisodeBlock(
                episode_id=episode_id,
                block_index=index,
                transition=transition,
                collected_at_utc=_str(row, "collected_at_utc"),
                valid=_bool(row.get("valid"), "valid"),
                termination_reason=termination_reason,
            )
        )

    reset = meta["reset"]
    reset_report = ResetReport(
        episode_id=episode_id,
        scenario_seed=_int(meta, "scenario_seed"),
        settled=bool(reset["settled"]),
        settle_steps=int(reset["settle_steps"]),
        settle_duration_s=_finite(reset["settle_duration_s"], "settle_duration_s"),
        final_observation=_frame(frames, reset["final_observation_id"], path),
        final_labels=_labels(reset["final_labels"]),
    )
    if not reset_report.settled:
        raise DatasetLoadError(f"{path}: episode was recorded from an unsettled reset")
    history_ids = meta["initial_history_observation_ids"]
    record = EpisodeRecord(
        episode_id=episode_id,
        run_id=_str(meta, "run_id"),
        split=DatasetSplit(meta["split"]),
        scenario_seed=_int(meta, "scenario_seed"),
        scene_id=_str(meta, "scene_id"),
        camera_id=_str(meta, "camera_id"),
        goal=Goal2D(**{k: _finite(v, f"goal.{k}") for k, v in meta["goal"].items()}),
        reset_report=reset_report,
        initial_history=tuple(_frame(frames, item, path) for item in history_ids),
        history_actions=tuple(_action(item) for item in meta["history_actions"]),
        blocks=tuple(blocks),
        metadata={str(k): str(v) for k, v in dict(meta.get("metadata", {})).items()},
    )
    if record.initial_history[0] is not reset_report.final_observation:
        raise DatasetLoadError(f"{path}: history does not start at the settled reset frame")
    _check_monotonic(record, path)
    validate_episode(
        record,
        block_duration_s=block_duration_s,
        history_observations=history_observations,
    )
    _verify_checksums(path)
    return record


def _load_frames(path: Path, episode_id: str) -> dict[str, RGBObservation]:
    frames: dict[str, RGBObservation] = {}
    for row in _read_jsonl(path / "frames.jsonl"):
        observation_id = row["observation_id"]
        if observation_id in frames:
            raise DatasetLoadError(f"{path}: duplicate frame id {observation_id!r}")
        if row.get("encoding") != "rgb8-packed":
            raise DatasetLoadError(f"{path}: unsupported frame encoding {row.get('encoding')!r}")
        if row.get("episode_id") != episode_id:
            raise DatasetLoadError(f"{path}: frame {observation_id!r} belongs to another episode")
        frame_path = path / row["path"]
        if not frame_path.resolve().is_relative_to(path.resolve()):
            raise DatasetLoadError(f"{path}: frame path escapes the episode directory")
        if not frame_path.exists():
            raise DatasetLoadError(f"{path}: missing frame file {row['path']}")
        payload = frame_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise DatasetLoadError(f"{path}: checksum mismatch for {row['path']}")
        frames[observation_id] = RGBObservation(
            observation_id=observation_id,
            episode_id=episode_id,
            sim_time_s=_finite(row["sim_time_s"], "sim_time_s"),
            camera_id=row["camera_id"],
            width_px=int(row["width_px"]),
            height_px=int(row["height_px"]),
            rgb=payload,
        )
    return frames


def _group_physics(
    rows: Iterable[Mapping[str, Any]], episode_id: str
) -> dict[int, tuple[PhysicsSample, ...]]:
    grouped: dict[int, list[tuple[int, PhysicsSample]]] = {}
    for row in rows:
        if row.get("episode_id") != episode_id:
            raise DatasetLoadError("physics sample belongs to another episode")
        sample = PhysicsSample(
            sim_time_s=_finite(row["sim_time_s"], "physics.sim_time_s"),
            contacts=tuple(
                ContactSample(
                    object_id=item["object_id"],
                    normal_impulse_ns=_finite(item["normal_impulse_ns"], "impulse"),
                )
                for item in row["contacts"]
            ),
            fallen=bool(row["fallen"]),
            out_of_bounds=_bool(row.get("out_of_bounds"), "out_of_bounds"),
        )
        grouped.setdefault(int(row["block_index"]), []).append((int(row["sample_index"]), sample))
    result: dict[int, tuple[PhysicsSample, ...]] = {}
    for block_index, items in grouped.items():
        order = [index for index, _ in items]
        if order != list(range(len(items))):
            raise DatasetLoadError(f"block {block_index} physics samples are out of order")
        result[block_index] = tuple(sample for _, sample in items)
    return result


def _check_monotonic(record: EpisodeRecord, path: Path) -> None:
    times = [observation.sim_time_s for observation in record.initial_history]
    times.extend(block.transition.end_observation.sim_time_s for block in record.blocks)
    for left, right in pairwise(times):
        if right <= left:
            raise DatasetLoadError(f"{path}: non-monotonic observation timestamps")


def _frame(frames: Mapping[str, RGBObservation], observation_id: str, path: Path) -> RGBObservation:
    try:
        return frames[observation_id]
    except KeyError:
        raise DatasetLoadError(f"{path}: missing frame {observation_id!r}") from None


def _action(raw: Mapping[str, Any]) -> ActionCommand:
    return ActionCommand(
        forward_velocity_mps=_finite(raw["forward_velocity_mps"], "forward_velocity_mps"),
        yaw_rate_rps=_finite(raw["yaw_rate_rps"], "yaw_rate_rps"),
        duration_s=_finite(raw["duration_s"], "duration_s"),
    )


def _labels(raw: Mapping[str, Any]) -> StateLabels:
    return StateLabels(
        sim_time_s=_finite(raw["sim_time_s"], "labels.sim_time_s"),
        robot_pose=_pose(raw["robot_pose"]),
        objects=tuple(
            ObjectState(
                object_id=item["object_id"],
                appearance_class=item["appearance_class"],
                pose=_pose(item["pose"]),
                movable=bool(item["movable"]),
            )
            for item in raw["objects"]
        ),
        forward_velocity_mps=_finite(raw["forward_velocity_mps"], "labels.forward"),
        yaw_rate_rps=_finite(raw["yaw_rate_rps"], "labels.yaw_rate"),
        fallen=bool(raw["fallen"]),
    )


def _pose(raw: Mapping[str, Any]) -> Pose2D:
    return Pose2D(
        _finite(raw["x_m"], "x_m"), _finite(raw["y_m"], "y_m"), _finite(raw["yaw_rad"], "yaw_rad")
    )


def _events(raw: Mapping[str, Any]) -> BlockEvents:
    return BlockEvents(
        contacted_object_ids=tuple(raw["contacted_object_ids"]),
        contact_sample_count=int(raw["contact_sample_count"]),
        first_contact_time_s=raw["first_contact_time_s"],
        last_contact_time_s=raw["last_contact_time_s"],
        max_normal_impulse_ns=_finite(raw["max_normal_impulse_ns"], "max_normal_impulse_ns"),
        fell=bool(raw["fell"]),
        fall_sample_count=int(raw["fall_sample_count"]),
        first_fall_time_s=raw["first_fall_time_s"],
        out_of_bounds=_bool(raw.get("out_of_bounds"), "events.out_of_bounds"),
        out_of_bounds_sample_count=int(raw["out_of_bounds_sample_count"]),
        first_out_of_bounds_time_s=raw["first_out_of_bounds_time_s"],
    )


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetLoadError(f"{name} must be numeric, got {value!r}")
    if not math.isfinite(value):
        raise DatasetLoadError(f"{name} must be finite, got {value!r}")
    return float(value)


def _str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise DatasetLoadError(f"{key} must be a non-empty string")
    return value


def _int(raw: Mapping[str, Any], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetLoadError(f"{key} must be an integer")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetLoadError(f"{name} must be a bool")
    return value


def _verify_checksums(path: Path) -> None:
    checksum_path = path / "checksums.sha256"
    if not checksum_path.is_file():
        raise DatasetLoadError(f"{path}: missing checksums.sha256")
    declared: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise DatasetLoadError(
                f"{path}: malformed checksums.sha256 line {line_number}"
            ) from error
        candidate = path / relative
        if not candidate.resolve().is_relative_to(path.resolve()):
            raise DatasetLoadError(f"{path}: checksum path escapes episode directory")
        if relative in declared:
            raise DatasetLoadError(f"{path}: duplicate checksum entry {relative!r}")
        declared[relative] = digest
    actual_paths = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "checksums.sha256"
    }
    if set(declared) != actual_paths:
        raise DatasetLoadError(f"{path}: checksum inventory does not match episode files")
    for relative, expected in declared.items():
        if len(expected) != 64:
            raise DatasetLoadError(f"{path}: invalid checksum for {relative}")
        observed = hashlib.sha256((path / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise DatasetLoadError(f"{path}: artifact checksum mismatch for {relative}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetLoadError(f"missing {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DatasetLoadError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DatasetLoadError(f"missing {path.name}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DatasetLoadError(f"{path.name}:{line_number} is not an object")
            rows.append(row)
    return rows
