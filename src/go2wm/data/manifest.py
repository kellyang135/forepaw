"""Dataset manifest creation, persistence, and leakage checks."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from go2wm.contracts import (
    TIME_TOLERANCE_S,
    DatasetSplit,
    EpisodeRecord,
    observations_match,
)


class ManifestValidationError(ValueError):
    """Raised when a dataset would violate the frozen data contract."""


@dataclass(frozen=True, slots=True)
class EpisodeManifestEntry:
    episode_id: str
    split: DatasetSplit
    scenario_seed: int
    scene_id: str
    camera_id: str
    block_count: int
    history_observation_count: int
    first_training_time_s: float
    end_time_s: float
    interaction_block_count: int
    fall_block_count: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    schema_version: str
    block_duration_s: float
    observation_rate_hz: float
    history_observations: int
    frameskip: int
    episodes: tuple[EpisodeManifestEntry, ...]
    metadata: Mapping[str, str]


def validate_episode(
    episode: EpisodeRecord,
    *,
    block_duration_s: float = 0.5,
    history_observations: int = 3,
) -> None:
    """Validate timestamps, boundaries, history, indices, and episode ownership."""

    errors: list[str] = []
    if len(episode.initial_history) != history_observations:
        errors.append(
            f"history has {len(episode.initial_history)} observations; "
            f"expected {history_observations}"
        )
    if len(episode.history_actions) != max(0, history_observations - 1):
        errors.append("history does not have exactly N-1 connecting commands")

    for index, action in enumerate(episode.history_actions):
        if not math.isclose(action.duration_s, block_duration_s, abs_tol=TIME_TOLERANCE_S):
            errors.append(f"history action {index} has the wrong duration")
        if action.forward_velocity_mps != 0 or action.yaw_rate_rps != 0:
            errors.append(f"history action {index} is not a zero command")

    for index, (left, right) in enumerate(
        zip(episode.initial_history, episode.initial_history[1:], strict=False)
    ):
        if not math.isclose(
            right.sim_time_s - left.sim_time_s,
            block_duration_s,
            abs_tol=TIME_TOLERANCE_S,
        ):
            errors.append(f"history frame interval {index} has the wrong duration")

    previous = episode.initial_history[-1] if episode.initial_history else None
    for expected_index, block in enumerate(episode.blocks):
        transition = block.transition
        if block.block_index != expected_index:
            errors.append(f"block {expected_index} has index {block.block_index}")
        if block.episode_id != episode.episode_id:
            errors.append(f"block {expected_index} crosses an episode boundary")
        if not math.isclose(
            transition.action.duration_s,
            block_duration_s,
            abs_tol=TIME_TOLERANCE_S,
        ):
            errors.append(f"block {expected_index} has the wrong action duration")
        if previous is not None and not observations_match(
            previous, transition.start_observation
        ):
            errors.append(f"block {expected_index} start does not match prior boundary")
        previous = transition.end_observation

    if errors:
        raise ManifestValidationError(
            f"invalid episode {episode.episode_id!r}: " + "; ".join(errors)
        )


def build_manifest(
    dataset_id: str,
    episodes: Iterable[EpisodeRecord],
    *,
    block_duration_s: float = 0.5,
    history_observations: int = 3,
    schema_version: str = "go2wm.dataset.v1",
    metadata: Mapping[str, str] | None = None,
) -> DatasetManifest:
    records = tuple(episodes)
    entries: list[EpisodeManifestEntry] = []
    for episode in records:
        validate_episode(
            episode,
            block_duration_s=block_duration_s,
            history_observations=history_observations,
        )
        first_time = episode.initial_history[-1].sim_time_s
        end_time = (
            episode.blocks[-1].transition.end_observation.sim_time_s
            if episode.blocks
            else first_time
        )
        entries.append(
            EpisodeManifestEntry(
                episode_id=episode.episode_id,
                split=episode.split,
                scenario_seed=episode.scenario_seed,
                scene_id=episode.scene_id,
                camera_id=episode.camera_id,
                block_count=len(episode.blocks),
                history_observation_count=len(episode.initial_history),
                first_training_time_s=first_time,
                end_time_s=end_time,
                interaction_block_count=sum(
                    bool(block.transition.events.contacted_object_ids)
                    for block in episode.blocks
                ),
                fall_block_count=sum(
                    block.transition.events.fell for block in episode.blocks
                ),
            )
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        schema_version=schema_version,
        block_duration_s=block_duration_s,
        observation_rate_hz=1.0 / block_duration_s,
        history_observations=history_observations,
        frameskip=1,
        episodes=tuple(entries),
        metadata=dict(metadata or {}),
    )
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: DatasetManifest) -> None:
    """Reject incomplete metadata and episode/seed leakage across data splits."""

    errors: list[str] = []
    if not isinstance(manifest.dataset_id, str) or not manifest.dataset_id.strip():
        errors.append("dataset_id is empty")
    if not math.isfinite(manifest.block_duration_s) or manifest.block_duration_s <= 0:
        errors.append("block_duration_s must be positive")
    else:
        if not math.isfinite(manifest.observation_rate_hz) or not math.isclose(
            manifest.observation_rate_hz,
            1.0 / manifest.block_duration_s,
            rel_tol=1e-9,
        ):
            errors.append("observation rate does not match block duration")
    if manifest.history_observations < 2:
        errors.append("history_observations must be at least two")
    if manifest.frameskip != 1:
        errors.append("frameskip must be 1 after block aggregation")

    seen_ids: set[str] = set()
    seed_splits: dict[int, DatasetSplit] = {}
    for entry in manifest.episodes:
        if entry.episode_id in seen_ids:
            errors.append(f"duplicate episode id {entry.episode_id!r}")
        seen_ids.add(entry.episode_id)
        previous_split = seed_splits.setdefault(entry.scenario_seed, entry.split)
        if previous_split != entry.split:
            errors.append(
                f"scenario seed {entry.scenario_seed} leaks across "
                f"{previous_split.value} and {entry.split.value}"
            )
        if entry.block_count < 0:
            errors.append(f"episode {entry.episode_id!r} has negative block count")
        if entry.history_observation_count != manifest.history_observations:
            errors.append(f"episode {entry.episode_id!r} has the wrong history length")
        if not math.isfinite(entry.first_training_time_s) or not math.isfinite(
            entry.end_time_s
        ):
            errors.append(f"episode {entry.episode_id!r} has non-finite timestamps")
        elif manifest.block_duration_s > 0:
            expected_end = (
                entry.first_training_time_s
                + entry.block_count * manifest.block_duration_s
            )
            if not math.isclose(
                entry.end_time_s, expected_end, abs_tol=TIME_TOLERANCE_S
            ):
                errors.append(f"episode {entry.episode_id!r} duration is inconsistent")
        if not entry.scene_id or not entry.camera_id:
            errors.append(f"episode {entry.episode_id!r} lacks scene/camera identity")
        if not 0 <= entry.interaction_block_count <= entry.block_count:
            errors.append(f"episode {entry.episode_id!r} has invalid interaction count")
        if not 0 <= entry.fall_block_count <= entry.block_count:
            errors.append(f"episode {entry.episode_id!r} has invalid fall count")

    if errors:
        raise ManifestValidationError("invalid dataset manifest: " + "; ".join(errors))


def write_manifest(path: str | Path, manifest: DatasetManifest) -> None:
    validate_manifest(manifest)
    payload = asdict(manifest)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> DatasetManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = tuple(
        EpisodeManifestEntry(
            episode_id=item["episode_id"],
            split=DatasetSplit(item["split"]),
            scenario_seed=item["scenario_seed"],
            scene_id=item["scene_id"],
            camera_id=item["camera_id"],
            block_count=item["block_count"],
            history_observation_count=item["history_observation_count"],
            first_training_time_s=item["first_training_time_s"],
            end_time_s=item["end_time_s"],
            interaction_block_count=item["interaction_block_count"],
            fall_block_count=item["fall_block_count"],
        )
        for item in payload["episodes"]
    )
    manifest = DatasetManifest(
        dataset_id=payload["dataset_id"],
        schema_version=payload["schema_version"],
        block_duration_s=payload["block_duration_s"],
        observation_rate_hz=payload["observation_rate_hz"],
        history_observations=payload["history_observations"],
        frameskip=payload["frameskip"],
        episodes=entries,
        metadata=payload.get("metadata", {}),
    )
    validate_manifest(manifest)
    return manifest
