"""Auditable on-disk episode writer using only the Python standard library."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from go2wm.contracts import ActionCommand, EpisodeRecord, RGBObservation, StateLabels
from go2wm.data.manifest import validate_episode

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class EpisodeArtifacts:
    episode_directory: Path
    metadata_path: Path
    frames_index_path: Path
    blocks_path: Path
    physics_samples_path: Path
    checksums_path: Path
    frame_count: int
    block_count: int
    physics_sample_count: int


def write_episode(root: str | Path, episode: EpisodeRecord) -> EpisodeArtifacts:
    """Write one complete episode without overwriting existing evidence.

    RGB payloads are stored as raw packed bytes and described in
    ``frames.jsonl``.  High-frequency physics samples live separately from
    ``blocks.jsonl`` so a brief contact or fall remains inspectable after the
    aggregate is computed.
    """

    validate_episode(episode)
    if not _SAFE_IDENTIFIER.fullmatch(episode.episode_id):
        raise ValueError(
            "episode_id must contain only letters, digits, '.', '_', and '-'"
        )
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / episode.episode_id
    if target.exists():
        raise FileExistsError(f"refusing to overwrite episode directory {target}")
    staging = root_path / f".{episode.episode_id}.tmp-{uuid.uuid4().hex}"
    frames_dir = staging / "frames"
    frames_dir.mkdir(parents=True)

    try:
        observations = _ordered_observations(episode)
        frame_rows: list[dict[str, Any]] = []
        observation_files: dict[str, str] = {}
        for index, observation in enumerate(observations):
            relative_path = f"frames/{index:06d}.rgb"
            (staging / relative_path).write_bytes(observation.rgb)
            observation_files[observation.observation_id] = relative_path
            frame_rows.append(
                {
                    "observation_id": observation.observation_id,
                    "episode_id": observation.episode_id,
                    "sim_time_s": observation.sim_time_s,
                    "camera_id": observation.camera_id,
                    "width_px": observation.width_px,
                    "height_px": observation.height_px,
                    "encoding": "rgb8-packed",
                    "path": relative_path,
                    "sha256": observation.sha256,
                }
            )

        block_rows: list[dict[str, Any]] = []
        physics_rows: list[dict[str, Any]] = []
        for block in episode.blocks:
            transition = block.transition
            block_rows.append(
                {
                    "schema_version": "go2wm.block.v2",
                    "run_id": episode.run_id,
                    "episode_id": block.episode_id,
                    "block_index": block.block_index,
                    "collected_at_utc": block.collected_at_utc,
                    "sim_start_time_s": transition.start_observation.sim_time_s,
                    "sim_end_time_s": transition.end_observation.sim_time_s,
                    "start_observation_id": transition.start_observation.observation_id,
                    "end_observation_id": transition.end_observation.observation_id,
                    "start_frame_path": observation_files[
                        transition.start_observation.observation_id
                    ],
                    "end_frame_path": observation_files[
                        transition.end_observation.observation_id
                    ],
                    "requested_action": _action_dict(transition.requested_action),
                    "applied_action": _action_dict(transition.action),
                    "action_units": {
                        "forward_velocity": "m/s",
                        "yaw_rate": "rad/s",
                        "duration": "s",
                    },
                    "start_labels": _labels_dict(transition.start_labels),
                    "end_labels": _labels_dict(transition.end_labels),
                    "events": asdict(transition.events),
                    "physics_sample_count": len(transition.physics_samples),
                    "physics_dt_s": transition.physics_dt_s,
                    "valid": block.valid,
                    "termination_reason": block.termination_reason,
                }
            )
            for sample_index, sample in enumerate(transition.physics_samples):
                physics_rows.append(
                    {
                        "episode_id": block.episode_id,
                        "block_index": block.block_index,
                        "sample_index": sample_index,
                        "sim_time_s": sample.sim_time_s,
                        "fallen": sample.fallen,
                        "out_of_bounds": sample.out_of_bounds,
                        "contacts": [asdict(contact) for contact in sample.contacts],
                    }
                )

        metadata = {
            "schema_version": "go2wm.episode.v2",
            "run_id": episode.run_id,
            "episode_id": episode.episode_id,
            "split": episode.split.value,
            "scenario_seed": episode.scenario_seed,
            "scene_id": episode.scene_id,
            "camera_id": episode.camera_id,
            "goal": asdict(episode.goal),
            "reset": {
                "settled": episode.reset_report.settled,
                "settle_steps": episode.reset_report.settle_steps,
                "settle_duration_s": episode.reset_report.settle_duration_s,
                "final_observation_id": episode.reset_report.final_observation.observation_id,
                "final_labels": _labels_dict(episode.reset_report.final_labels),
            },
            "initial_history_observation_ids": [
                observation.observation_id for observation in episode.initial_history
            ],
            "history_actions": [_action_dict(action) for action in episode.history_actions],
            "block_count": len(episode.blocks),
            "metadata": dict(episode.metadata),
        }
        (staging / "episode.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_jsonl(staging / "frames.jsonl", frame_rows)
        _write_jsonl(staging / "blocks.jsonl", block_rows)
        _write_jsonl(staging / "physics_samples.jsonl", physics_rows)
        _write_checksums(staging)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return EpisodeArtifacts(
        episode_directory=target,
        metadata_path=target / "episode.json",
        frames_index_path=target / "frames.jsonl",
        blocks_path=target / "blocks.jsonl",
        physics_samples_path=target / "physics_samples.jsonl",
        checksums_path=target / "checksums.sha256",
        frame_count=len(observations),
        block_count=len(block_rows),
        physics_sample_count=len(physics_rows),
    )


def _ordered_observations(episode: EpisodeRecord) -> tuple[RGBObservation, ...]:
    ordered: list[RGBObservation] = []
    by_id: dict[str, RGBObservation] = {}
    candidates = list(episode.initial_history)
    for block in episode.blocks:
        candidates.extend(
            (block.transition.start_observation, block.transition.end_observation)
        )
    for observation in candidates:
        prior = by_id.get(observation.observation_id)
        if prior is not None:
            if prior != observation:
                raise ValueError(
                    f"observation id {observation.observation_id!r} has conflicting payloads"
                )
            continue
        by_id[observation.observation_id] = observation
        ordered.append(observation)
    return tuple(ordered)


def _action_dict(action: ActionCommand) -> dict[str, float]:
    return {
        "forward_velocity_mps": action.forward_velocity_mps,
        "yaw_rate_rps": action.yaw_rate_rps,
        "duration_s": action.duration_s,
    }


def _labels_dict(labels: StateLabels) -> dict[str, Any]:
    return {
        "sim_time_s": labels.sim_time_s,
        "robot_pose": asdict(labels.robot_pose),
        "objects": [asdict(item) for item in labels.objects],
        "forward_velocity_mps": labels.forward_velocity_mps,
        "yaw_rate_rps": labels.yaw_rate_rps,
        "fallen": labels.fallen,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_checksums(directory: Path) -> None:
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    rows = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(directory).as_posix()}")
    (directory / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
