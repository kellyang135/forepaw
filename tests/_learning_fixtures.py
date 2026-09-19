"""Shared fake-backend fixtures for the Person B / ML tests."""

from __future__ import annotations

from pathlib import Path

from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    EpisodeRecord,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
)
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest, write_episode
from go2wm.sim import DeterministicFakeSimulator


def push_episode(
    episode_id: str = "ep-push",
    *,
    seed: int = 7,
    split: DatasetSplit = DatasetSplit.TRAIN,
    actions: tuple[ActionCommand, ...] | None = None,
) -> EpisodeRecord:
    """Robot faces the movable box 0.5 m away, drives into it, then turns and stops."""

    collector = EpisodeCollector(DeterministicFakeSimulator(), CollectionConfig())
    reset = ResetRequest(
        episode_id=episode_id,
        scenario_seed=seed,
        robot_pose=Pose2D(-0.8, 0.0, 0.0),
        objects=(
            ObjectState("light", "blue", Pose2D(-0.3, 0.0), True),
            ObjectState("resistant", "red", Pose2D(0.8, 0.8), False),
        ),
    )
    commands = actions or (
        ActionCommand(0.4, 0.0),
        ActionCommand(0.4, 0.0),
        ActionCommand(0.4, 0.0),
        ActionCommand(0.0, 0.8),
        ActionCommand(0.3, 0.0),
        ActionCommand(0.0, 0.0),
        ActionCommand(0.2, -0.4),
        ActionCommand(0.0, 0.0),
    )
    return collector.collect(
        EpisodeRequest(
            reset=reset,
            split=split,
            goal=Goal2D(1.0, 0.0, 0.15),
            run_id="learning-fixture-run",
        ),
        commands,
    )


def written_episode(root: Path, **kwargs: object) -> Path:
    episode = push_episode(**kwargs)  # type: ignore[arg-type]
    return write_episode(root, episode).episode_directory
