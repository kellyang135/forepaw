from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
)
from go2wm.data.collector import EpisodeCollector, EpisodeRequest
from go2wm.data.manifest import (
    ManifestValidationError,
    build_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)
from go2wm.sim.fake import DeterministicFakeSimulator


def collect(episode_id: str, seed: int, split: DatasetSplit):
    collector = EpisodeCollector(DeterministicFakeSimulator())
    return collector.collect(
        EpisodeRequest(
            reset=ResetRequest(
                episode_id=episode_id,
                scenario_seed=seed,
                robot_pose=Pose2D(-1.0, 0.0, 0.0),
                objects=(ObjectState("light", "red", Pose2D(-0.1, 0.0), True),),
            ),
            split=split,
            goal=Goal2D(1.0, 0.0, 0.2),
            run_id="manifest-test-run",
        ),
        (ActionCommand(0.4, 0.0), ActionCommand(0.0, 0.4)),
    )


class ManifestTests(unittest.TestCase):
    def test_round_trip_and_counts(self) -> None:
        manifest = build_manifest(
            "tiny-v1",
            (
                collect("train-a", 1, DatasetSplit.TRAIN),
                collect("validation-a", 2, DatasetSplit.VALIDATION),
                collect("test-a", 3, DatasetSplit.TEST),
            ),
        )
        self.assertEqual(len(manifest.episodes), 3)
        self.assertTrue(all(entry.block_count == 2 for entry in manifest.episodes))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, manifest)
            self.assertEqual(read_manifest(path), manifest)

    def test_scenario_seed_cannot_leak_across_splits(self) -> None:
        train = collect("train-a", 44, DatasetSplit.TRAIN)
        test = collect("test-a", 44, DatasetSplit.TEST)
        with self.assertRaisesRegex(ManifestValidationError, "leaks"):
            build_manifest("leaky", (train, test))

    def test_manifest_detects_inconsistent_episode_duration(self) -> None:
        manifest = build_manifest(
            "tiny-v1", (collect("train-a", 1, DatasetSplit.TRAIN),)
        )
        bad_entry = replace(manifest.episodes[0], end_time_s=999.0)
        with self.assertRaisesRegex(ManifestValidationError, "duration"):
            validate_manifest(replace(manifest, episodes=(bad_entry,)))

    def test_duplicate_episode_ids_are_rejected(self) -> None:
        episode = collect("same", 1, DatasetSplit.TRAIN)
        with self.assertRaisesRegex(ManifestValidationError, "duplicate"):
            build_manifest("duplicate", (episode, episode))

    def test_invalid_zero_duration_reports_validation_error(self) -> None:
        manifest = build_manifest(
            "tiny-v1", (collect("train-a", 1, DatasetSplit.TRAIN),)
        )
        with self.assertRaisesRegex(ManifestValidationError, "positive"):
            validate_manifest(replace(manifest, block_duration_s=0.0))


if __name__ == "__main__":
    unittest.main()
