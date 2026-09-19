from __future__ import annotations

import json
import tempfile
import unittest

from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
)
from go2wm.data.collector import EpisodeCollector, EpisodeRequest
from go2wm.data.storage import write_episode
from go2wm.sim.fake import DeterministicFakeSimulator


class EpisodeStorageTests(unittest.TestCase):
    def _episode(self, episode_id: str = "storage-test"):
        return EpisodeCollector(DeterministicFakeSimulator()).collect(
            EpisodeRequest(
                reset=ResetRequest(
                    episode_id=episode_id,
                    scenario_seed=9,
                    robot_pose=Pose2D(-0.8, 0.0),
                    objects=(ObjectState("light", "red", Pose2D(-0.25, 0.0), True),),
                ),
                split=DatasetSplit.TRAIN,
                goal=Goal2D(1.0, 0.0, 0.2),
                run_id="storage-test-run",
                metadata={"collector_commit": "test-only"},
            ),
            (ActionCommand(0.8, 0.0),),
        )

    def test_writes_frames_blocks_and_all_substep_samples(self) -> None:
        episode = self._episode()
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_episode(directory, episode)
            self.assertEqual(artifacts.block_count, 1)
            self.assertEqual(artifacts.physics_sample_count, 50)
            self.assertEqual(artifacts.frame_count, 4)
            self.assertTrue(artifacts.checksums_path.is_file())
            metadata = json.loads(artifacts.metadata_path.read_text())
            self.assertEqual(metadata["schema_version"], "go2wm.episode.v2")
            self.assertEqual(metadata["run_id"], "storage-test-run")
            self.assertEqual(metadata["initial_history_observation_ids"], [
                observation.observation_id for observation in episode.initial_history
            ])
            rows = [
                json.loads(line)
                for line in artifacts.physics_samples_path.read_text().splitlines()
            ]
            self.assertEqual(len(rows), 50)
            self.assertTrue(any(row["contacts"] for row in rows))
            frame_rows = [
                json.loads(line)
                for line in artifacts.frames_index_path.read_text().splitlines()
            ]
            for row in frame_rows:
                self.assertTrue((artifacts.episode_directory / row["path"]).is_file())
            block = json.loads(artifacts.blocks_path.read_text().splitlines()[0])
            self.assertEqual(block["schema_version"], "go2wm.block.v2")
            self.assertEqual(block["requested_action"], block["applied_action"])
            self.assertEqual(block["physics_dt_s"], 0.01)
            self.assertEqual(block["physics_sample_count"], 50)
            self.assertTrue(block["valid"])
            self.assertEqual(block["termination_reason"], "fixed_length")

    def test_refuses_to_overwrite_existing_episode(self) -> None:
        episode = self._episode()
        with tempfile.TemporaryDirectory() as directory:
            write_episode(directory, episode)
            with self.assertRaises(FileExistsError):
                write_episode(directory, episode)

    def test_rejects_path_like_episode_id(self) -> None:
        episode = self._episode("../escape")
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "episode_id"),
        ):
            write_episode(directory, episode)


if __name__ == "__main__":
    unittest.main()
