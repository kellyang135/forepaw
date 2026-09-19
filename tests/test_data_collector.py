from __future__ import annotations

import unittest

from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
)
from go2wm.data.collector import CollectionError, EpisodeCollector, EpisodeRequest
from go2wm.data.manifest import validate_episode
from go2wm.sim.fake import DeterministicFakeSimulator, FakeSimulatorConfig


def request(episode_id: str, seed: int = 1) -> EpisodeRequest:
    return EpisodeRequest(
        reset=ResetRequest(
            episode_id=episode_id,
            scenario_seed=seed,
            robot_pose=Pose2D(-1.0, 0.0, 0.0),
            objects=(ObjectState("light", "red", Pose2D(-0.2, 0.0), True),),
        ),
        split=DatasetSplit.TRAIN,
        goal=Goal2D(1.0, 0.0, 0.2),
        run_id="collector-test-run",
    )


class CollectorTests(unittest.TestCase):
    def test_fresh_three_frame_history_is_not_training_data(self) -> None:
        collector = EpisodeCollector(DeterministicFakeSimulator())
        episode = collector.collect(
            request("episode-a"),
            (ActionCommand(0.4, 0.0), ActionCommand(0.2, 0.3)),
        )
        self.assertEqual(len(episode.initial_history), 3)
        self.assertEqual(len(episode.history_actions), 2)
        self.assertEqual(len(episode.blocks), 2)
        self.assertTrue(
            all(
                action.forward_velocity_mps == 0 and action.yaw_rate_rps == 0
                for action in episode.history_actions
            )
        )
        self.assertIs(
            episode.initial_history[-1],
            episode.blocks[0].transition.start_observation,
        )
        validate_episode(episode)

    def test_reset_creates_a_new_episode_local_history(self) -> None:
        collector = EpisodeCollector(DeterministicFakeSimulator())
        first = collector.collect(request("episode-a", 1), (ActionCommand(0.2, 0.0),))
        second = collector.collect(request("episode-b", 2), (ActionCommand(0.2, 0.0),))
        self.assertTrue(all(obs.episode_id == "episode-a" for obs in first.initial_history))
        self.assertTrue(all(obs.episode_id == "episode-b" for obs in second.initial_history))
        self.assertEqual(second.blocks[0].block_index, 0)

    def test_wrong_action_duration_fails_before_reset(self) -> None:
        simulator = DeterministicFakeSimulator()
        collector = EpisodeCollector(simulator)
        with self.assertRaisesRegex(CollectionError, "duration"):
            collector.collect(request("bad-duration"), (ActionCommand(0.2, 0.0, 0.25),))
        with self.assertRaisesRegex(RuntimeError, "reset"):
            simulator.observe()

    def test_collector_rejects_unsettled_reset(self) -> None:
        simulator = DeterministicFakeSimulator(
            FakeSimulatorConfig(settle_decay=1.0, max_settle_steps=2)
        )
        collector = EpisodeCollector(simulator)
        with self.assertRaisesRegex(CollectionError, "settled reset"):
            collector.collect(request("unsettled"), ())

    def test_logs_requested_and_boundary_clipped_action(self) -> None:
        episode = EpisodeCollector(DeterministicFakeSimulator()).collect(
            request("clipped-action"),
            (ActionCommand(2.5, -4.0),),
        )
        transition = episode.blocks[0].transition
        self.assertEqual(transition.requested_action.forward_velocity_mps, 2.5)
        self.assertEqual(transition.requested_action.yaw_rate_rps, -4.0)
        self.assertEqual(transition.action.forward_velocity_mps, 1.0)
        self.assertEqual(transition.action.yaw_rate_rps, -3.0)
        self.assertEqual(transition.physics_dt_s, 0.01)

    def test_fall_terminates_without_post_termination_blocks(self) -> None:
        episode = EpisodeCollector(DeterministicFakeSimulator()).collect(
            request("fall-stop"),
            (ActionCommand(0.0, 3.0), ActionCommand(0.2, 0.0)),
        )
        self.assertEqual(len(episode.blocks), 1)
        self.assertEqual(episode.blocks[0].termination_reason, "fall")
        self.assertTrue(episode.blocks[0].valid)
        validate_episode(episode)


if __name__ == "__main__":
    unittest.main()
