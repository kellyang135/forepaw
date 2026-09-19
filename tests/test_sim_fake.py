from __future__ import annotations

import unittest

from go2wm.contracts import ActionCommand, ObjectState, Pose2D, ResetRequest
from go2wm.sim.fake import DeterministicFakeSimulator, FakeSimulatorConfig


def reset_request(episode_id: str = "episode-a", seed: int = 11) -> ResetRequest:
    return ResetRequest(
        episode_id=episode_id,
        scenario_seed=seed,
        robot_pose=Pose2D(-0.8, 0.0, 0.0),
        objects=(
            ObjectState("light", "red", Pose2D(-0.25, 0.0), True),
            ObjectState("heavy", "blue", Pose2D(0.9, 0.7), False),
        ),
    )


class FakeSimulatorTests(unittest.TestCase):
    def test_exact_half_second_block_and_high_frequency_samples(self) -> None:
        simulator = DeterministicFakeSimulator()
        report = simulator.reset(reset_request())
        self.assertTrue(report.settled)
        transition = simulator.execute_block(ActionCommand(0.8, 0.0))
        self.assertAlmostEqual(
            transition.end_observation.sim_time_s
            - transition.start_observation.sim_time_s,
            0.5,
        )
        self.assertEqual(len(transition.physics_samples), 50)
        self.assertIn("light", transition.events.contacted_object_ids)

    def test_same_reset_and_commands_are_deterministic(self) -> None:
        def run() -> tuple[object, ...]:
            simulator = DeterministicFakeSimulator()
            report = simulator.reset(reset_request())
            outputs = [report.final_observation.sha256]
            for command in (ActionCommand(0.5, 0.2), ActionCommand(0.2, -0.4)):
                transition = simulator.execute_block(command)
                outputs.extend(
                    (
                        transition.end_observation.sha256,
                        transition.end_labels.robot_pose,
                        transition.end_labels.objects,
                        transition.events,
                    )
                )
            return tuple(outputs)

        self.assertEqual(run(), run())

    def test_unsettled_reset_prevents_execution(self) -> None:
        simulator = DeterministicFakeSimulator(
            FakeSimulatorConfig(settle_decay=1.0, max_settle_steps=4)
        )
        report = simulator.reset(reset_request())
        self.assertFalse(report.settled)
        with self.assertRaisesRegex(RuntimeError, "unsettled"):
            simulator.execute_block(ActionCommand(0.1, 0.0))

    def test_immovable_object_blocks_robot_while_movable_object_moves(self) -> None:
        movable = DeterministicFakeSimulator()
        movable.reset(reset_request())
        before_light = movable.labels().object("light").pose.x_m
        movable.execute_block(ActionCommand(0.8, 0.0))
        after_light = movable.labels().object("light").pose.x_m
        self.assertGreater(after_light, before_light)

        blocked_request = ResetRequest(
            episode_id="blocked",
            scenario_seed=12,
            robot_pose=Pose2D(-0.8, 0.0, 0.0),
            objects=(ObjectState("heavy", "blue", Pose2D(-0.25, 0.0), False),),
        )
        blocked = DeterministicFakeSimulator()
        blocked.reset(blocked_request)
        transition = blocked.execute_block(ActionCommand(0.8, 0.0))
        self.assertIn("heavy", transition.events.contacted_object_ids)
        self.assertAlmostEqual(
            blocked.labels().object("heavy").pose.x_m,
            -0.25,
        )


if __name__ == "__main__":
    unittest.main()

