from __future__ import annotations

import math
import tempfile
import unittest

from go2wm.contracts import ActionCommand, DatasetSplit
from go2wm.data import CollectionError, EpisodeCollector, EpisodeRequest, write_episode
from go2wm.data.scripted_policy import ArenaGeometry, CommandBounds, ScenarioSampler, ScriptedPolicy
from go2wm.learning.dataset import load_dataset
from go2wm.sim.fake import DeterministicFakeSimulator

ARENA = ArenaGeometry(3.0, 3.0, box_half_extent_m=0.16)
BLOCKS = 40


def collect(seed: int, sampler: ScenarioSampler | None = None):
    sampler = sampler or ScenarioSampler(ARENA)
    scenario = sampler.sample(seed)
    policy = ScriptedPolicy(scenario, ARENA, CommandBounds(), seed=seed)
    request = EpisodeRequest(scenario.reset, DatasetSplit.TRAIN, scenario.goal, "policy-test")
    record = EpisodeCollector(DeterministicFakeSimulator()).collect_with_policy(
        request, policy, BLOCKS
    )
    return scenario, record


class ScenarioSamplerTests(unittest.TestCase):
    def test_same_seed_gives_same_scenario(self) -> None:
        sampler = ScenarioSampler(ARENA)
        self.assertEqual(sampler.sample(11), sampler.sample(11))

    def test_resets_respect_clearance_and_margins(self) -> None:
        sampler = ScenarioSampler(ARENA)
        for seed in range(1, 120):
            reset = sampler.sample(seed).reset
            robot = reset.robot_pose
            self.assertTrue(ARENA.inside(robot.x_m, robot.y_m, ARENA.spawn_margin_m))
            classes = {item.appearance_class for item in reset.objects}
            self.assertEqual(classes, {"blue", "red"})
            for item in reset.objects:
                gap = math.dist((robot.x_m, robot.y_m), (item.pose.x_m, item.pose.y_m))
                self.assertGreaterEqual(gap, ARENA.robot_box_clearance_m, f"seed {seed}")

    def test_rejects_unknown_modes(self) -> None:
        with self.assertRaises(ValueError):
            ScenarioSampler(ARENA, mode_weights={"dance": 1.0})


class ScriptedPolicyTests(unittest.TestCase):
    def test_commands_stay_inside_bounds(self) -> None:
        bounds = CommandBounds()
        for seed in range(1, 25):
            _, record = collect(seed)
            for block in record.blocks:
                action = block.transition.action
                self.assertLessEqual(action.forward_velocity_mps, bounds.max_forward_mps)
                self.assertGreaterEqual(action.forward_velocity_mps, bounds.min_forward_mps)
                self.assertLessEqual(abs(action.yaw_rate_rps), bounds.max_abs_yaw_rps)

    def test_no_episode_leaves_the_arena(self) -> None:
        for seed in range(1, 41):
            _, record = collect(seed)
            for block in record.blocks:
                self.assertFalse(block.transition.events.out_of_bounds, f"seed {seed}")

    def test_push_mode_moves_the_light_box(self) -> None:
        sampler = ScenarioSampler(ARENA, mode_weights={"push": 1.0})
        moved = 0
        for seed in range(1, 11):
            scenario, record = collect(seed, sampler)
            start = {o.object_id: o.pose for o in scenario.reset.objects}["light"]
            end = {o.object_id: o.pose for o in record.blocks[-1].transition.end_labels.objects}[
                "light"
            ]
            moved += math.dist((start.x_m, start.y_m), (end.x_m, end.y_m)) > 0.1
        self.assertGreaterEqual(moved, 8)

    def test_policy_mode_mix_reaches_every_mode(self) -> None:
        sampler = ScenarioSampler(ARENA)
        modes = {sampler.sample(seed).mode for seed in range(1, 80)}
        self.assertEqual(modes, {"push", "resist", "detour", "free"})


class CollectWithPolicyTests(unittest.TestCase):
    def test_policy_command_off_the_block_contract_is_refused(self) -> None:
        scenario = ScenarioSampler(ARENA).sample(3)
        request = EpisodeRequest(scenario.reset, DatasetSplit.TRAIN, scenario.goal, "policy-test")
        collector = EpisodeCollector(DeterministicFakeSimulator())
        with self.assertRaises(CollectionError):
            collector.collect_with_policy(request, lambda _i, _l: ActionCommand(0.3, 0.0, 0.25), 3)

    def test_policy_episodes_round_trip_through_the_strict_loader(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            for seed in (1, 2, 3):
                write_episode(root, collect(seed)[1])
            dataset = load_dataset(root)
        self.assertEqual(len(dataset.episodes), 3)
        self.assertEqual(dataset.rejected, ())
        self.assertTrue(all(len(e.blocks) == BLOCKS for e in dataset.episodes))


if __name__ == "__main__":
    unittest.main()
