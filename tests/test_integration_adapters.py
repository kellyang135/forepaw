from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
)
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest
from go2wm.integration import action_to_model, action_to_simulator, model_input_before_block
from go2wm.sim import DeterministicFakeSimulator


def test_action_conversion_is_lossless() -> None:
    original = ActionCommand(0.4, -0.3, 0.5)
    assert action_to_simulator(action_to_model(original)) == original


def test_model_input_has_three_frames_and_two_connecting_commands() -> None:
    simulator = DeterministicFakeSimulator()
    collector = EpisodeCollector(simulator, CollectionConfig())
    episode = collector.collect(
        EpisodeRequest(
            reset=ResetRequest(
                episode_id="adapter-test",
                scenario_seed=5,
                robot_pose=Pose2D(-0.8, 0.0, 0.0),
                objects=(
                    ObjectState("light", "blue", Pose2D(0.0, 0.0), True),
                    ObjectState("heavy", "red", Pose2D(0.6, 0.5), False),
                ),
            ),
            split=DatasetSplit.TRAIN,
            goal=Goal2D(1.0, 0.0, 0.1),
            run_id="integration-test-run",
        ),
        [ActionCommand(0.2, 0.0), ActionCommand(0.3, 0.1)],
    )

    initial = model_input_before_block(episode, 0)
    assert len(initial.observations) == 3
    assert len(initial.command_history) == 2
    assert all(action.forward_mps == 0.0 for action in initial.command_history)

    after_first = model_input_before_block(episode, 1)
    assert len(after_first.observations) == 3
    assert len(after_first.command_history) == 2
    assert after_first.command_history[-1].forward_mps == 0.2
    assert after_first.observations[-1] == episode.blocks[0].transition.end_observation
