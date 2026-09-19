"""Explicit conversions between collection and model/planning contracts."""

from __future__ import annotations

from go2wm.contracts import ActionCommand, EpisodeRecord, RGBObservation
from go2wm.model import ActionBlock, ModelInput


def action_to_model(action: ActionCommand) -> ActionBlock:
    return ActionBlock(
        forward_mps=action.forward_velocity_mps,
        yaw_rate_rps=action.yaw_rate_rps,
        duration_s=action.duration_s,
    )


def action_to_simulator(action: ActionBlock) -> ActionCommand:
    return ActionCommand(
        forward_velocity_mps=action.forward_mps,
        yaw_rate_rps=action.yaw_rate_rps,
        duration_s=action.duration_s,
    )


def model_input_before_block(
    episode: EpisodeRecord,
    block_index: int,
    *,
    history_frames: int = 3,
) -> ModelInput:
    """Build an aligned visual/action history immediately before one block.

    A three-frame input always contains exactly two commands: each command is
    the transition from observation ``i`` to observation ``i + 1``. Future or
    privileged labels are never exposed.
    """

    if block_index < 0 or block_index > len(episode.blocks):
        raise IndexError("block_index must name an existing boundary")
    observations: list[RGBObservation] = list(episode.initial_history)
    commands: list[ActionCommand] = list(episode.history_actions)
    for block in episode.blocks[:block_index]:
        observations.append(block.transition.end_observation)
        commands.append(block.transition.action)

    if len(observations) < history_frames:
        raise ValueError("episode does not contain enough observations for model history")
    selected_observations = tuple(observations[-history_frames:])
    selected_commands = tuple(
        action_to_model(action) for action in commands[-(history_frames - 1) :]
    )
    result = ModelInput(
        observations=selected_observations,
        command_history=selected_commands,
    )
    result.validate(
        history_frames=history_frames,
        block_duration_s=selected_commands[0].duration_s,
    )
    return result

