"""Aligned training/evaluation windows cut from validated episodes.

A window starting at block ``k`` holds the three observations ending at
``t_k``, the two commands connecting them, the next ``horizon`` commands, and
the privileged labels at ``t_k`` .. ``t_{k+horizon}``.  Labels are used only to
fit readouts and score predictions; they never enter ``model_input``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from go2wm.contracts import EpisodeRecord, RGBObservation, StateLabels
from go2wm.integration.adapters import action_to_model, model_input_before_block
from go2wm.model import ActionBlock, ModelInput

from .diagnostics import classify_block


@dataclass(frozen=True, slots=True)
class PredictionWindow:
    episode_id: str
    scenario_seed: int
    start_block: int
    model_input: ModelInput
    future_actions: tuple[ActionBlock, ...]
    start_labels: StateLabels
    future_labels: tuple[StateLabels, ...]
    future_observations: tuple[RGBObservation, ...]
    interaction_types: tuple[str, ...]

    @property
    def horizon(self) -> int:
        return len(self.future_actions)

    @property
    def has_interaction(self) -> bool:
        return any(kind != "free" for kind in self.interaction_types)


def episode_windows(
    episode: EpisodeRecord,
    *,
    horizon: int = 6,
    history_frames: int = 3,
    stride: int = 1,
) -> tuple[PredictionWindow, ...]:
    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    windows: list[PredictionWindow] = []
    for start in range(0, len(episode.blocks) - horizon + 1, stride):
        blocks = episode.blocks[start : start + horizon]
        windows.append(
            PredictionWindow(
                episode_id=episode.episode_id,
                scenario_seed=episode.scenario_seed,
                start_block=start,
                model_input=model_input_before_block(episode, start, history_frames=history_frames),
                future_actions=tuple(action_to_model(b.transition.action) for b in blocks),
                start_labels=blocks[0].transition.start_labels,
                future_labels=tuple(b.transition.end_labels for b in blocks),
                future_observations=tuple(b.transition.end_observation for b in blocks),
                interaction_types=tuple(classify_block(b.transition) for b in blocks),
            )
        )
    return tuple(windows)


def dataset_windows(
    episodes: Iterable[EpisodeRecord],
    *,
    horizon: int = 6,
    history_frames: int = 3,
    stride: int = 1,
) -> tuple[PredictionWindow, ...]:
    result: list[PredictionWindow] = []
    for episode in episodes:
        result.extend(
            episode_windows(episode, horizon=horizon, history_frames=history_frames, stride=stride)
        )
    return tuple(result)


def labeled_observations(
    episodes: Iterable[EpisodeRecord],
) -> tuple[tuple[RGBObservation, StateLabels], ...]:
    """Every observation that has aligned labels: the reset frame plus each block end."""

    pairs: list[tuple[RGBObservation, StateLabels]] = []
    for episode in episodes:
        if episode.blocks:
            first = episode.blocks[0].transition
            pairs.append((first.start_observation, first.start_labels))
        pairs.extend(
            (block.transition.end_observation, block.transition.end_labels)
            for block in episode.blocks
        )
    return tuple(pairs)


def episode_sequence(
    episode: EpisodeRecord,
) -> tuple[tuple[RGBObservation, ...], tuple[ActionBlock, ...]]:
    """Full observation stream and the N-1 commands between consecutive frames."""

    observations = list(episode.initial_history)
    actions = [action_to_model(action) for action in episode.history_actions]
    for block in episode.blocks:
        observations.append(block.transition.end_observation)
        actions.append(action_to_model(block.transition.action))
    return tuple(observations), tuple(actions)
