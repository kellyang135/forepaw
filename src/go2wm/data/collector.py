"""Block-aligned episode collection."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    EpisodeBlock,
    EpisodeRecord,
    Goal2D,
    ResetRequest,
    RGBObservation,
)
from go2wm.sim.base import SimulatorAdapter


class CollectionError(RuntimeError):
    """Raised when a simulator cannot satisfy the collection contract."""


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    block_duration_s: float = 0.5
    history_observations: int = 3
    max_episode_blocks: int = 256

    def __post_init__(self) -> None:
        if self.block_duration_s <= 0:
            raise ValueError("block_duration_s must be positive")
        if self.history_observations < 2:
            raise ValueError("history_observations must be at least two")
        if self.max_episode_blocks <= 0:
            raise ValueError("max_episode_blocks must be positive")


@dataclass(frozen=True, slots=True)
class EpisodeRequest:
    reset: ResetRequest
    split: DatasetSplit
    goal: Goal2D
    run_id: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip():
            raise ValueError("run_id must be non-empty")


class EpisodeCollector:
    """Collect complete episodes with fresh, episode-local history.

    History consists of N observations and N-1 connecting zero commands.  The
    warm-up transitions are intentionally not returned as training blocks.
    """

    def __init__(
        self,
        simulator: SimulatorAdapter,
        config: CollectionConfig | None = None,
    ) -> None:
        self.simulator = simulator
        self.config = config or CollectionConfig()
        if not math.isclose(
            self.simulator.block_duration_s,
            self.config.block_duration_s,
            abs_tol=1e-9,
        ):
            raise ValueError("collector and simulator block durations differ")

    def collect(
        self,
        request: EpisodeRequest,
        actions: Iterable[ActionCommand],
    ) -> EpisodeRecord:
        commands = tuple(actions)
        if len(commands) > self.config.max_episode_blocks:
            raise CollectionError(
                f"episode has {len(commands)} blocks; maximum is "
                f"{self.config.max_episode_blocks}"
            )
        self._validate_commands(commands)

        reset_report = self.simulator.reset(request.reset)
        if not reset_report.settled:
            raise CollectionError(
                f"episode {request.reset.episode_id!r} did not reach a settled reset"
            )

        history: list[RGBObservation] = [reset_report.final_observation]
        history_actions: list[ActionCommand] = []
        for _ in range(self.config.history_observations - 1):
            stop = ActionCommand.stopped(self.config.block_duration_s)
            warmup = self.simulator.execute_block(stop)
            self._require_boundary_match(history[-1], warmup.start_observation)
            history_actions.append(stop)
            history.append(warmup.end_observation)

        blocks: list[EpisodeBlock] = []
        previous = history[-1]
        for index, action in enumerate(commands):
            transition = self.simulator.execute_block(action)
            self._require_boundary_match(previous, transition.start_observation)
            if transition.events.fell:
                termination_reason = "fall"
            elif transition.events.out_of_bounds:
                termination_reason = "out_of_bounds"
            elif index == len(commands) - 1:
                termination_reason = "fixed_length"
            else:
                termination_reason = None
            blocks.append(
                EpisodeBlock(
                    episode_id=request.reset.episode_id,
                    block_index=index,
                    transition=transition,
                    collected_at_utc=datetime.now(UTC).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    valid=True,
                    termination_reason=termination_reason,
                )
            )
            previous = transition.end_observation
            if termination_reason in {"fall", "out_of_bounds"}:
                break

        return EpisodeRecord(
            episode_id=request.reset.episode_id,
            run_id=request.run_id,
            split=request.split,
            scenario_seed=request.reset.scenario_seed,
            scene_id=self.simulator.scene_config.scene_id,
            camera_id=self.simulator.camera_config.camera_id,
            goal=request.goal,
            reset_report=reset_report,
            initial_history=tuple(history),
            history_actions=tuple(history_actions),
            blocks=tuple(blocks),
            metadata=dict(request.metadata),
        )

    def _validate_commands(self, actions: tuple[ActionCommand, ...]) -> None:
        for index, action in enumerate(actions):
            if not math.isclose(
                action.duration_s,
                self.config.block_duration_s,
                abs_tol=1e-9,
            ):
                raise CollectionError(
                    f"action {index} duration {action.duration_s} does not match "
                    f"the {self.config.block_duration_s}-second data contract"
                )

    @staticmethod
    def _require_boundary_match(
        expected: RGBObservation, actual: RGBObservation
    ) -> None:
        if expected is actual:
            return
        if (
            expected.episode_id != actual.episode_id
            or expected.camera_id != actual.camera_id
            or not math.isclose(expected.sim_time_s, actual.sim_time_s, abs_tol=1e-9)
            or expected.rgb != actual.rgb
        ):
            raise CollectionError("simulator changed state between adjacent block boundaries")
