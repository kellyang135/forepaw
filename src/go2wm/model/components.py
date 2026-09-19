"""Composable adapter seam for integrating LeWM/PyTorch components later."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .bundle import BundleManifest
from .types import (
    ActionBlock,
    CandidateRollout,
    Latent,
    ModelInput,
    PredictedState,
    as_latent,
)


class CandidateLike(Protocol):
    candidate_id: str
    family: str
    actions: tuple[ActionBlock, ...]


class ObservationEncoder(Protocol):
    """Preprocess and encode one RGB observation into the bundle's latent space."""

    def encode(self, observation: Any) -> Sequence[float]: ...


class ActionConditionedPredictor(Protocol):
    """Autoregress through every command in one candidate."""

    def rollout(
        self,
        history_latents: tuple[Latent, ...],
        history_actions: tuple[ActionBlock, ...],
        candidate_actions: tuple[ActionBlock, ...],
    ) -> Sequence[Sequence[float]]: ...


class LatentReadout(Protocol):
    """Decode robot, object, and risk values from a predicted latent."""

    def decode(self, latent: Latent, *, step: int) -> PredictedState: ...


@dataclass(slots=True)
class ComponentWorldModelBackend:
    """Framework-neutral orchestration around trainable model components.

    A PyTorch adapter only needs to implement the three small protocols above.
    This class ensures observations are encoded once, candidate rollouts have the
    exact requested horizon, and readouts are evaluated on predicted (not merely
    real encoded) latents.
    """

    manifest: BundleManifest
    encoder: ObservationEncoder
    predictor: ActionConditionedPredictor
    readout: LatentReadout

    def predict_candidates(
        self,
        model_input: ModelInput,
        candidates: Sequence[CandidateLike],
    ) -> tuple[CandidateRollout, ...]:
        history_latents = tuple(
            as_latent(self.encoder.encode(observation), expected_dim=self.manifest.latent_dim)
            for observation in model_input.observations
        )
        raw_rollouts = self._rollouts(history_latents, model_input, candidates)
        output: list[CandidateRollout] = []
        for candidate, raw in zip(candidates, raw_rollouts, strict=True):
            predicted = tuple(
                as_latent(latent, expected_dim=self.manifest.latent_dim) for latent in raw
            )
            if len(predicted) != len(candidate.actions):
                raise ValueError(
                    f"predictor returned {len(predicted)} states for candidate "
                    f"{candidate.candidate_id!r} with {len(candidate.actions)} actions"
                )
            states = tuple(
                replace(self.readout.decode(latent, step=step), latent=latent)
                for step, latent in enumerate(predicted, start=1)
            )
            output.append(
                CandidateRollout(
                    candidate_id=candidate.candidate_id,
                    family=candidate.family,
                    actions=candidate.actions,
                    states=states,
                    bundle_id=self.manifest.bundle_id,
                )
            )
        return tuple(output)

    def _rollouts(
        self,
        history_latents: tuple[Latent, ...],
        model_input: ModelInput,
        candidates: Sequence[CandidateLike],
    ) -> list[Sequence[Sequence[float]]]:
        """One predictor call per candidate, or one batched call per horizon if supported.

        Batching changes speed only: a predictor's ``rollout_batch`` must return
        exactly what per-candidate ``rollout`` calls would.
        """

        batch = getattr(self.predictor, "rollout_batch", None)
        if batch is None:
            return [
                self.predictor.rollout(
                    history_latents, model_input.command_history, candidate.actions
                )
                for candidate in candidates
            ]
        results: list[Sequence[Sequence[float]] | None] = [None] * len(candidates)
        by_horizon: dict[int, list[int]] = {}
        for index, candidate in enumerate(candidates):
            by_horizon.setdefault(len(candidate.actions), []).append(index)
        for indexes in by_horizon.values():
            outputs = batch(
                history_latents,
                model_input.command_history,
                [candidates[i].actions for i in indexes],
            )
            if len(outputs) != len(indexes):
                raise ValueError("rollout_batch returned the wrong number of rollouts")
            for i, rollout in zip(indexes, outputs, strict=True):
                results[i] = rollout
        return [item for item in results if item is not None]
