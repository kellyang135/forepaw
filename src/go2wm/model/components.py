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
            as_latent(
                self.encoder.encode(observation), expected_dim=self.manifest.latent_dim
            )
            for observation in model_input.observations
        )
        output: list[CandidateRollout] = []
        for candidate in candidates:
            predicted = tuple(
                as_latent(latent, expected_dim=self.manifest.latent_dim)
                for latent in self.predictor.rollout(
                    history_latents,
                    model_input.command_history,
                    candidate.actions,
                )
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

