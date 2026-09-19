"""A transparent, CPU-only world model used as a baseline and pipeline stand-in.

``PooledPixelEncoder`` average-pools the RGB frame to an 8x8 grid (8 * 8 * 3 =
192 values, matching the committed latent size).  ``LinearLatentPredictor``
fits ``z[t+1] - z[t] = W [z[t], z[t] - z[t-1], a[t], 1]`` by ridge regression.

Neither component is LeWM.  They exist so every downstream stage (readouts,
horizon metrics, shuffled-action ablation, surprise calibration, bundle
publication, planning) runs end to end before the GPU model is ready, and so
the learned model has a simple learned reference to beat.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from go2wm.contracts import EpisodeRecord, RGBObservation
from go2wm.model import ActionBlock, Latent

from .readouts import DEFAULT_ALPHAS, Standardizer, group_folds, ridge_fit, ridge_predict
from .windows import episode_sequence


@dataclass(frozen=True, slots=True)
class PooledPixelEncoder:
    grid: int = 8
    version: str = "pooled-pixels-8x8-v1"

    @property
    def latent_dim(self) -> int:
        return self.grid * self.grid * 3

    def encode(self, observation: RGBObservation) -> Latent:
        return tuple(self.encode_array(observation).tolist())

    def encode_array(self, observation: RGBObservation) -> np.ndarray:
        height, width = observation.height_px, observation.width_px
        if height < self.grid or width < self.grid:
            raise ValueError("observation is smaller than the pooling grid")
        image = np.frombuffer(observation.rgb, dtype=np.uint8).reshape(height, width, 3)
        rows = np.linspace(0, height, self.grid + 1).astype(int)
        cols = np.linspace(0, width, self.grid + 1).astype(int)
        pooled = np.empty((self.grid, self.grid, 3), dtype=np.float64)
        for i in range(self.grid):
            for j in range(self.grid):
                cell = image[rows[i] : rows[i + 1], cols[j] : cols[j + 1]]
                pooled[i, j] = cell.reshape(-1, 3).mean(axis=0)
        return (pooled / 255.0).ravel()

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "grid": self.grid}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PooledPixelEncoder:
        return cls(grid=int(raw["grid"]), version=raw["version"])


def action_features(action: ActionBlock) -> np.ndarray:
    return np.asarray([action.forward_mps, action.yaw_rate_rps], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LinearLatentPredictor:
    """Implements ``go2wm.model.ActionConditionedPredictor``."""

    weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    alpha: float
    latent_dim: int
    version: str = "linear-latent-delta-v1"

    def step(self, current: np.ndarray, previous: np.ndarray, action: ActionBlock) -> np.ndarray:
        features = np.concatenate([current, current - previous, action_features(action)])
        normalized = (features - self.feature_mean) / self.feature_scale
        return current + ridge_predict(self.weights, normalized[None, :])[0]

    def rollout(
        self,
        history_latents: tuple[Latent, ...],
        history_actions: tuple[ActionBlock, ...],
        candidate_actions: tuple[ActionBlock, ...],
    ) -> Sequence[Sequence[float]]:
        if len(history_latents) < 2:
            raise ValueError("linear predictor needs at least two history latents")
        previous = np.asarray(history_latents[-2], dtype=np.float64)
        current = np.asarray(history_latents[-1], dtype=np.float64)
        outputs: list[tuple[float, ...]] = []
        for action in candidate_actions:
            following = self.step(current, previous, action)
            if not np.isfinite(following).all():
                raise FloatingPointError("linear predictor produced a non-finite latent")
            outputs.append(tuple(following.tolist()))
            previous, current = current, following
        return outputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "latent_dim": self.latent_dim,
            "alpha": self.alpha,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LinearLatentPredictor:
        return cls(
            weights=np.asarray(raw["weights"], dtype=np.float64),
            feature_mean=np.asarray(raw["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(raw["feature_scale"], dtype=np.float64),
            alpha=float(raw["alpha"]),
            latent_dim=int(raw["latent_dim"]),
            version=raw["version"],
        )


def encode_episode(
    encoder: PooledPixelEncoder, episode: EpisodeRecord
) -> tuple[np.ndarray, tuple[ActionBlock, ...]]:
    observations, actions = episode_sequence(episode)
    return np.stack([encoder.encode_array(item) for item in observations]), actions


def _encoded(
    encoder: PooledPixelEncoder,
    episode: EpisodeRecord,
    cache: dict[str, tuple[np.ndarray, tuple[ActionBlock, ...]]] | None,
) -> tuple[np.ndarray, tuple[ActionBlock, ...]]:
    if cache is not None and episode.episode_id in cache:
        return cache[episode.episode_id]
    return encode_episode(encoder, episode)


def fit_linear_predictor(
    encoder: PooledPixelEncoder,
    episodes: Iterable[EpisodeRecord],
    *,
    alpha: float = 10.0,
    encoded: dict[str, tuple[np.ndarray, tuple[ActionBlock, ...]]] | None = None,
) -> LinearLatentPredictor:
    """Fit one-step latent dynamics on training episodes (never validation/test)."""

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for episode in episodes:
        latents, actions = _encoded(encoder, episode, encoded)
        for t in range(1, len(latents) - 1):
            features.append(
                np.concatenate(
                    [latents[t], latents[t] - latents[t - 1], action_features(actions[t])]
                )
            )
            targets.append(latents[t + 1] - latents[t])
    if len(features) < 2:
        raise ValueError("not enough transitions to fit the linear predictor")
    x = np.stack(features)
    y = np.stack(targets)
    norm = Standardizer.fit(x)
    weights = ridge_fit(norm.apply(x), y, alpha=alpha)
    return LinearLatentPredictor(weights, norm.mean, norm.scale, alpha, encoder.latent_dim)


def rollout_latent_error(
    encoder: PooledPixelEncoder,
    predictor: LinearLatentPredictor,
    episodes: Iterable[EpisodeRecord],
    *,
    horizon: int = 6,
    encoded: dict[str, tuple[np.ndarray, tuple[ActionBlock, ...]]] | None = None,
) -> float:
    """Median latent RMSE after ``horizon`` autoregressive steps over every start."""

    errors: list[float] = []
    for episode in episodes:
        latents, actions = _encoded(encoder, episode, encoded)
        for start in range(1, len(latents) - horizon):
            previous, current = latents[start - 1], latents[start]
            with np.errstate(all="ignore"):
                for step in range(horizon):
                    previous, current = (
                        current,
                        predictor.step(current, previous, actions[start + step]),
                    )
                error = float(np.sqrt(np.mean((current - latents[start + horizon]) ** 2)))
            errors.append(error if np.isfinite(error) else float("inf"))
    return float(np.median(errors)) if errors else float("inf")


def select_predictor_alpha(
    encoder: PooledPixelEncoder,
    episodes: Sequence[EpisodeRecord],
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    folds: int = 5,
    horizon: int = 6,
) -> tuple[float, dict[str, float]]:
    """Episode-grouped CV on training data, scored on the 6-step (3 s) rollout.

    Scoring the full horizon, not one step, rejects penalties whose dynamics are
    accurate for one block but diverge when rolled out.
    """

    fold_indices = group_folds([e.episode_id for e in episodes], folds)
    encoded = {e.episode_id: encode_episode(encoder, e) for e in episodes}
    scores: dict[str, float] = {}
    for alpha in alphas:
        fold_scores = []
        for held in fold_indices:
            held_set = set(held.tolist())
            train = [e for i, e in enumerate(episodes) if i not in held_set]
            predictor = fit_linear_predictor(encoder, train, alpha=alpha, encoded=encoded)
            fold_scores.append(
                rollout_latent_error(
                    encoder,
                    predictor,
                    [episodes[i] for i in held],
                    horizon=horizon,
                    encoded=encoded,
                )
            )
        scores[f"{alpha:g}"] = float(np.mean(fold_scores))
    best = min(alphas, key=lambda a: scores[f"{a:g}"])
    return float(best), scores
