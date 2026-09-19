"""Build surprise-calibration pairs from held-out normal windows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from go2wm.model import ActionConditionedPredictor, ObservationEncoder
from go2wm.runtime import SurpriseCalibration, calibrate_surprise, latent_discrepancy

from .windows import PredictionWindow


def one_step_pairs(
    windows: Sequence[PredictionWindow],
    *,
    encoder: ObservationEncoder,
    predictor: ActionConditionedPredictor,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    """Pair the latent predicted for ``t_{k+1}`` with the encoder's latent of that frame.

    This is exactly the comparison the runtime ``SurpriseStopGuard`` makes:
    the selected plan's first predicted latent versus the next observation.
    """

    pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for window in windows:
        history = tuple(tuple(encoder.encode(obs)) for obs in window.model_input.observations)
        predicted = predictor.rollout(
            history, window.model_input.command_history, window.future_actions[:1]
        )
        observed = tuple(encoder.encode(window.future_observations[0]))
        pairs.append((tuple(predicted[0]), observed))
    return pairs


def fit_surprise(
    normal_windows: Sequence[PredictionWindow],
    *,
    encoder: ObservationEncoder,
    predictor: ActionConditionedPredictor,
    bundle_id: str,
    calibration_version: str,
    quantile: float = 0.99,
    consecutive_breaches: int = 1,
) -> SurpriseCalibration:
    """Calibrate on validation windows that contain ordinary pushes and turns only."""

    pairs = one_step_pairs(normal_windows, encoder=encoder, predictor=predictor)
    return calibrate_surprise(
        pairs,
        bundle_id=bundle_id,
        calibration_version=calibration_version,
        metric="rmse",
        quantile=quantile,
        consecutive_breaches=consecutive_breaches,
    )


def false_stop_fraction(
    windows: Sequence[PredictionWindow],
    calibration: SurpriseCalibration,
    *,
    encoder: ObservationEncoder,
    predictor: ActionConditionedPredictor,
) -> float:
    """Fraction of normal one-step transitions whose residual exceeds the threshold."""

    pairs = one_step_pairs(windows, encoder=encoder, predictor=predictor)
    residuals = np.asarray([latent_discrepancy(p, o, metric=calibration.metric) for p, o in pairs])
    return float((residuals > calibration.threshold).mean()) if len(residuals) else 0.0
