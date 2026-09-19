"""Calibrated latent-surprise monitoring with a latched emergency stop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Literal

from go2wm.model import BundleCompatibilityError, BundleManifest, Latent, as_latent

SurpriseMetric = Literal["rmse", "cosine"]


@dataclass(frozen=True, slots=True)
class SurpriseCalibration:
    """Calibration fitted only from held-out, non-anomalous operation."""

    calibration_version: str
    bundle_id: str
    latent_dim: int
    metric: SurpriseMetric
    threshold: float
    consecutive_breaches: int = 1
    normal_sample_count: int = 0
    quantile: float = 0.99

    def __post_init__(self) -> None:
        if not self.calibration_version or not self.bundle_id:
            raise ValueError("calibration_version and bundle_id must not be empty")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.metric not in ("rmse", "cosine"):
            raise ValueError(f"unsupported surprise metric: {self.metric!r}")
        if self.threshold < 0:
            raise ValueError("surprise threshold must be non-negative")
        if self.consecutive_breaches < 1:
            raise ValueError("consecutive_breaches must be at least one")
        if not 0.0 < self.quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")

    def validate_manifest(self, manifest: BundleManifest) -> None:
        mismatches: list[str] = []
        if self.bundle_id != manifest.bundle_id:
            mismatches.append(
                f"surprise bundle_id {self.bundle_id!r} does not match "
                f"model {manifest.bundle_id!r}"
            )
        if self.calibration_version != manifest.surprise_calibration_version:
            mismatches.append(
                f"surprise calibration {self.calibration_version!r} does not match "
                f"manifest {manifest.surprise_calibration_version!r}"
            )
        if self.latent_dim != manifest.latent_dim:
            mismatches.append(
                f"surprise latent_dim {self.latent_dim} does not match "
                f"model {manifest.latent_dim}"
            )
        if mismatches:
            raise BundleCompatibilityError(mismatches)


@dataclass(frozen=True, slots=True)
class SurpriseEvent:
    status: Literal["unarmed", "normal", "breach", "alarm", "alarm_latched"]
    discrepancy: float | None
    threshold: float
    consecutive_breaches: int
    stop_commanded: bool
    bundle_id: str
    plan_locked_at_utc: str | None = None


def latent_discrepancy(
    predicted: Sequence[float], observed: Sequence[float], *, metric: SurpriseMetric
) -> float:
    """Compute a scale-readable error without requiring NumPy."""

    left = as_latent(predicted)
    right = as_latent(observed)
    if len(left) != len(right):
        raise ValueError(f"latent dimension mismatch: {len(left)} != {len(right)}")
    if metric == "rmse":
        return sqrt(
            sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left)
        )
    if metric == "cosine":
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if left_norm == 0.0 and right_norm == 0.0:
            return 0.0
        if left_norm == 0.0 or right_norm == 0.0:
            return 1.0
        similarity = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
        return 1.0 - similarity
    raise ValueError(f"unsupported surprise metric: {metric!r}")


def calibrate_surprise(
    normal_pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    bundle_id: str,
    calibration_version: str,
    metric: SurpriseMetric = "rmse",
    quantile: float = 0.99,
    consecutive_breaches: int = 1,
) -> SurpriseCalibration:
    """Fit a threshold from held-out normal prediction/observation pairs.

    The caller is responsible for including ordinary pushes and command changes
    and for excluding the controlled behavior-change/anomaly test set.
    """

    if not normal_pairs:
        raise ValueError("normal calibration pairs must not be empty")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    first_dim = len(normal_pairs[0][0])
    if first_dim <= 0:
        raise ValueError("calibration latents must not be empty")
    residuals: list[float] = []
    for predicted, observed in normal_pairs:
        if len(predicted) != first_dim or len(observed) != first_dim:
            raise ValueError("all calibration latents must have the same dimension")
        residuals.append(latent_discrepancy(predicted, observed, metric=metric))
    residuals.sort()
    # Nearest-rank is conservative and reproducible for a small calibration set.
    index = max(0, int((len(residuals) * quantile + 0.999999999) // 1) - 1)
    index = min(index, len(residuals) - 1)
    return SurpriseCalibration(
        calibration_version=calibration_version,
        bundle_id=bundle_id,
        latent_dim=first_dim,
        metric=metric,
        threshold=residuals[index],
        consecutive_breaches=consecutive_breaches,
        normal_sample_count=len(residuals),
        quantile=quantile,
    )


class SurpriseStopGuard:
    """Compare a locked next-step prediction and stop on calibrated surprise.

    A prediction is single-use: each observation consumes it.  Once an alarm is
    raised it remains latched until an explicit ``reset_alarm`` call, preventing a
    later normal frame from silently re-enabling motion.
    """

    def __init__(
        self,
        calibration: SurpriseCalibration,
        *,
        stop_callback: Callable[[str], None],
    ) -> None:
        self.calibration = calibration
        self._stop_callback = stop_callback
        self._predicted_next: Latent | None = None
        self._plan_locked_at_utc: str | None = None
        self._breach_count = 0
        self._alarm_latched = False

    @property
    def alarm_latched(self) -> bool:
        return self._alarm_latched

    def arm(
        self,
        predicted_next: Sequence[float],
        *,
        bundle_id: str,
        plan_locked_at_utc: str | None = None,
    ) -> None:
        if self._alarm_latched:
            raise RuntimeError("cannot arm surprise monitor while alarm is latched")
        if bundle_id != self.calibration.bundle_id:
            raise BundleCompatibilityError(
                [
                    f"prediction bundle_id {bundle_id!r} does not match surprise "
                    f"calibration {self.calibration.bundle_id!r}"
                ]
            )
        self._predicted_next = as_latent(
            predicted_next, expected_dim=self.calibration.latent_dim
        )
        self._plan_locked_at_utc = plan_locked_at_utc

    def observe(self, observed: Sequence[float]) -> SurpriseEvent:
        if self._alarm_latched:
            return self._event("alarm_latched", None, stop_commanded=False)
        if self._predicted_next is None:
            return self._event("unarmed", None, stop_commanded=False)

        observed_latent = as_latent(observed, expected_dim=self.calibration.latent_dim)
        discrepancy = latent_discrepancy(
            self._predicted_next, observed_latent, metric=self.calibration.metric
        )
        self._predicted_next = None
        if discrepancy > self.calibration.threshold:
            self._breach_count += 1
            if self._breach_count >= self.calibration.consecutive_breaches:
                self._alarm_latched = True
                reason = (
                    f"latent surprise {discrepancy:.6g} exceeded calibrated "
                    f"threshold {self.calibration.threshold:.6g}"
                )
                self._stop_callback(reason)
                return self._event("alarm", discrepancy, stop_commanded=True)
            return self._event("breach", discrepancy, stop_commanded=False)

        self._breach_count = 0
        return self._event("normal", discrepancy, stop_commanded=False)

    def reset_alarm(self) -> None:
        """Explicitly clear the latch; runtime policy must authorize this call."""

        self._alarm_latched = False
        self._breach_count = 0
        self._predicted_next = None
        self._plan_locked_at_utc = None

    def _event(
        self,
        status: Literal["unarmed", "normal", "breach", "alarm", "alarm_latched"],
        discrepancy: float | None,
        *,
        stop_commanded: bool,
    ) -> SurpriseEvent:
        return SurpriseEvent(
            status=status,
            discrepancy=discrepancy,
            threshold=self.calibration.threshold,
            consecutive_breaches=self._breach_count,
            stop_commanded=stop_commanded,
            bundle_id=self.calibration.bundle_id,
            plan_locked_at_utc=self._plan_locked_at_utc,
        )
