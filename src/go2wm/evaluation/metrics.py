"""Pure metric functions used by validation and untouched final evaluation.

The functions intentionally accept already-aligned scalar measurements.  Image,
tensor, and simulator handling belongs outside this module, making the metric
implementation easy to test and difficult to silently change between methods.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for finite scalar values."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class ErrorSummary:
    count: int
    mean: float
    median: float
    p95: float
    maximum: float


def error_summary(errors: Sequence[float]) -> ErrorSummary:
    if not errors:
        raise ValueError("at least one error is required")
    normalized = tuple(float(value) for value in errors)
    if any(not math.isfinite(value) or value < 0 for value in normalized):
        raise ValueError("errors must be finite and non-negative")
    return ErrorSummary(
        count=len(normalized),
        mean=sum(normalized) / len(normalized),
        median=percentile(normalized, 0.5),
        p95=percentile(normalized, 0.95),
        maximum=max(normalized),
    )


def action_conditioning_improvement(*, matched_error: float, shuffled_error: float) -> float:
    """Fractional error reduction of matched actions relative to shuffled actions."""

    if matched_error < 0 or shuffled_error <= 0:
        raise ValueError("matched error must be non-negative and shuffled error positive")
    if not math.isfinite(matched_error) or not math.isfinite(shuffled_error):
        raise ValueError("errors must be finite")
    return (shuffled_error - matched_error) / shuffled_error


def ranking_success_fraction(useful_alternative_ranked_higher: Iterable[bool]) -> float:
    outcomes = tuple(useful_alternative_ranked_higher)
    if not outcomes:
        raise ValueError("at least one ranking comparison is required")
    return sum(outcomes) / len(outcomes)


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    scenario_id: str
    behavior: str
    repeat_index: int
    success: bool
    elapsed_sim_s: float
    fell: bool
    stalled: bool
    excluded: bool = False
    exclusion_reason: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        if self.behavior not in {"push", "detour"}:
            raise ValueError("behavior must be push or detour")
        if self.repeat_index < 1:
            raise ValueError("repeat_index must be one-based")
        if self.elapsed_sim_s < 0 or not math.isfinite(self.elapsed_sim_s):
            raise ValueError("elapsed_sim_s must be finite and non-negative")
        if self.excluded and not self.exclusion_reason:
            raise ValueError("excluded outcomes require an exclusion reason")
        if not self.excluded and self.exclusion_reason:
            raise ValueError("non-excluded outcomes cannot have an exclusion reason")


@dataclass(frozen=True, slots=True)
class TaskSummary:
    included_count: int
    excluded_count: int
    success_count: int
    success_fraction: float
    fall_count: int
    stall_count: int
    median_success_time_s: float | None


def summarize_tasks(outcomes: Sequence[TaskOutcome]) -> TaskSummary:
    included = tuple(outcome for outcome in outcomes if not outcome.excluded)
    if not included:
        raise ValueError("at least one non-excluded outcome is required")
    successful_times = tuple(outcome.elapsed_sim_s for outcome in included if outcome.success)
    return TaskSummary(
        included_count=len(included),
        excluded_count=len(outcomes) - len(included),
        success_count=sum(outcome.success for outcome in included),
        success_fraction=sum(outcome.success for outcome in included) / len(included),
        fall_count=sum(outcome.fell for outcome in included),
        stall_count=sum(outcome.stalled for outcome in included),
        median_success_time_s=(
            percentile(successful_times, 0.5) if successful_times else None
        ),
    )


@dataclass(frozen=True, slots=True)
class SurpriseSummary:
    normal_blocks: int
    false_stops: int
    false_stop_fraction: float
    changed_trials: int
    detected_trials: int
    detection_fraction: float
    median_detection_delay_blocks: float | None


def summarize_surprise(
    *,
    normal_stop_flags: Sequence[bool],
    changed_detection_delays: Sequence[int | None],
) -> SurpriseSummary:
    if not normal_stop_flags:
        raise ValueError("normal operation must contain at least one block")
    if not changed_detection_delays:
        raise ValueError("at least one changed-behavior trial is required")
    if any(delay is not None and delay < 0 for delay in changed_detection_delays):
        raise ValueError("detection delays must be non-negative block counts")
    detected = tuple(delay for delay in changed_detection_delays if delay is not None)
    false_stops = sum(normal_stop_flags)
    return SurpriseSummary(
        normal_blocks=len(normal_stop_flags),
        false_stops=false_stops,
        false_stop_fraction=false_stops / len(normal_stop_flags),
        changed_trials=len(changed_detection_delays),
        detected_trials=len(detected),
        detection_fraction=len(detected) / len(changed_detection_delays),
        median_detection_delay_blocks=percentile(detected, 0.5) if detected else None,
    )

