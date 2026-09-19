"""Training-only color-conditioned mobility lookup for the simple baseline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InteractionExample:
    appearance_class: str
    commanded_progress_m: float
    object_displacement_m: float


@dataclass(frozen=True, slots=True)
class MobilityLookup:
    displacement_ratio_by_appearance: dict[str, float]

    @classmethod
    def fit(
        cls,
        examples: list[InteractionExample],
        *,
        minimum_commanded_progress_m: float = 0.02,
    ) -> MobilityLookup:
        ratios: dict[str, list[float]] = {}
        for example in examples:
            if not example.appearance_class:
                raise ValueError("appearance_class must not be empty")
            if example.commanded_progress_m < 0 or example.object_displacement_m < 0:
                raise ValueError("interaction distances must be non-negative")
            if example.commanded_progress_m < minimum_commanded_progress_m:
                continue
            ratio = min(example.object_displacement_m / example.commanded_progress_m, 1.0)
            ratios.setdefault(example.appearance_class, []).append(ratio)
        if not ratios:
            raise ValueError("no usable interaction examples")
        return cls(
            {
                appearance: sum(values) / len(values)
                for appearance, values in sorted(ratios.items())
            }
        )

    def expected_displacement(self, appearance_class: str, commanded_progress_m: float) -> float:
        try:
            ratio = self.displacement_ratio_by_appearance[appearance_class]
        except KeyError as error:
            raise KeyError(
                f"appearance class {appearance_class!r} was not observed in training"
            ) from error
        return max(commanded_progress_m, 0.0) * ratio
