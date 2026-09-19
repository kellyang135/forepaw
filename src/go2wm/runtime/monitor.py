"""Bridge immutable planning results to the single-step surprise monitor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from go2wm.planning import PlanResult

from .surprise import SurpriseEvent, SurpriseStopGuard


@dataclass(frozen=True, slots=True)
class ArmedPlan:
    bundle_id: str
    candidate_id: str
    locked_at_utc: str


class PlanExecutionMonitor:
    """Arm only after a plan is locked, then compare the next observation."""

    def __init__(self, surprise_guard: SurpriseStopGuard) -> None:
        self.surprise_guard = surprise_guard
        self._armed_plan: ArmedPlan | None = None

    @property
    def armed_plan(self) -> ArmedPlan | None:
        return self._armed_plan

    def arm_plan(self, plan: PlanResult) -> ArmedPlan:
        predicted = plan.predicted_next_latent
        if predicted is None:
            raise ValueError(
                "selected rollout has no next latent; surprise monitoring cannot be armed"
            )
        self.surprise_guard.arm(
            predicted,
            bundle_id=plan.bundle_id,
            plan_locked_at_utc=plan.locked_at_utc,
        )
        self._armed_plan = ArmedPlan(
            bundle_id=plan.bundle_id,
            candidate_id=plan.selected.rollout.candidate_id,
            locked_at_utc=plan.locked_at_utc,
        )
        return self._armed_plan

    def observe(self, observed_latent: Sequence[float]) -> SurpriseEvent:
        event = self.surprise_guard.observe(observed_latent)
        if event.status not in ("unarmed", "alarm_latched"):
            self._armed_plan = None
        return event
