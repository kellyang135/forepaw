"""Receding-horizon planner over versioned world-model predictions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from go2wm.model import (
    ActionBlock,
    CandidateRollout,
    ModelBundle,
    ModelInput,
    Point2D,
)

from .candidates import CandidateSequence, build_candidate_library
from .scoring import RolloutScorer, ScoreBreakdown


class PlanningError(RuntimeError):
    """Raised when predictions cannot safely be associated with candidates."""


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    model_input: ModelInput
    goal_xy: Point2D
    current_robot_xy: Point2D
    previous_action: ActionBlock | None = None


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    rank: int
    rollout: CandidateRollout
    score: ScoreBreakdown


@dataclass(frozen=True, slots=True)
class PlanResult:
    """Immutable prediction snapshot produced before any command is executed."""

    status: str
    selection_reason: str
    bundle_id: str
    goal_xy: Point2D
    locked_at_utc: str
    selected: RankedCandidate
    rankings: tuple[RankedCandidate, ...]

    @property
    def first_action(self) -> ActionBlock:
        """The only action authorized for execution before replanning."""

        return self.selected.rollout.actions[0]

    @property
    def predicted_next_latent(self) -> tuple[float, ...] | None:
        return self.selected.rollout.states[0].latent


@dataclass(frozen=True, slots=True)
class PlannerSafetyPolicy:
    """Optional deterministic fallback when every moving plan looks dangerous."""

    max_acceptable_failure_risk: float = 0.65
    stop_candidate_id: str = "stop_hold"

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_acceptable_failure_risk <= 1.0:
            raise ValueError("max_acceptable_failure_risk must be in [0, 1]")


class RecedingHorizonPlanner:
    """Predict all candidates, score them, and authorize one 0.5-second block."""

    def __init__(
        self,
        model_bundle: ModelBundle,
        *,
        candidates: Iterable[CandidateSequence] | None = None,
        scorer: RolloutScorer | None = None,
        safety_policy: PlannerSafetyPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.model_bundle = model_bundle
        self.candidates = tuple(candidates or build_candidate_library())
        self.scorer = scorer or RolloutScorer()
        self.safety_policy = safety_policy or PlannerSafetyPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._validate_candidates()

    def plan(self, request: PlanningRequest) -> PlanResult:
        """Return locked predictions and the first action to execute.

        Calling this method does not move the robot.  Runtime integration should
        persist/display the result first, then execute ``first_action``, collect a
        fresh observation, and invoke the planner again.
        """

        rollouts = self.model_bundle.predict_candidates(request.model_input, self.candidates)
        by_id = self._validate_and_index_rollouts(rollouts)
        scored = [
            (
                by_id[candidate.candidate_id],
                self.scorer.score(
                    by_id[candidate.candidate_id],
                    goal_xy=request.goal_xy,
                    initial_robot_xy=request.current_robot_xy,
                    previous_action=request.previous_action,
                ),
            )
            for candidate in self.candidates
        ]
        scored.sort(key=lambda item: (item[1].total, item[0].candidate_id))
        rankings = tuple(
            RankedCandidate(rank=index, rollout=rollout, score=score)
            for index, (rollout, score) in enumerate(scored, start=1)
        )

        selected = rankings[0]
        reason = "minimum_score"
        moving = [item for item in rankings if item.rollout.family != "stop"]
        if moving and all(
            item.score.maximum_failure_risk
            > self.safety_policy.max_acceptable_failure_risk
            for item in moving
        ):
            selected = next(
                (
                    item
                    for item in rankings
                    if item.rollout.candidate_id == self.safety_policy.stop_candidate_id
                ),
                selected,
            )
            reason = "all_moving_candidates_exceed_risk_limit"

        return PlanResult(
            status="planned",
            selection_reason=reason,
            bundle_id=self.model_bundle.manifest.bundle_id,
            goal_xy=request.goal_xy,
            locked_at_utc=self._clock().astimezone(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            selected=selected,
            rankings=rankings,
        )

    def _validate_candidates(self) -> None:
        if not self.candidates:
            raise ValueError("planner requires at least one candidate")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate IDs must be unique")
        manifest = self.model_bundle.manifest
        for candidate in self.candidates:
            if len(candidate.actions) != manifest.horizon_blocks:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} has {len(candidate.actions)} "
                    f"blocks; bundle expects {manifest.horizon_blocks}"
                )
            if any(
                abs(action.duration_s - manifest.block_duration_s) > 1e-9
                for action in candidate.actions
            ):
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} has incompatible timing"
                )

    def _validate_and_index_rollouts(
        self, rollouts: tuple[CandidateRollout, ...]
    ) -> dict[str, CandidateRollout]:
        expected = {candidate.candidate_id: candidate for candidate in self.candidates}
        actual_ids = [rollout.candidate_id for rollout in rollouts]
        duplicates = sorted(
            candidate_id
            for candidate_id in set(actual_ids)
            if actual_ids.count(candidate_id) > 1
        )
        missing = sorted(set(expected) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected))
        if duplicates or missing or unexpected:
            raise PlanningError(
                "prediction/candidate mismatch: "
                f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
            )
        by_id = {rollout.candidate_id: rollout for rollout in rollouts}
        for candidate_id, candidate in expected.items():
            rollout = by_id[candidate_id]
            if rollout.family != candidate.family:
                raise PlanningError(f"family mismatch for {candidate_id!r}")
            if rollout.actions != candidate.actions:
                raise PlanningError(f"action mismatch for {candidate_id!r}")
        return by_id
