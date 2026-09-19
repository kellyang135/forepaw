"""Versioned interfaces for a deployable visual world-model bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from .types import CandidateRollout, ModelInput

CURRENT_BUNDLE_SCHEMA = 1


class BundleCompatibilityError(RuntimeError):
    """Raised when model artifacts do not match the runtime contract."""

    def __init__(self, mismatches: Sequence[str]) -> None:
        self.mismatches = tuple(mismatches)
        super().__init__("incompatible model bundle: " + "; ".join(self.mismatches))


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """Identity and tensor/timing contract for one inseparable artifact bundle."""

    bundle_id: str
    encoder_version: str
    predictor_version: str
    readout_version: str
    normalization_version: str
    surprise_calibration_version: str
    latent_dim: int = 192
    action_dim: int = 2
    block_duration_s: float = 0.5
    history_frames: int = 3
    horizon_blocks: int = 6
    schema_version: int = CURRENT_BUNDLE_SCHEMA
    created_at_utc: str = ""

    def __post_init__(self) -> None:
        required = {
            "bundle_id": self.bundle_id,
            "encoder_version": self.encoder_version,
            "predictor_version": self.predictor_version,
            "readout_version": self.readout_version,
            "normalization_version": self.normalization_version,
            "surprise_calibration_version": self.surprise_calibration_version,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError("manifest fields must not be empty: " + ", ".join(missing))
        if self.latent_dim <= 0 or self.action_dim <= 0:
            raise ValueError("latent_dim and action_dim must be positive")
        if self.block_duration_s <= 0:
            raise ValueError("block_duration_s must be positive")
        if self.history_frames < 1 or self.horizon_blocks < 1:
            raise ValueError("history_frames and horizon_blocks must be positive")

    @classmethod
    def create(cls, **values: Any) -> BundleManifest:
        """Create a manifest with an auditable UTC timestamp."""

        values.setdefault(
            "created_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        return cls(**values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> BundleManifest:
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("unknown manifest fields: " + ", ".join(unknown))
        return cls(**dict(values))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeRequirements:
    """Expected contract at the planning/deployment boundary."""

    latent_dim: int = 192
    action_dim: int = 2
    block_duration_s: float = 0.5
    history_frames: int = 3
    horizon_blocks: int = 6
    schema_version: int = CURRENT_BUNDLE_SCHEMA
    required_bundle_id: str | None = None

    def validate(self, manifest: BundleManifest) -> None:
        mismatches: list[str] = []
        exact_fields = (
            "latent_dim",
            "action_dim",
            "history_frames",
            "horizon_blocks",
            "schema_version",
        )
        for name in exact_fields:
            expected = getattr(self, name)
            actual = getattr(manifest, name)
            if actual != expected:
                mismatches.append(f"{name}: expected {expected!r}, got {actual!r}")
        if abs(manifest.block_duration_s - self.block_duration_s) > 1e-9:
            mismatches.append(
                "block_duration_s: expected "
                f"{self.block_duration_s!r}, got {manifest.block_duration_s!r}"
            )
        if self.required_bundle_id and manifest.bundle_id != self.required_bundle_id:
            mismatches.append(
                f"bundle_id: expected {self.required_bundle_id!r}, "
                f"got {manifest.bundle_id!r}"
            )
        if mismatches:
            raise BundleCompatibilityError(mismatches)


@runtime_checkable
class WorldModelBackend(Protocol):
    """Adapter boundary to LeWM/PyTorch or a deterministic test double."""

    @property
    def manifest(self) -> BundleManifest:
        """Return the manifest for the weights currently loaded in memory."""

    def predict_candidates(
        self,
        model_input: ModelInput,
        candidates: Sequence[Any],
    ) -> Sequence[CandidateRollout]:
        """Predict one rollout per candidate without consulting simulator state."""


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Validated handle used by the planner.

    Concrete backends own preprocessing, encoding, autoregression, and readouts.
    They must emit plain ``CandidateRollout`` values at this boundary.
    """

    backend: WorldModelBackend
    requirements: RuntimeRequirements = RuntimeRequirements()

    def __post_init__(self) -> None:
        self.requirements.validate(self.backend.manifest)

    @property
    def manifest(self) -> BundleManifest:
        return self.backend.manifest

    def predict_candidates(
        self, model_input: ModelInput, candidates: Sequence[Any]
    ) -> tuple[CandidateRollout, ...]:
        model_input.validate(
            history_frames=self.manifest.history_frames,
            block_duration_s=self.manifest.block_duration_s,
        )
        rollouts = tuple(self.backend.predict_candidates(model_input, candidates))
        for rollout in rollouts:
            if rollout.bundle_id != self.manifest.bundle_id:
                raise BundleCompatibilityError(
                    [
                        "backend emitted rollout bundle_id "
                        f"{rollout.bundle_id!r}, expected {self.manifest.bundle_id!r}"
                    ]
                )
            for action in rollout.actions:
                if abs(action.duration_s - self.manifest.block_duration_s) > 1e-9:
                    raise BundleCompatibilityError(
                        ["backend emitted an action with an incompatible duration"]
                    )
        return rollouts
