"""World-model contracts and versioned artifact bundles."""

from .bundle import (
    CURRENT_BUNDLE_SCHEMA,
    BundleCompatibilityError,
    BundleManifest,
    ModelBundle,
    RuntimeRequirements,
    WorldModelBackend,
)
from .components import (
    ActionConditionedPredictor,
    CandidateLike,
    ComponentWorldModelBackend,
    LatentReadout,
    ObservationEncoder,
)
from .types import (
    ActionBlock,
    CandidateRollout,
    Latent,
    ModelInput,
    ObjectState,
    Point2D,
    PredictedState,
    RobotState,
    as_latent,
)

__all__ = [
    "CURRENT_BUNDLE_SCHEMA",
    "ActionBlock",
    "ActionConditionedPredictor",
    "BundleCompatibilityError",
    "BundleManifest",
    "CandidateLike",
    "CandidateRollout",
    "ComponentWorldModelBackend",
    "Latent",
    "LatentReadout",
    "ModelBundle",
    "ModelInput",
    "ObjectState",
    "ObservationEncoder",
    "Point2D",
    "PredictedState",
    "RobotState",
    "RuntimeRequirements",
    "WorldModelBackend",
    "as_latent",
]
