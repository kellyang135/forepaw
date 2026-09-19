"""Simulator adapters and the deterministic test double."""

from .base import SimulatorAdapter
from .controller_handoff import (
    ControllerHandoffError,
    ControllerHandoffSummary,
    verify_controller_handoff,
)
from .fake import DeterministicFakeSimulator, FakeSimulatorConfig
from .training_host import (
    TrainingHostAssessment,
    TrainingHostFacts,
    assess_training_host,
    collect_training_host_facts,
    training_host_report,
)

__all__ = [
    "ControllerHandoffError",
    "ControllerHandoffSummary",
    "DeterministicFakeSimulator",
    "FakeSimulatorConfig",
    "SimulatorAdapter",
    "TrainingHostAssessment",
    "TrainingHostFacts",
    "assess_training_host",
    "collect_training_host_facts",
    "training_host_report",
    "verify_controller_handoff",
]
