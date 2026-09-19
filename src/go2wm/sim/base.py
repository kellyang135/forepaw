"""Boundary implemented by real and fake simulator adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from go2wm.contracts import (
    ActionCommand,
    BlockTransition,
    CameraConfig,
    ResetReport,
    ResetRequest,
    RGBObservation,
    SceneConfig,
    StateLabels,
)


@runtime_checkable
class SimulatorAdapter(Protocol):
    """Minimal simulator interface used by collection and evaluation.

    Implementations own physics stepping and must aggregate every physics
    substep into ``BlockTransition.physics_samples``.  This prevents the caller
    from accidentally sampling contacts only at a block boundary.
    """

    @property
    def block_duration_s(self) -> float: ...

    @property
    def physics_dt_s(self) -> float: ...

    @property
    def camera_config(self) -> CameraConfig: ...

    @property
    def scene_config(self) -> SceneConfig: ...

    def reset(self, request: ResetRequest) -> ResetReport: ...

    def observe(self) -> RGBObservation: ...

    def labels(self) -> StateLabels: ...

    def execute_block(self, action: ActionCommand) -> BlockTransition: ...

