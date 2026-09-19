"""JSON-safe skill boundary for dimOS.

The core facade has no dimOS dependency, so its validation and stop behavior can
be tested in the lightweight environment. ``build_dimos_skill_module`` performs
the optional imports only when the real deployment environment is ready.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

JsonObject = dict[str, Any]


@runtime_checkable
class ControllerBackend(Protocol):
    """Backend used by both an in-process model and a local RPC client."""

    def imagine(self, candidate_commands: list[list[JsonObject]]) -> JsonObject: ...

    def plan_to(self, x_m: float, y_m: float) -> JsonObject: ...

    def stop(self, reason: str) -> JsonObject: ...


@dataclass(slots=True)
class WorldModelSkillFacade:
    backend: ControllerBackend

    def imagine(self, candidate_commands: list[list[JsonObject]]) -> str:
        """Predict candidate futures without moving the robot."""

        _validate_candidate_commands(candidate_commands)
        return _encode_result(self.backend.imagine(candidate_commands))

    def plan_to(self, x: float, y: float) -> str:
        """Plan toward a world-frame goal and execute only the first action block."""

        _require_finite("x", x)
        _require_finite("y", y)
        return _encode_result(self.backend.plan_to(float(x), float(y)))

    def stop(self) -> str:
        """Cancel planning and command the single motion owner to stop."""

        return _encode_result(self.backend.stop("skill_requested"))


def build_dimos_skill_module(backend: ControllerBackend) -> object:
    """Construct the real dimOS ``Module`` only when dimOS is installed.

    Current dimOS documentation specifies that skill parameters must be JSON-
    serializable primitives and that skill results may be strings. The facade
    deliberately returns JSON strings to keep this boundary explicit.
    """

    try:
        from dimos.agents.annotation import skill
        from dimos.core.module import Module
    except ImportError as error:  # pragma: no cover - exercised in a dimOS environment
        raise RuntimeError(
            "dimOS is not installed; use WorldModelSkillFacade for core tests"
        ) from error

    facade = WorldModelSkillFacade(backend)

    class Go2WorldModelSkills(Module):
        @skill
        def imagine(self, candidate_commands: list[list[dict]]) -> str:
            """Predict robot and box futures for candidate commands; do not move."""

            return facade.imagine(candidate_commands)

        @skill
        def plan_to(self, x: float, y: float) -> str:
            """Replan toward world-frame (x, y) and execute one 0.5-second block."""

            return facade.plan_to(x, y)

        @skill
        def stop(self) -> str:
            """Cancel the active plan and issue a zero motion command."""

            return facade.stop()

    return Go2WorldModelSkills()


def _validate_candidate_commands(candidates: list[list[JsonObject]]) -> None:
    if not candidates:
        raise ValueError("candidate_commands must not be empty")
    for candidate_index, candidate in enumerate(candidates):
        if not candidate:
            raise ValueError(f"candidate {candidate_index} must contain at least one block")
        for block_index, block in enumerate(candidate):
            if not isinstance(block, dict):
                raise TypeError(
                    f"candidate {candidate_index} block {block_index} must be an object"
                )
            unknown = set(block) - {"forward_velocity_mps", "yaw_rate_rps", "duration_s"}
            if unknown:
                raise ValueError(f"unknown action fields: {sorted(unknown)}")
            for field in ("forward_velocity_mps", "yaw_rate_rps"):
                if field not in block:
                    raise ValueError(f"action block is missing {field}")
                _require_finite(field, block[field])
            duration = block.get("duration_s", 0.5)
            _require_finite("duration_s", duration)
            if float(duration) != 0.5:
                raise ValueError("the initial deployment contract requires 0.5-second blocks")


def _require_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _encode_result(value: JsonObject) -> str:
    if not isinstance(value, dict):
        raise TypeError("controller backend results must be JSON objects")
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"controller result is not JSON serializable: {error}") from error

