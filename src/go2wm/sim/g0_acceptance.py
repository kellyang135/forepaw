"""Validated, measurement-backed G0 acceptance configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


class G0AcceptanceError(ValueError):
    """Raised when the frozen G0 criteria are missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class G0Acceptance:
    source_path: Path
    status: str
    coowner_acknowledged: bool
    controller_id: str
    policy_sha256: str
    robot_xml_sha256: str
    min_forward_mps: float
    max_forward_mps: float
    max_abs_yaw_rps: float
    movable_min_displacement_m: float
    resistant_max_displacement_m: float
    required_successes_of_five: int
    movable_mass_kg: float
    resistant_mass_kg: float
    box_half_extent_m: float
    box_friction: float
    arena_width_m: float
    arena_height_m: float
    camera_id: str
    image_width_px: int
    image_height_px: int
    camera_fovy_deg: float
    goal_center_margin_m: float
    goal_radius_m: float
    settle_min_s: float
    settle_max_s: float
    settle_window_s: float
    settle_speed_tolerance_mps: float
    settle_yaw_rate_tolerance_rps: float
    settle_heading_tolerance_rad: float
    required_settled_of_ten: int


def _table(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise G0AcceptanceError(f"missing [{name}] table")
    return value


def _number(table: dict[str, Any], name: str, *, allow_zero: bool = False) -> float:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G0AcceptanceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise G0AcceptanceError(f"{name} must be finite and {qualifier}")
    return result


def _integer(table: dict[str, Any], name: str) -> int:
    value = table.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise G0AcceptanceError(f"{name} must be a positive integer")
    return value


def _string(table: dict[str, Any], name: str) -> str:
    value = table.get(name)
    if not isinstance(value, str) or not value.strip():
        raise G0AcceptanceError(f"{name} must be a non-empty string")
    return value


def load_g0_acceptance(path: str | Path) -> G0Acceptance:
    source = Path(path)
    with source.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema_version") != 1:
        raise G0AcceptanceError("unsupported G0 acceptance schema")
    status = _string(payload, "status")
    if status not in {"FROZEN_PENDING_COOWNER_ACK", "FROZEN"}:
        raise G0AcceptanceError("G0 acceptance status is not frozen")
    coowner = payload.get("coowner_acknowledged")
    if not isinstance(coowner, bool):
        raise G0AcceptanceError("coowner_acknowledged must be boolean")
    expected_status = "FROZEN" if coowner else "FROZEN_PENDING_COOWNER_ACK"
    if status != expected_status:
        raise G0AcceptanceError(
            f"status must be {expected_status!r} when coowner_acknowledged={coowner}"
        )
    action = _table(payload, "action")
    boxes = _table(payload, "boxes")
    scene = _table(payload, "scene")
    reset = _table(payload, "reset")
    result = G0Acceptance(
        source_path=source,
        status=status,
        coowner_acknowledged=coowner,
        controller_id=_string(payload, "controller_id"),
        policy_sha256=_string(payload, "policy_sha256"),
        robot_xml_sha256=_string(payload, "robot_xml_sha256"),
        min_forward_mps=_number(action, "min_forward_velocity_mps", allow_zero=True),
        max_forward_mps=_number(action, "max_forward_velocity_mps"),
        max_abs_yaw_rps=_number(action, "max_abs_yaw_rate_rps"),
        movable_min_displacement_m=_number(boxes, "movable_min_displacement_m"),
        resistant_max_displacement_m=_number(boxes, "resistant_max_displacement_m"),
        required_successes_of_five=_integer(boxes, "required_successes_of_five"),
        movable_mass_kg=_number(boxes, "movable_mass_kg"),
        resistant_mass_kg=_number(boxes, "resistant_mass_kg"),
        box_half_extent_m=_number(boxes, "half_extent_m"),
        box_friction=_number(boxes, "friction"),
        arena_width_m=_number(scene, "arena_width_m"),
        arena_height_m=_number(scene, "arena_height_m"),
        camera_id=_string(scene, "camera_id"),
        image_width_px=_integer(scene, "image_width_px"),
        image_height_px=_integer(scene, "image_height_px"),
        camera_fovy_deg=_number(scene, "camera_fovy_deg"),
        goal_center_margin_m=_number(scene, "goal_center_margin_m"),
        goal_radius_m=_number(scene, "goal_radius_m"),
        settle_min_s=_number(reset, "settle_min_s"),
        settle_max_s=_number(reset, "settle_max_s"),
        settle_window_s=_number(reset, "quiet_window_s"),
        settle_speed_tolerance_mps=_number(reset, "speed_tolerance_mps"),
        settle_yaw_rate_tolerance_rps=_number(reset, "yaw_rate_tolerance_rps"),
        settle_heading_tolerance_rad=_number(reset, "heading_tolerance_rad"),
        required_settled_of_ten=_integer(reset, "required_settled_of_ten"),
    )
    if result.min_forward_mps > result.max_forward_mps:
        raise G0AcceptanceError("forward bounds are inverted")
    if result.required_successes_of_five > 5:
        raise G0AcceptanceError("required_successes_of_five cannot exceed five")
    if result.required_settled_of_ten > 10:
        raise G0AcceptanceError("required_settled_of_ten cannot exceed ten")
    if result.settle_min_s > result.settle_max_s:
        raise G0AcceptanceError("settle bounds are inverted")
    if result.movable_mass_kg >= result.resistant_mass_kg:
        raise G0AcceptanceError("movable mass must be below resistant mass")
    return result
