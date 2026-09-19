"""Strict verification for a trained true-Go2 controller handoff."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "go2wm.go2-controller-handoff.v1"
PINNED_REPOSITORY = "https://github.com/unitreerobotics/unitree_rl_mjlab"
PINNED_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
PINNED_TASK = "Unitree-Go2-Flat"
PINNED_MODEL_XML_SHA256 = "077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912"
EXPECTED_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
EXPECTED_COMMAND_ORDER = (
    "forward_velocity_mps",
    "lateral_velocity_mps",
    "yaw_rate_rps",
)
REQUIRED_FILES = frozenset(
    {
        "policy_onnx",
        "checkpoint",
        "environment_config",
        "agent_config",
        "deployment_config",
        "robot_xml",
        "training_log",
    }
)


class ControllerHandoffError(ValueError):
    """Raised when a controller packet cannot be trusted or is incompatible."""


@dataclass(frozen=True, slots=True)
class ControllerHandoffSummary:
    controller_id: str
    source_revision: str
    task: str
    policy_path: Path
    checkpoint_path: Path
    physics_dt_s: float
    controller_dt_s: float
    file_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ControllerHandoffError(f"{name} must be an object")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ControllerHandoffError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ControllerHandoffError(f"{name} must be finite and positive")
    return result


def verify_controller_handoff(directory: str | Path) -> ControllerHandoffSummary:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControllerHandoffError(f"cannot read {manifest_path}: {error}") from error
    manifest = _mapping(raw, "manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ControllerHandoffError("unsupported controller handoff schema")
    controller_id = manifest.get("controller_id")
    if not isinstance(controller_id, str) or not re.fullmatch(
        r"[A-Za-z0-9._-]+", controller_id
    ):
        raise ControllerHandoffError("controller_id is missing or unsafe")

    source = _mapping(manifest.get("source"), "source")
    if source.get("repository") != PINNED_REPOSITORY:
        raise ControllerHandoffError("controller repository does not match the pinned source")
    if source.get("revision") != PINNED_REVISION:
        raise ControllerHandoffError("controller revision does not match the pinned source")
    if source.get("task") != PINNED_TASK:
        raise ControllerHandoffError("controller task must be Unitree-Go2-Flat")

    compatibility = _mapping(manifest.get("compatibility"), "compatibility")
    if compatibility.get("model_xml_sha256") != PINNED_MODEL_XML_SHA256:
        raise ControllerHandoffError("controller was not trained against the pinned Go2 XML")
    if tuple(compatibility.get("joint_order", ())) != EXPECTED_JOINT_ORDER:
        raise ControllerHandoffError("controller joint order is incompatible")
    if tuple(compatibility.get("command_order", ())) != EXPECTED_COMMAND_ORDER:
        raise ControllerHandoffError("controller command order is incompatible")
    if compatibility.get("action_semantics") != "scaled_joint_position_target":
        raise ControllerHandoffError("controller action semantics are incompatible")
    onnx_data_mode = compatibility.get("onnx_data_mode")
    if onnx_data_mode not in {"self_contained", "external"}:
        raise ControllerHandoffError(
            "onnx_data_mode must be either self_contained or external"
        )
    physics_dt_s = _finite_positive(compatibility.get("physics_dt_s"), "physics_dt_s")
    controller_dt_s = _finite_positive(
        compatibility.get("controller_dt_s"), "controller_dt_s"
    )
    substeps = compatibility.get("physics_substeps_per_control")
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps <= 0:
        raise ControllerHandoffError("physics_substeps_per_control must be positive")
    if not math.isclose(physics_dt_s * substeps, controller_dt_s, abs_tol=1e-12):
        raise ControllerHandoffError("controller and physics timing are inconsistent")

    validation = _mapping(manifest.get("validation"), "validation")
    for key in (
        "training_smoke_passed",
        "checkpoint_playback_passed",
        "onnx_sim_playback_passed",
        "onnx_export_present",
    ):
        if validation.get(key) is not True:
            raise ControllerHandoffError(f"validation.{key} must be true")

    files = _mapping(manifest.get("files"), "files")
    missing = REQUIRED_FILES - set(files)
    if missing:
        raise ControllerHandoffError(f"missing required files: {sorted(missing)}")
    has_external_data = "policy_onnx_data" in files
    if onnx_data_mode == "external" and not has_external_data:
        raise ControllerHandoffError("external ONNX data file is missing")
    if onnx_data_mode == "self_contained" and has_external_data:
        raise ControllerHandoffError("self-contained ONNX packet declares external data")
    resolved: dict[str, Path] = {}
    for name, description in files.items():
        item = _mapping(description, f"files.{name}")
        relative = item.get("path")
        expected_digest = item.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ControllerHandoffError(f"files.{name}.path must be non-empty")
        if not isinstance(expected_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_digest
        ):
            raise ControllerHandoffError(f"files.{name}.sha256 is invalid")
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ControllerHandoffError(f"files.{name} escapes the handoff directory")
        if path.is_symlink() or not path.is_file():
            raise ControllerHandoffError(f"files.{name} is missing or is not a regular file")
        if path.stat().st_size <= 0:
            raise ControllerHandoffError(f"files.{name} is empty")
        if _sha256(path) != expected_digest:
            raise ControllerHandoffError(f"files.{name} checksum mismatch")
        resolved[name] = path

    robot_digest = files["robot_xml"]["sha256"]
    if robot_digest != PINNED_MODEL_XML_SHA256:
        raise ControllerHandoffError("robot XML file does not match the pinned Go2 XML")

    return ControllerHandoffSummary(
        controller_id=controller_id,
        source_revision=PINNED_REVISION,
        task=PINNED_TASK,
        policy_path=resolved["policy_onnx"],
        checkpoint_path=resolved["checkpoint"],
        physics_dt_s=physics_dt_s,
        controller_dt_s=controller_dt_s,
        file_count=len(files),
    )
