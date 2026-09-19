from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from go2wm.sim.controller_handoff import (
    EXPECTED_COMMAND_ORDER,
    EXPECTED_JOINT_ORDER,
    PINNED_MODEL_XML_SHA256,
    PINNED_REPOSITORY,
    PINNED_REVISION,
    PINNED_TASK,
    REQUIRED_FILES,
    SCHEMA_VERSION,
    ControllerHandoffError,
    verify_controller_handoff,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    packet = tmp_path / "packet"
    packet.mkdir()
    files: dict[str, dict[str, str]] = {}
    for name in sorted(REQUIRED_FILES):
        path = packet / f"{name}.artifact"
        if name == "robot_xml":
            # The verifier pins the actual robot XML bytes, not only a manifest claim.
            path.write_bytes(b"test robot XML")
        else:
            path.write_text(f"fixture for {name}\n", encoding="utf-8")
        files[name] = {"path": path.name, "sha256": _sha256(path)}
    policy_data = packet / "policy_onnx_data.artifact"
    policy_data.write_text("fixture for policy_onnx_data\n", encoding="utf-8")
    files["policy_onnx_data"] = {
        "path": policy_data.name,
        "sha256": _sha256(policy_data),
    }

    # Unit tests cannot reconstruct the upstream XML preimage. Temporarily use the
    # fixture digest as the pinned value at each call site that needs a passing packet.
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "controller_id": "go2-flat-seed42",
        "source": {
            "repository": PINNED_REPOSITORY,
            "revision": PINNED_REVISION,
            "task": PINNED_TASK,
        },
        "compatibility": {
            "model_xml_sha256": files["robot_xml"]["sha256"],
            "joint_order": list(EXPECTED_JOINT_ORDER),
            "command_order": list(EXPECTED_COMMAND_ORDER),
            "action_semantics": "scaled_joint_position_target",
            "onnx_data_mode": "external",
            "physics_dt_s": 0.005,
            "controller_dt_s": 0.02,
            "physics_substeps_per_control": 4,
        },
        "validation": {
            "training_smoke_passed": True,
            "checkpoint_playback_passed": True,
            "onnx_sim_playback_passed": True,
            "onnx_export_present": True,
        },
        "files": files,
    }
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return packet, manifest


def _write_manifest(packet: Path, manifest: dict[str, Any]) -> None:
    (packet / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _set_path(manifest: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = manifest
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def test_valid_controller_packet_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)

    summary = verify_controller_handoff(packet)

    assert summary.controller_id == "go2-flat-seed42"
    assert summary.file_count == len(REQUIRED_FILES) + 1
    assert summary.physics_dt_s == 0.005
    assert summary.controller_dt_s == 0.02


def test_tampered_policy_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    (packet / "policy_onnx.artifact").write_text("tampered", encoding="utf-8")

    with pytest.raises(ControllerHandoffError, match="checksum mismatch"):
        verify_controller_handoff(packet)


def test_manifest_cannot_claim_a_different_robot_xml(tmp_path: Path) -> None:
    packet, _ = _packet(tmp_path)

    with pytest.raises(ControllerHandoffError, match="pinned Go2 XML"):
        verify_controller_handoff(packet)


def test_path_escape_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    outside = tmp_path / "outside.onnx"
    outside.write_text("outside", encoding="utf-8")
    manifest["files"]["policy_onnx"] = {
        "path": "../outside.onnx",
        "sha256": _sha256(outside),
    }
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match="escapes"):
        verify_controller_handoff(packet)


def test_inconsistent_control_timing_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    manifest["compatibility"]["physics_substeps_per_control"] = 3
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match="timing are inconsistent"):
        verify_controller_handoff(packet)


def test_reference_digest_is_not_accidentally_changed() -> None:
    assert PINNED_MODEL_XML_SHA256 == (
        "077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912"
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", "unknown", "unsupported"),
        ("controller_id", "../unsafe", "unsafe"),
        ("source.repository", "https://example.invalid", "repository"),
        ("source.revision", "0" * 40, "revision"),
        ("source.task", "Unitree-Go1-Flat", "task"),
        ("compatibility.joint_order", [], "joint order"),
        ("compatibility.command_order", [], "command order"),
        ("compatibility.action_semantics", "torque", "action semantics"),
        ("compatibility.onnx_data_mode", "unknown", "onnx_data_mode"),
        ("compatibility.physics_dt_s", True, "must be numeric"),
        ("compatibility.controller_dt_s", float("nan"), "finite and positive"),
        ("compatibility.physics_substeps_per_control", 0, "must be positive"),
        ("validation.training_smoke_passed", False, "must be true"),
        ("validation.checkpoint_playback_passed", False, "must be true"),
        ("validation.onnx_sim_playback_passed", False, "must be true"),
        ("validation.onnx_export_present", False, "must be true"),
        ("files.policy_onnx.path", "", "path must be non-empty"),
        ("files.policy_onnx.sha256", "not-a-digest", "sha256 is invalid"),
    ],
)
def test_manifest_contract_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: Any,
    message: str,
) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    _set_path(manifest, path, value)
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match=message):
        verify_controller_handoff(packet)


def test_missing_required_file_entry_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    del manifest["files"]["checkpoint"]
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match="missing required files"):
        verify_controller_handoff(packet)


def test_external_onnx_data_is_required_when_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    del manifest["files"]["policy_onnx_data"]
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match="external ONNX data file is missing"):
        verify_controller_handoff(packet)


def test_self_contained_onnx_packet_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    manifest["compatibility"]["onnx_data_mode"] = "self_contained"
    del manifest["files"]["policy_onnx_data"]
    _write_manifest(packet, manifest)

    summary = verify_controller_handoff(packet)

    assert summary.file_count == len(REQUIRED_FILES)


def test_self_contained_packet_cannot_declare_external_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    manifest["compatibility"]["onnx_data_mode"] = "self_contained"
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match="declares external data"):
        verify_controller_handoff(packet)


def test_empty_artifact_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet, manifest = _packet(tmp_path)
    robot_digest = manifest["files"]["robot_xml"]["sha256"]
    monkeypatch.setattr("go2wm.sim.controller_handoff.PINNED_MODEL_XML_SHA256", robot_digest)
    empty = packet / "empty.pt"
    empty.touch()
    manifest["files"]["checkpoint"] = {
        "path": empty.name,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match="is empty"):
        verify_controller_handoff(packet)


def test_unreadable_manifest_is_rejected(tmp_path: Path) -> None:
    packet = tmp_path / "bad-packet"
    packet.mkdir()
    (packet / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ControllerHandoffError, match="cannot read"):
        verify_controller_handoff(packet)


def test_non_object_section_is_rejected(tmp_path: Path) -> None:
    packet, manifest = _packet(tmp_path)
    manifest["source"] = []
    _write_manifest(packet, manifest)

    with pytest.raises(ControllerHandoffError, match="source must be an object"):
        verify_controller_handoff(packet)
