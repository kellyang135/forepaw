from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from go2wm.contracts import CameraConfig, DatasetSplit, SceneConfig
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest, write_episode
from go2wm.data.g1 import (
    ALIGNMENT_SCHEDULE,
    G1_PACKET_SCHEMA,
    G1_ROLE_MODES,
    G1_ROLE_SEEDS,
    G1_SCOPE,
    G1_SPLIT_ID,
    G1_SPLIT_SALT,
    G1VerificationError,
    alignment_schedule_payload,
    build_g1_montage,
    sha256_file,
    verify_g1_packet,
    verify_master_split,
    write_g1_verification,
)
from go2wm.data.scripted_policy import ArenaGeometry, CommandBounds, ScenarioSampler, ScriptedPolicy
from go2wm.learning.splits import build_split_manifest, write_split_manifest
from go2wm.sim.controller_handoff import ControllerHandoffSummary
from go2wm.sim.fake import DeterministicFakeSimulator, FakeSimulatorConfig


def _packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, bad_schedule: bool = False) -> Path:
    root = tmp_path / "packet"
    root.mkdir()
    handoff = tmp_path / "handoff"
    (handoff / "policy").mkdir(parents=True)
    (handoff / "robot").mkdir()
    (handoff / "config").mkdir()
    for path, payload in (
        (handoff / "manifest.json", b"manifest"),
        (handoff / "policy" / "policy.onnx", b"policy"),
        (handoff / "robot" / "go2.xml", b"robot"),
        (handoff / "config" / "deploy.yaml", b"deploy"),
    ):
        path.write_bytes(payload)
    summary = ControllerHandoffSummary(
        controller_id="mjlab-test-iter10000",
        source_revision="test",
        task="Unitree-Go2-Flat",
        policy_path=handoff / "policy" / "policy.onnx",
        checkpoint_path=handoff / "checkpoint.pt",
        physics_dt_s=0.005,
        controller_dt_s=0.02,
        file_count=4,
    )
    monkeypatch.setattr("go2wm.data.g1.verify_controller_handoff", lambda _path: summary)

    manifest = build_split_manifest(
        range(1, 401),
        salt=G1_SPLIT_SALT,
        reserved_test_seeds=range(377, 401),
    )
    assert manifest.split_id == G1_SPLIT_ID
    write_split_manifest(root / "splits.json", manifest)
    arena = ArenaGeometry(4.0, 4.0, box_half_extent_m=0.2, robot_box_clearance_m=0.6)
    sampler = ScenarioSampler(arena)
    config = FakeSimulatorConfig(
        physics_dt_s=0.005,
        camera=CameraConfig("overhead_v1", 32, 32),
        scene=SceneConfig("g1-test-scene", 4.0, 4.0),
    )
    policy_digest = sha256_file(summary.policy_path)
    data = root / "data"
    for role in ("alignment", "push", "resist"):
        seed = G1_ROLE_SEEDS[role]
        scenario = sampler.sample(seed, episode_prefix=f"g1a-{role}")
        scenario = replace(
            scenario,
            reset=replace(scenario.reset, episode_id=f"g1a-{role}-{seed:05d}"),
        )
        simulator = DeterministicFakeSimulator(config)
        collector = EpisodeCollector(simulator, CollectionConfig(block_duration_s=0.5))
        request = EpisodeRequest(
            scenario.reset,
            DatasetSplit.TRAIN,
            scenario.goal,
            "g1-test-packet",
            {
                "g1_scope": G1_SCOPE,
                "g1_role": role,
                "collection_mode": G1_ROLE_MODES[role],
                "controller_id": summary.controller_id,
                "adapter_controller_id": "unitree_rl_mjlab-test",
                "policy_sha256": policy_digest,
                "backend": "mujoco-mjlab-direct",
                "dimos_status": "pending_g5_l7",
                "collection_policy": (
                    "g1a-alignment-schedule-v1"
                    if role == "alignment"
                    else "scripted-collection-policy-v1"
                ),
            },
        )
        if role == "alignment":
            schedule = ALIGNMENT_SCHEDULE
            if bad_schedule:
                schedule = (replace(schedule[0], yaw_rate_rps=0.2), *schedule[1:])
            episode = collector.collect(request, schedule)
        else:
            policy = ScriptedPolicy(scenario, arena, CommandBounds(), seed=seed)
            episode = collector.collect_with_policy(request, policy, 20)
        write_episode(data, episode)

    montage_path = build_g1_montage(root)

    packet = {
        "schema_version": G1_PACKET_SCHEMA,
        "packet_id": "g1-test-packet",
        "scope": G1_SCOPE,
        "dimos_status": "pending_g5_l7",
        "split_id": G1_SPLIT_ID,
        "seeds": G1_ROLE_SEEDS,
        "alignment_schedule": alignment_schedule_payload(),
        "backend": "mujoco-mjlab-direct",
        "adapter_controller_id": "unitree_rl_mjlab-test",
        "scene_id": "g1-test-scene",
        "camera_id": "overhead_v1",
        "montage": {
            "path": "montage.html",
            "block_count": 10,
            "sha256": sha256_file(montage_path),
        },
        "controller_handoff": {
            "path": str(handoff),
            "controller_id": summary.controller_id,
            "manifest_sha256": sha256_file(handoff / "manifest.json"),
            "policy_sha256": policy_digest,
            "robot_xml_sha256": sha256_file(handoff / "robot" / "go2.xml"),
            "deploy_yaml_sha256": sha256_file(handoff / "config" / "deploy.yaml"),
        },
    }
    (root / "packet.json").write_text(json.dumps(packet), encoding="utf-8")
    return root


def test_alignment_schedule_is_exactly_twenty_blocks() -> None:
    expected = (
        [(0.0, 0.0)] * 2
        + [(0.4, 0.0)] * 4
        + [(0.0, 0.0)] * 2
        + [(0.0, 0.6)] * 4
        + [(0.0, 0.0)] * 2
        + [(0.0, -0.6)] * 2
        + [(0.0, 0.0)] * 2
        + [(0.2, 0.0), (0.0, 0.0)]
    )
    assert [
        (action.forward_velocity_mps, action.yaw_rate_rps) for action in ALIGNMENT_SCHEDULE
    ] == expected
    assert len(ALIGNMENT_SCHEDULE) == 20
    assert all(action.duration_s == 0.5 for action in ALIGNMENT_SCHEDULE)


def test_complete_packet_passes_and_seals_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _packet(tmp_path, monkeypatch)
    report = verify_g1_packet(root)
    assert report["passed"] is True
    assert report["dimos_status"] == "pending_g5_l7"
    assert report["checks"]["twenty_consecutive_valid_blocks_each"] is True
    assert report["transient_contact_examples"]
    montage = (root / "montage.html").read_text(encoding="utf-8")
    assert montage.count('data-block-row="true"') == 10
    assert montage.count("data:image/png;base64,") == 20
    report_path, checksums_path = write_g1_verification(root, report)
    assert report_path.is_file() and checksums_path.is_file()
    inventory = checksums_path.read_text(encoding="utf-8")
    assert "verification.json" in inventory
    assert "data/g1a-alignment-00003/episode.json" in inventory
    with pytest.raises(FileExistsError):
        write_g1_verification(root, report)


def test_changed_alignment_schedule_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _packet(tmp_path, monkeypatch, bad_schedule=True)
    with pytest.raises(G1VerificationError, match="frozen schedule"):
        verify_g1_packet(root)


def test_private_three_seed_split_is_rejected() -> None:
    private = build_split_manifest(G1_ROLE_SEEDS.values(), salt=G1_SPLIT_SALT)
    with pytest.raises(G1VerificationError, match="frozen split id"):
        verify_master_split(private)


def test_builder_retains_failed_evidence(tmp_path: Path) -> None:
    manifest = build_split_manifest(
        range(1, 401),
        salt=G1_SPLIT_SALT,
        reserved_test_seeds=range(377, 401),
    )
    split_path = tmp_path / "splits.json"
    write_split_manifest(split_path, manifest)
    target = tmp_path / "packet"
    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(project / "src")}
    result = subprocess.run(
        [
            sys.executable,
            str(project / "scripts" / "collect_g1_packet.py"),
            "--handoff",
            str(tmp_path / "missing-handoff"),
            "--splits",
            str(split_path),
            "--out",
            str(target),
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not target.exists()
    failed = target.with_name("packet.failed")
    failure = json.loads((failed / "failure.json").read_text(encoding="utf-8"))
    assert failure["schema_version"] == "go2wm.g1a-collection-failure.v1"
    assert failure["dimos_status"] == "pending_g5_l7"
    assert "retained failed G1A evidence" in result.stderr
