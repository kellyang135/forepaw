#!/usr/bin/env python3
"""Build the direct-MjLab G1A data-contract packet.

This gate deliberately bypasses dimOS. It proves direct production-adapter
timing and storage; true-Go2 dimOS transport remains pending for G5/L7.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from go2wm.contracts import DatasetSplit
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest, write_episode
from go2wm.data.g1 import (
    ALIGNMENT_SCHEDULE,
    G1_PACKET_SCHEMA,
    G1_ROLE_MODES,
    G1_ROLE_SEEDS,
    G1_SCOPE,
    alignment_schedule_payload,
    build_g1_montage,
    sha256_file,
    verify_g1_packet,
    verify_master_split,
)
from go2wm.data.scripted_policy import ArenaGeometry, CommandBounds, ScenarioSampler, ScriptedPolicy
from go2wm.learning.splits import read_split_manifest
from go2wm.sim.controller_handoff import verify_controller_handoff
from go2wm.sim.locomotion import (
    MJLAB_GO2_FLAT,
    OnnxPolicy,
    check_mjlab_deploy_yaml,
    check_mjlab_onnx_metadata,
)
from go2wm.sim.mujoco_go2 import MujocoGo2Config, MujocoGo2Simulator


def source_state(root: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=all").splitlines()
    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(dirty), "dirty_paths": dirty}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--packet-id", default=None)
    args = parser.parse_args()

    target = args.out
    if target.exists():
        raise SystemExit(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    sim = None
    try:
        handoff = verify_controller_handoff(args.handoff)
        policy_path = handoff.policy_path
        robot_xml = args.handoff / "robot" / "go2.xml"
        deploy_yaml = args.handoff / "config" / "deploy.yaml"
        check_mjlab_onnx_metadata(policy_path, MJLAB_GO2_FLAT)
        check_mjlab_deploy_yaml(deploy_yaml, MJLAB_GO2_FLAT)
        policy = OnnxPolicy(policy_path, MJLAB_GO2_FLAT.observation_size)
        config = MujocoGo2Config()
        sim = MujocoGo2Simulator(MJLAB_GO2_FLAT, policy, robot_xml, config)
        arena = ArenaGeometry(
            config.scene.arena_width_m,
            config.scene.arena_height_m,
            box_half_extent_m=config.scene.box_half_extent_m,
            robot_box_clearance_m=config.min_robot_box_clearance_m,
        )
        bounds = CommandBounds(
            config.min_forward_velocity_mps,
            config.max_forward_velocity_mps,
            config.max_abs_yaw_rate_rps,
            config.block_duration_s,
        )
        sampler = ScenarioSampler(arena)
        collector = EpisodeCollector(sim, CollectionConfig(block_duration_s=0.5))
        split_manifest = read_split_manifest(args.splits)
        verify_master_split(split_manifest)
        shutil.copyfile(args.splits, staging / "splits.json")

        packet_id = args.packet_id or target.name
        policy_digest = sha256_file(policy_path)
        common_metadata = {
            "g1_scope": G1_SCOPE,
            "controller_id": handoff.controller_id,
            "adapter_controller_id": sim.controller_id,
            "policy_sha256": policy_digest,
            "backend": "mujoco-mjlab-direct",
            "dimos_status": "pending_g5_l7",
        }
        data_root = staging / "data"
        for role in ("alignment", "push", "resist"):
            seed = G1_ROLE_SEEDS[role]
            scenario = sampler.sample(seed, episode_prefix=f"g1a-{role}")
            if scenario.mode != G1_ROLE_MODES[role]:
                raise RuntimeError(
                    f"seed {seed} produced {scenario.mode!r}, expected {G1_ROLE_MODES[role]!r}"
                )
            scenario = replace(
                scenario,
                reset=replace(scenario.reset, episode_id=f"g1a-{role}-{seed:05d}"),
            )
            metadata = {
                **common_metadata,
                "g1_role": role,
                "collection_mode": scenario.mode,
                "collection_policy": (
                    "g1a-alignment-schedule-v1"
                    if role == "alignment"
                    else "scripted-collection-policy-v1"
                ),
            }
            request = EpisodeRequest(
                reset=scenario.reset,
                split=DatasetSplit.TRAIN,
                goal=scenario.goal,
                run_id=packet_id,
                metadata=metadata,
            )
            if role == "alignment":
                record = collector.collect(request, ALIGNMENT_SCHEDULE)
            else:
                scripted = ScriptedPolicy(scenario, arena, bounds, seed=seed)
                record = collector.collect_with_policy(request, scripted, 20)
            write_episode(data_root, record)

        montage_path = build_g1_montage(staging)

        project_root = Path(__file__).resolve().parents[1]
        packet = {
            "schema_version": G1_PACKET_SCHEMA,
            "packet_id": packet_id,
            "scope": G1_SCOPE,
            "dimos_status": "pending_g5_l7",
            "split_id": split_manifest.split_id,
            "seeds": G1_ROLE_SEEDS,
            "alignment_schedule": alignment_schedule_payload(),
            "backend": "mujoco-mjlab-direct",
            "adapter_controller_id": sim.controller_id,
            "scene_id": sim.scene_config.scene_id,
            "camera_id": sim.camera_config.camera_id,
            "montage": {
                "path": "montage.html",
                "block_count": 10,
                "sha256": sha256_file(montage_path),
            },
            "source": source_state(project_root),
            "controller_handoff": {
                "path": str(args.handoff.resolve()),
                "controller_id": handoff.controller_id,
                "manifest_sha256": sha256_file(args.handoff / "manifest.json"),
                "policy_sha256": policy_digest,
                "robot_xml_sha256": sha256_file(robot_xml),
                "deploy_yaml_sha256": sha256_file(deploy_yaml),
            },
        }
        (staging / "packet.json").write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_g1_packet(staging)
        staging.rename(target)
    except BaseException as error:
        failed = target.with_name(f"{target.name}.failed")
        if failed.exists():
            failed = target.with_name(f"{target.name}.failed-{uuid.uuid4().hex}")
        failure = {
            "schema_version": "go2wm.g1a-collection-failure.v1",
            "scope": G1_SCOPE,
            "dimos_status": "pending_g5_l7",
            "requested_target": str(target),
            "failed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            (staging / "failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            staging.rename(failed)
            print(f"retained failed G1A evidence: {failed}", file=sys.stderr)
        except BaseException as retention_error:
            print(
                f"could not retain failed G1A evidence at {staging}: {retention_error}",
                file=sys.stderr,
            )
        raise
    finally:
        if sim is not None:
            sim.close()

    print(json.dumps({"packet": str(target), "packet_id": packet_id, "status": "COLLECTED"}))
    print("Run scripts/verify_g1_packet.py for independent verification and outer checksums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
