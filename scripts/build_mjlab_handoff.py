#!/usr/bin/env python3
"""Build an immutable, checksum-complete Go2 controller handoff packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from go2wm.sim.controller_handoff import (
    EXPECTED_COMMAND_ORDER,
    EXPECTED_JOINT_ORDER,
    PINNED_MODEL_XML_SHA256,
    PINNED_REPOSITORY,
    PINNED_REVISION,
    PINNED_TASK,
    SCHEMA_VERSION,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(directory: Path) -> None:
    manifest = directory / "checksums.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = directory / name
        if sha256_file(path) != expected:
            raise ValueError(f"checksum mismatch for {path}")


def load_pass_report(directory: Path, name: str) -> dict[str, Any]:
    verify_checksums(directory)
    path = directory / name
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("verdict", report.get("status")) not in {
        "PASS",
        "FEASIBILITY_ALL_CHECKS_PASS",
    }:
        raise ValueError(f"{path} is not a passing report")
    return report


def copy_file(
    source: Path, output: Path, relative: str, files: dict[str, dict[str, Any]], key: str
) -> None:
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    files[key] = {
        "path": relative,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    training_manifest = json.loads((args.training / "manifest.json").read_text())
    if training_manifest.get("schema_version") != "go2wm.training-evidence.v1":
        raise ValueError("unexpected training evidence schema")
    source = training_manifest["source"]
    if (
        source.get("repository") != PINNED_REPOSITORY
        or source.get("revision") != PINNED_REVISION
        or source.get("task") != PINNED_TASK
    ):
        raise ValueError("training evidence source does not match the pinned controller")
    for description in training_manifest["files"].values():
        path = args.training / description["path"]
        if sha256_file(path) != description["sha256"]:
            raise ValueError(f"training evidence checksum mismatch for {path}")

    checkpoint_report = load_pass_report(args.checkpoint_playback, "report.json")
    adapter_report = load_pass_report(args.adapter_playback, "report.json")
    comparison_report = load_pass_report(args.comparison, "comparison.json")
    g0_report = load_pass_report(args.g0_trials, "report.json")
    if comparison_report["inputs"]["upstream_report"]["sha256"] != sha256_file(
        args.checkpoint_playback / "report.json"
    ):
        raise ValueError("comparison does not reference the supplied checkpoint playback")
    if comparison_report["inputs"]["adapter_report"]["sha256"] != sha256_file(
        args.adapter_playback / "report.json"
    ):
        raise ValueError("comparison does not reference the supplied adapter playback")
    if checkpoint_report["checkpoint"]["sha256"] != training_manifest["files"][
        "checkpoint"
    ]["sha256"]:
        raise ValueError("checkpoint playback did not use the training checkpoint")
    if adapter_report["controller"]["policy_sha256"] != training_manifest["files"][
        "policy_onnx"
    ]["sha256"]:
        raise ValueError("adapter playback did not use the training ONNX policy")
    if g0_report["controller"]["policy_sha256"] != training_manifest["files"][
        "policy_onnx"
    ]["sha256"]:
        raise ValueError("G0 draft trials did not use the training ONNX policy")

    output = args.output
    output.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    training_files = {
        "policy_onnx": ("policy/policy.onnx", "policy/policy.onnx"),
        "checkpoint": ("checkpoint/model_10000.pt", "checkpoint/model_10000.pt"),
        "environment_config": ("config/env.yaml", "config/env.yaml"),
        "agent_config": ("config/agent.yaml", "config/agent.yaml"),
        "deployment_config": ("config/deploy.yaml", "config/deploy.yaml"),
        "robot_xml": ("source/go2.xml", "robot/go2.xml"),
        "training_log": ("logs/training-full.log", "logs/training-full.log"),
    }
    for key, (source_relative, destination_relative) in training_files.items():
        copy_file(
            args.training / source_relative,
            output,
            destination_relative,
            files,
            key,
        )

    asset_dir = args.checkout / "src/assets/robots/unitree_go2/xmls/assets"
    assets = sorted(path for path in asset_dir.iterdir() if path.is_file())
    if len(assets) != 16:
        raise ValueError(f"expected 16 Go2 meshes, found {len(assets)}")
    for index, asset in enumerate(assets):
        copy_file(
            asset,
            output,
            f"robot/assets/{asset.name}",
            files,
            f"robot_mesh_{index:02d}",
        )

    evidence_files = (
        (
            "checkpoint_playback_report",
            args.checkpoint_playback / "report.json",
            "validation/checkpoint-playback-report.json",
        ),
        (
            "checkpoint_playback_trajectories",
            args.checkpoint_playback / "trajectories.jsonl",
            "validation/checkpoint-playback-trajectories.jsonl",
        ),
        (
            "adapter_playback_report",
            args.adapter_playback / "report.json",
            "validation/adapter-playback-report.json",
        ),
        (
            "playback_comparison",
            args.comparison / "comparison.json",
            "validation/playback-comparison.json",
        ),
        (
            "g0_draft_report",
            args.g0_trials / "report.json",
            "validation/g0-draft-report.json",
        ),
        (
            "upstream_play_script",
            args.checkout / "scripts/play.py",
            "source/play.py",
        ),
        (
            "upstream_go2_env_config",
            args.checkout / "src/tasks/velocity/config/go2/env_cfgs.py",
            "source/go2-env_cfgs.py",
        ),
    )
    for key, source_path, relative in evidence_files:
        copy_file(source_path, output, relative, files, key)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "controller_id": "unitree_rl_mjlab-1425b15-Unitree-Go2-Flat-iter10000",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            "repository": PINNED_REPOSITORY,
            "revision": PINNED_REVISION,
            "task": PINNED_TASK,
        },
        "compatibility": {
            "model_xml_sha256": PINNED_MODEL_XML_SHA256,
            "joint_order": list(EXPECTED_JOINT_ORDER),
            "command_order": list(EXPECTED_COMMAND_ORDER),
            "action_semantics": "scaled_joint_position_target",
            "onnx_data_mode": "self_contained",
            "physics_dt_s": 0.005,
            "controller_dt_s": 0.02,
            "physics_substeps_per_control": 4,
        },
        "validation": {
            "training_smoke_passed": True,
            "checkpoint_playback_passed": True,
            "onnx_sim_playback_passed": True,
            "onnx_export_present": True,
            "checkpoint_playback_report_sha256": files["checkpoint_playback_report"][
                "sha256"
            ],
            "adapter_playback_report_sha256": files["adapter_playback_report"]["sha256"],
            "playback_comparison_sha256": files["playback_comparison"]["sha256"],
            "g0_signoff": False,
        },
        "files": files,
        "limitations": [
            "The C++ unitree_mujoco deployment route was not exercised because it "
            "requires a gamepad to drive FSM transitions and commands.",
            "The retained G0 trials are a draft-adapter feasibility run, not joint G0 sign-off.",
            "dimOS transport and physical-robot deployment are not verified.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    packet_files = sorted(path for path in output.rglob("*") if path.is_file())
    (output / "checksums.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output)}\n" for path in packet_files
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-playback", type=Path, required=True)
    parser.add_argument("--adapter-playback", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--g0-trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = build(args)
    print(json.dumps({"controller_id": manifest["controller_id"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
