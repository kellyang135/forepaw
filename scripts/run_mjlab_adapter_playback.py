#!/usr/bin/env python3
"""Run the predeclared MjLab command matrix through the project ONNX adapter.

This is the companion to ``run_mjlab_checkpoint_playback.py``. It exercises
the exported policy in ``MujocoGo2Simulator`` with five resets per command and
writes a checksum-protected report for the cross-simulator comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from go2wm.contracts import ActionCommand, Pose2D, ResetRequest
from go2wm.sim.locomotion import (
    MJLAB_GO2_FLAT,
    OnnxPolicy,
    check_mjlab_deploy_yaml,
    check_mjlab_onnx_metadata,
    sha256_file,
)
from go2wm.sim.mujoco_go2 import MujocoGo2Simulator

COMMAND_CASES = (
    ("stand", 0.0, 0.0, 10.0),
    ("forward_0.2", 0.2, 0.0, 4.0),
    ("forward_0.4", 0.4, 0.0, 4.0),
    ("forward_0.6", 0.6, 0.0, 4.0),
    ("turn_-0.96", 0.0, -0.96, 4.0),
    ("turn_+0.96", 0.0, 0.96, 4.0),
)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle]


def evaluate(cases: list[dict[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for case in cases:
        good = [run for run in case["runs"] if not run["failed"]]
        checks[f"{case['name']}: five safe repeats"] = len(good) == 5
        if not good:
            checks[f"{case['name']}: response"] = False
            continue
        vx = median([run["mean_forward_mps"] for run in good])
        wz = median([run["mean_yaw_rate_rps"] for run in good])
        if case["name"] == "stand":
            displacement = median([run["net_displacement_m"] for run in good])
            checks[f"{case['name']}: response"] = (
                abs(vx) <= 0.10 and abs(wz) <= 0.15 and displacement <= 0.50
            )
        elif case["forward_command_mps"]:
            checks[f"{case['name']}: response"] = (
                abs(vx - case["forward_command_mps"]) <= 0.15 and abs(wz) <= 0.15
            )
        else:
            checks[f"{case['name']}: response"] = (
                abs(wz - case["yaw_command_rps"]) <= 0.18 and abs(vx) <= 0.15
            )
    return checks


def source_snapshot(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    relevant = (
        "scripts/run_mjlab_adapter_playback.py",
        "src/go2wm/sim/locomotion.py",
        "src/go2wm/sim/mujoco_go2.py",
        "src/go2wm/sim/task_scene.py",
    )
    return {
        "revision": git("rev-parse", "HEAD"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "relevant_files": {name: sha256_file(root / name) for name in relevant},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mujoco
    import onnxruntime

    check_mjlab_onnx_metadata(args.policy)
    check_mjlab_deploy_yaml(args.deploy_yaml)
    policy = OnnxPolicy(args.policy, MJLAB_GO2_FLAT.observation_size)
    sim = MujocoGo2Simulator(MJLAB_GO2_FLAT, policy, args.robot_xml)
    cases: list[dict[str, Any]] = []
    try:
        for name, forward, yaw_rate, duration_s in COMMAND_CASES:
            case: dict[str, Any] = {
                "name": name,
                "forward_command_mps": forward,
                "lateral_command_mps": 0.0,
                "yaw_command_rps": yaw_rate,
                "duration_s": duration_s,
                "runs": [],
            }
            for repeat in range(args.repeats):
                failure: str | None = None
                speeds: list[float] = []
                yaw_rates: list[float] = []
                try:
                    reset = sim.reset(
                        ResetRequest(
                            f"adapter-playback-{name}-{repeat}",
                            args.seed + repeat,
                            Pose2D(0.0, 0.0, 0.0),
                            (),
                        )
                    )
                    start = reset.final_labels.robot_pose
                    previous = reset.final_labels
                    blocks = round(duration_s / sim.block_duration_s)
                    for block in range(blocks):
                        transition = sim.execute_block(ActionCommand(forward, yaw_rate))
                        if transition.events.fell:
                            failure = "project fall threshold"
                            break
                        end = transition.end_labels
                        if block >= 2:  # one-second warmup, matching the upstream run
                            heading = previous.robot_pose.yaw_rad
                            dx = end.robot_pose.x_m - previous.robot_pose.x_m
                            dy = end.robot_pose.y_m - previous.robot_pose.y_m
                            speeds.append(
                                (dx * math.cos(heading) + dy * math.sin(heading))
                                / sim.block_duration_s
                            )
                            yaw_rates.append(
                                wrap(end.robot_pose.yaw_rad - previous.robot_pose.yaw_rad)
                                / sim.block_duration_s
                            )
                        previous = end
                    if not speeds or not yaw_rates:
                        failure = failure or "no scored samples"
                    run_result = {
                        "repeat": repeat,
                        "failed": failure is not None,
                        "failure": failure,
                        "mean_forward_mps": sum(speeds) / len(speeds) if speeds else math.nan,
                        "mean_yaw_rate_rps": (
                            sum(yaw_rates) / len(yaw_rates) if yaw_rates else math.nan
                        ),
                        "net_displacement_m": math.hypot(
                            previous.robot_pose.x_m - start.x_m,
                            previous.robot_pose.y_m - start.y_m,
                        ),
                    }
                except Exception as error:
                    run_result = {
                        "repeat": repeat,
                        "failed": True,
                        "failure": f"{type(error).__name__}: {error}",
                    }
                case["runs"].append(run_result)
            cases.append(case)
    finally:
        sim.close()
    checks = evaluate(cases)
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": "go2wm.mjlab-adapter-playback.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "completion_state": "complete",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "source": source_snapshot(root),
        "controller": {
            "policy_file": args.policy.name,
            "policy_sha256": sha256_file(args.policy),
            "robot_xml_sha256": sha256_file(args.robot_xml),
            "deploy_yaml_sha256": sha256_file(args.deploy_yaml),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "protocol": {
            "seed": args.seed,
            "repeats": args.repeats,
            "block_duration_s": sim.block_duration_s,
            "warmup_s": 1.0,
            "lateral_velocity_mps": 0.0,
        },
        "checks": checks,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--deploy-yaml", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats != 5:
        raise ValueError("the predeclared acceptance protocol requires exactly five repeats")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    try:
        report = run(args)
    except Exception as error:
        report = {
            "schema_version": "go2wm.mjlab-adapter-playback.v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "completion_state": "failed",
            "verdict": "ERROR",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        f"{hashlib.sha256(report_path.read_bytes()).hexdigest()}  report.json\n",
        encoding="utf-8",
    )
    if sha256_file(report_path) != checksum_path.read_text().split()[0]:
        raise RuntimeError("post-write checksum verification failed for report.json")
    print(json.dumps({"verdict": report["verdict"], "output_dir": str(args.output_dir)}))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
