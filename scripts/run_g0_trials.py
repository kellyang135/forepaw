#!/usr/bin/env python3
"""Run the G0 physics trials through the project ``MujocoGo2Simulator`` (DRAFT).

Trials (TWO_PERSON_PLAN.md section 6, G0 row): stand, walk, turn, a command
response grid (for P-001 action bounds), aimed light/resistant box pushes,
reset repeatability, camera coverage, and throughput. The pushing collector
steers with privileged poses, which is allowed for collection only.

Tolerances marked PROPOSED are starting values; SIM owns the final ones.
The output is a feasibility record for SIM review, not a G0 sign-off.

Example (rl_sar policy on the Menagerie Go2, proposed D-020):

    MUJOCO_GL=egl python scripts/run_g0_trials.py --controller rl_sar \\
        --policy /tmp/robot_lab_policy.pt --menagerie-cache /tmp/go2wm-menagerie-cache \\
        --output-dir artifacts/runs/$(date -u +%Y%m%dT%H%M%SZ)-g0-trials-rl_sar

Example (MjLab ONNX policy on the MjLab Go2 XML, D-019):

    python scripts/run_g0_trials.py --controller mjlab --policy <packet>/policy.onnx \\
        --robot-xml <unitree_rl_mjlab>/src/assets/robots/unitree_go2/xmls/go2.xml \\
        --deploy-yaml <packet>/deploy.yaml --output-dir artifacts/runs/<id>-g0-trials-mjlab
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from go2wm.contracts import ActionCommand, ObjectState, Pose2D, ResetRequest
from go2wm.sim.g0_acceptance import G0Acceptance, load_g0_acceptance
from go2wm.sim.locomotion import (
    MJLAB_GO2_FLAT,
    RL_SAR_ROBOT_LAB_GO2,
    OnnxPolicy,
    check_mjlab_deploy_yaml,
    check_mjlab_onnx_metadata,
    load_rl_sar_policy,
    sha256_file,
)
from go2wm.sim.mujoco_go2 import MujocoGo2Config, MujocoGo2Simulator
from go2wm.sim.task_scene import menagerie_scene_xml

LIGHT_MIN_DISPLACEMENT_M = 0.10  # PROPOSED
RESISTANT_MAX_DISPLACEMENT_M = 0.05  # PROPOSED
REQUIRED_OF_FIVE = 4
# Approximate Go2 top-view extent (base frame): head to tail, hip to hip plus feet.
ROBOT_FOOTPRINT_CORNERS_M = ((0.36, 0.17), (0.36, -0.17), (-0.36, 0.17), (-0.36, -0.17))


def source_snapshot() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=all").splitlines()
    paths = (
        "scripts/run_g0_trials.py",
        "src/go2wm/sim/locomotion.py",
        "src/go2wm/sim/mujoco_go2.py",
        "src/go2wm/sim/task_scene.py",
    )
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty_tree": bool(status),
        "dirty_path_count": len(status),
        "relevant_files": {
            path: sha256_file(root / path)
            for path in paths
        },
    }


def write_png(path: Path, rgb: bytes, width: int, height: int) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + rgb[r * width * 3 : (r + 1) * width * 3] for r in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Trials:
    def __init__(
        self,
        sim: MujocoGo2Simulator,
        output_dir: Path,
        seed: int,
        acceptance: G0Acceptance | None = None,
    ) -> None:
        self.sim = sim
        self.out = output_dir
        self.rng = np.random.default_rng(seed)
        self.blocks = 0
        self.block_wall_s = 0.0
        self.camera_points_checked = 0
        self.camera_points_outside: list[dict[str, Any]] = []
        self.episode = 0
        self.light_min_displacement_m = (
            acceptance.movable_min_displacement_m
            if acceptance is not None
            else LIGHT_MIN_DISPLACEMENT_M
        )
        self.resistant_max_displacement_m = (
            acceptance.resistant_max_displacement_m
            if acceptance is not None
            else RESISTANT_MAX_DISPLACEMENT_M
        )
        self.required_of_five = (
            acceptance.required_successes_of_five if acceptance is not None else REQUIRED_OF_FIVE
        )

    # ---------------------------------------------------------------- helpers
    def reset(self, pose: Pose2D, objects: tuple[ObjectState, ...] = ()) -> Any:
        self.episode += 1
        report = self.sim.reset(ResetRequest(f"g0-{self.episode:04d}", self.episode, pose, objects))
        self.check_camera()
        return report

    def step(self, forward: float, yaw_rate: float) -> Any:
        started = time.perf_counter()
        transition = self.sim.execute_block(ActionCommand(forward, yaw_rate))
        self.block_wall_s += time.perf_counter() - started
        self.blocks += 1
        self.check_camera()
        return transition

    def jitter_pose(
        self, x: float, y: float, yaw: float, xy: float = 0.1, dyaw: float = 0.1
    ) -> Pose2D:
        return Pose2D(
            x + float(self.rng.uniform(-xy, xy)),
            y + float(self.rng.uniform(-xy, xy)),
            yaw + float(self.rng.uniform(-dyaw, dyaw)),
        )

    def check_camera(self) -> None:
        """Project the robot footprint and box top corners into the overhead image."""

        scene = self.sim.config.scene
        height = scene.camera_height_m
        focal = (scene.image_height_px / 2) / math.tan(math.radians(scene.camera_fovy_deg) / 2)
        labels = self.sim.labels()
        robot = labels.robot_pose
        rc, rs = math.cos(robot.yaw_rad), math.sin(robot.yaw_rad)
        points = [
            ("robot", robot.x_m + rc * dx - rs * dy, robot.y_m + rs * dx + rc * dy, 0.35)
            for dx, dy in ROBOT_FOOTPRINT_CORNERS_M
        ]
        half = scene.box_half_extent_m
        for item in labels.objects:
            c, s = math.cos(item.pose.yaw_rad), math.sin(item.pose.yaw_rad)
            for dx, dy in ((half, half), (half, -half), (-half, half), (-half, -half)):
                points.append(
                    (
                        item.object_id,
                        item.pose.x_m + c * dx - s * dy,
                        item.pose.y_m + s * dx + c * dy,
                        2 * half,
                    )
                )
        for name, x, y, z in points:
            self._check_camera_point(name, x, y, z, focal, height)

    def check_goal_extrema(self, center_margin_m: float, radius_m: float) -> None:
        """Check every edge of goal regions at all declared center extrema."""

        scene = self.sim.config.scene
        height = scene.camera_height_m
        focal = (scene.image_height_px / 2) / math.tan(math.radians(scene.camera_fovy_deg) / 2)
        x_limit = scene.arena_width_m / 2 - center_margin_m
        y_limit = scene.arena_height_m / 2 - center_margin_m
        for cx in (-x_limit, x_limit):
            for cy in (-y_limit, y_limit):
                for index, (dx, dy) in enumerate(
                    ((radius_m, 0.0), (-radius_m, 0.0), (0.0, radius_m), (0.0, -radius_m))
                ):
                    self._check_camera_point(
                        f"goal-extreme-{cx:+.2f}-{cy:+.2f}-{index}",
                        cx + dx,
                        cy + dy,
                        0.01,
                        focal,
                        height,
                    )

    def _check_camera_point(
        self, name: str, x: float, y: float, z: float, focal: float, height: float
    ) -> None:
        scene = self.sim.config.scene
        u = scene.image_width_px / 2 + focal * x / (height - z)
        v = scene.image_height_px / 2 - focal * y / (height - z)
        self.camera_points_checked += 1
        if not (2 <= u <= scene.image_width_px - 2 and 2 <= v <= scene.image_height_px - 2):
            self.camera_points_outside.append(
                {"episode": self.episode, "point": name, "u": round(u, 1), "v": round(v, 1)}
            )

    def save_frame(self, name: str) -> str:
        observation = self.sim.observe()
        path = self.out / f"{name}.png"
        write_png(path, observation.rgb, observation.width_px, observation.height_px)
        return path.name

    # ----------------------------------------------------------------- trials
    def free_motion(
        self,
        label: str,
        forward: float,
        yaw_rate: float,
        blocks: int,
        repeats: int = 5,
        start_x: float | None = None,
    ) -> dict[str, Any]:
        runs = []
        for _ in range(repeats):
            start = self.reset(
                self.jitter_pose(
                    start_x if start_x is not None else (-1.2 if forward > 0 else 0.0), 0.0, 0.0
                )
            ).final_labels
            fell = False
            speeds, yaw_rates = [], []
            previous = start
            for index in range(blocks):
                transition = self.step(forward, yaw_rate)
                fell = fell or transition.events.fell
                end = transition.end_labels
                if index >= 2:
                    heading = previous.robot_pose.yaw_rad
                    dx = end.robot_pose.x_m - previous.robot_pose.x_m
                    dy = end.robot_pose.y_m - previous.robot_pose.y_m
                    speeds.append((dx * math.cos(heading) + dy * math.sin(heading)) / 0.5)
                    yaw_rates.append(
                        wrap(end.robot_pose.yaw_rad - previous.robot_pose.yaw_rad) / 0.5
                    )
                previous = end
            drift = math.hypot(
                previous.robot_pose.x_m - start.robot_pose.x_m,
                previous.robot_pose.y_m - start.robot_pose.y_m,
            )
            runs.append(
                {
                    "fell": fell,
                    "forward_mps": round(float(np.mean(speeds)), 3),
                    "yaw_rate_rps": round(float(np.mean(yaw_rates)), 3),
                    "displacement_m": round(drift, 3),
                    "yaw_change_rad": round(
                        wrap(previous.robot_pose.yaw_rad - start.robot_pose.yaw_rad), 3
                    ),
                }
            )
        return {
            "trial": label,
            "command": [forward, yaw_rate],
            "blocks": blocks,
            "falls": sum(r["fell"] for r in runs),
            "runs": runs,
        }

    def response_grid(self) -> list[dict[str, Any]]:
        config = self.sim.config
        grid = []
        for forward in (0.0, 0.2, 0.4, config.max_forward_velocity_mps):
            for yaw in (-config.max_abs_yaw_rate_rps, -0.6, 0.0, 0.6, config.max_abs_yaw_rate_rps):
                result = self.free_motion(
                    f"grid {forward:+.1f} {yaw:+.1f}",
                    forward,
                    yaw,
                    blocks=6,
                    repeats=1,
                    start_x=-0.9 * (forward > 0),
                )
                run = result["runs"][0]
                grid.append(
                    {
                        "command": [forward, yaw],
                        "measured_forward_mps": run["forward_mps"],
                        "measured_yaw_rate_rps": run["yaw_rate_rps"],
                        "fell": run["fell"],
                    }
                )
        return grid

    def pushes(self, appearance: str, movable: bool, speed: float) -> dict[str, Any]:
        runs = []
        for index, approach_deg in enumerate((-30, -15, 0, 15, 30)):
            approach = math.radians(approach_deg)
            box = Pose2D(-0.1, 0.0, float(self.rng.uniform(-0.2, 0.2)))
            distance = 1.1
            robot = Pose2D(
                box.x_m - distance * math.cos(approach),
                box.y_m - distance * math.sin(approach),
                approach,
            )
            item = ObjectState(f"box_{appearance}", appearance, box, movable)
            report = self.reset(robot, (item,))
            start = report.final_labels.object(item.object_id).pose
            if index == 2:
                self.save_frame(f"push_{appearance}_start")
            fell, contacts = False, 0
            for _ in range(12):
                labels = self.sim.labels()
                target = labels.object(item.object_id).pose
                bearing = math.atan2(
                    target.y_m - labels.robot_pose.y_m, target.x_m - labels.robot_pose.x_m
                )
                yaw_rate = 1.5 * wrap(bearing - labels.robot_pose.yaw_rad)
                transition = self.step(speed, yaw_rate)
                fell = fell or transition.events.fell
                contacts += transition.events.contact_sample_count
            if index == 2:
                self.save_frame(f"push_{appearance}_end")
            end = self.sim.labels().object(item.object_id).pose
            runs.append(
                {
                    "approach_deg": approach_deg,
                    "box_displacement_m": round(
                        math.hypot(end.x_m - start.x_m, end.y_m - start.y_m), 4
                    ),
                    "contact_samples": contacts,
                    "fell": fell,
                }
            )
        displacements = [r["box_displacement_m"] for r in runs]
        if movable:
            successes = sum(d >= self.light_min_displacement_m for d in displacements)
            rule = f"displacement >= {self.light_min_displacement_m} m"
        else:
            successes = sum(
                d <= self.resistant_max_displacement_m and r["contact_samples"] > 0
                for d, r in zip(displacements, runs, strict=True)
            )
            rule = f"contacted and displacement <= {self.resistant_max_displacement_m} m"
        return {
            "trial": f"push {appearance}",
            "mass_kg": self.sim.config.scene.box_classes[appearance].mass_kg,
            "command_forward_mps": speed,
            "rule": rule,
            "successes": successes,
            "falls": sum(r["fell"] for r in runs),
            "runs": runs,
        }

    def reset_repeatability(self, repeats: int = 10) -> dict[str, Any]:
        request_pose = Pose2D(-0.5, 0.3, 0.7)
        objects = (
            ObjectState("box_blue", "blue", Pose2D(0.6, 0.6, 0.0), True),
            ObjectState("box_red", "red", Pose2D(0.6, -0.6, 0.0), False),
        )
        poses, durations, settled, digests = [], [], [], set()
        for _ in range(repeats):
            report = self.reset(request_pose, objects)  # new scenario_seed each time
            pose = report.final_labels.robot_pose
            poses.append((pose.x_m, pose.y_m, pose.yaw_rad))
            durations.append(report.settle_duration_s)
            settled.append(report.settled)
            digests.add(report.final_observation.sha256)
        array = np.array(poses)
        return {
            "requested": [request_pose.x_m, request_pose.y_m, request_pose.yaw_rad],
            "settled": f"{sum(settled)}/{repeats}",
            "mean_offset_m": round(float(np.hypot(*(array[:, :2].mean(0) - [-0.5, 0.3]))), 4),
            "mean_yaw_offset_rad": round(float(array[:, 2].mean() - 0.7), 4),
            "std_xy_m": [round(float(v), 5) for v in array[:, :2].std(0)],
            "std_yaw_rad": round(float(array[:, 2].std()), 5),
            "settle_s_mean_max": [
                round(float(np.mean(durations)), 3),
                round(float(np.max(durations)), 3),
            ],
            "distinct_first_frames": len(digests),
            "note": "same requested pose; new scenario_seed (joint perturbation) per repeat",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--controller", choices=("rl_sar", "mjlab"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--menagerie-cache", type=Path, default=Path("/tmp/go2wm-menagerie-cache"))
    parser.add_argument("--robot-xml", type=Path, help="MjLab go2.xml (with its assets/ directory)")
    parser.add_argument(
        "--deploy-yaml", type=Path, help="MjLab deploy.yaml to check against the audited spec"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--push-speed", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--acceptance",
        type=Path,
        help="frozen G0 acceptance TOML; omission keeps the run explicitly draft",
    )
    return parser.parse_args()


def validate_frozen_configuration(
    acceptance: G0Acceptance,
    config: MujocoGo2Config,
    controller_id: str,
    policy: Path,
    robot_xml: Path,
) -> None:
    """Fail before simulation if the requested run differs from the frozen G0 values."""

    scene = config.scene
    actual = {
        "min_forward_mps": config.min_forward_velocity_mps,
        "max_forward_mps": config.max_forward_velocity_mps,
        "max_abs_yaw_rps": config.max_abs_yaw_rate_rps,
        "movable_mass_kg": scene.box_classes["blue"].mass_kg,
        "resistant_mass_kg": scene.box_classes["red"].mass_kg,
        "box_half_extent_m": scene.box_half_extent_m,
        "box_friction": scene.box_friction,
        "arena_width_m": scene.arena_width_m,
        "arena_height_m": scene.arena_height_m,
        "image_width_px": scene.image_width_px,
        "image_height_px": scene.image_height_px,
        "camera_fovy_deg": scene.camera_fovy_deg,
        "settle_min_s": config.settle_min_s,
        "settle_max_s": config.settle_max_s,
        "settle_window_s": config.settle_window_s,
        "settle_speed_tolerance_mps": config.settle_speed_tolerance_mps,
        "settle_yaw_rate_tolerance_rps": config.settle_yaw_rate_tolerance_rps,
        "settle_heading_tolerance_rad": config.settle_heading_tolerance_rad,
    }
    for name, value in actual.items():
        expected = getattr(acceptance, name)
        if not math.isclose(float(value), float(expected), abs_tol=1e-12):
            raise ValueError(f"frozen {name}={expected} does not match runtime {value}")
    if controller_id != acceptance.controller_id:
        raise ValueError("controller id does not match frozen G0 acceptance")
    if scene.camera_id != acceptance.camera_id:
        raise ValueError("frozen camera_id does not match runtime")
    if sha256_file(policy) != acceptance.policy_sha256:
        raise ValueError("policy does not match frozen G0 acceptance")
    if sha256_file(robot_xml) != acceptance.robot_xml_sha256:
        raise ValueError("robot XML does not match frozen G0 acceptance")


def main() -> int:
    args = parse_args()
    acceptance = load_g0_acceptance(args.acceptance) if args.acceptance is not None else None
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.controller == "rl_sar":
        spec = RL_SAR_ROBOT_LAB_GO2
        policy = load_rl_sar_policy(args.policy)
        robot_xml = menagerie_scene_xml(args.menagerie_cache)
    else:
        spec = MJLAB_GO2_FLAT
        if args.robot_xml is None:
            raise SystemExit("--robot-xml is required for --controller mjlab")
        if args.deploy_yaml is not None:
            check_mjlab_deploy_yaml(args.deploy_yaml, spec)
        check_mjlab_onnx_metadata(args.policy, spec)
        policy = OnnxPolicy(args.policy, spec.observation_size)
        robot_xml = args.robot_xml
    args.output_dir.mkdir(parents=True)

    import mujoco

    config = MujocoGo2Config()
    if acceptance is not None:
        validate_frozen_configuration(
            acceptance, config, spec.controller_id, args.policy, robot_xml
        )
    sim = MujocoGo2Simulator(spec, policy, robot_xml, config)
    trials = Trials(sim, args.output_dir, args.seed, acceptance)
    started = time.perf_counter()
    max_yaw = config.max_abs_yaw_rate_rps
    free = [
        trials.free_motion("stand", 0.0, 0.0, blocks=20),
        trials.free_motion("walk", config.max_forward_velocity_mps, 0.0, blocks=8),
        trials.free_motion("turn left", 0.0, 0.8 * max_yaw, blocks=8),
        trials.free_motion("turn right", 0.0, -0.8 * max_yaw, blocks=8),
    ]
    grid = trials.response_grid()
    light_class = next(k for k, v in config.scene.box_classes.items() if v.movable)
    heavy_class = next(k for k, v in config.scene.box_classes.items() if not v.movable)
    pushes = [
        trials.pushes(light_class, True, args.push_speed),
        trials.pushes(heavy_class, False, args.push_speed),
    ]
    resets = trials.reset_repeatability()
    if acceptance is not None:
        trials.check_goal_extrema(
            acceptance.goal_center_margin_m,
            acceptance.goal_radius_m,
        )
    trials.save_frame("reset_example")
    wall_s = time.perf_counter() - started

    checks = {
        "free_motion_no_falls": all(t["falls"] == 0 for t in free),
        "light_box_displaced_4_of_5": pushes[0]["successes"] >= trials.required_of_five,
        "resistant_box_holds_4_of_5": pushes[1]["successes"] >= trials.required_of_five,
        "no_falls_while_pushing": all(p["falls"] == 0 for p in pushes),
        "camera_keeps_robot_and_boxes_in_frame": not trials.camera_points_outside,
        "resets_settle": int(resets["settled"].split("/")[0])
        >= (acceptance.required_settled_of_ten if acceptance is not None else 10),
    }
    report = {
        "schema_version": "go2wm.g0-trials.v1" if acceptance else "go2wm.g0-trials.v0-draft",
        "run_id": args.output_dir.name,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": (
            "FORMAL_TRIALS_PASS_PENDING_COOWNER_SIGNOFF"
            if acceptance and all(checks.values()) and not acceptance.coowner_acknowledged
            else "FORMAL_G0_PASS"
            if acceptance and all(checks.values())
            else "FEASIBILITY_ALL_CHECKS_PASS"
            if all(checks.values())
            else "CHECKS_FAILED"
        ),
        "scope": (
            "formal G0 measurements; co-owner sign-off still required"
            if acceptance and not acceptance.coowner_acknowledged
            else "formal G0"
            if acceptance
            else "draft adapter and task scene; not a G0 sign-off until SIM reviews and reruns"
        ),
        "source": source_snapshot(),
        "seed": args.seed,
        "acceptance": (
            {
                "path": str(args.acceptance),
                "sha256": sha256_file(args.acceptance),
                "status": acceptance.status,
                "coowner_acknowledged": acceptance.coowner_acknowledged,
            }
            if acceptance is not None
            else None
        ),
        "checks": checks,
        "controller": {
            "id": spec.controller_id,
            "source": spec.source,
            "policy_file": args.policy.name,
            "policy_sha256": sha256_file(args.policy),
            "robot_xml_sha256": sim.built.robot_xml_sha256,
            "yaw_rate_trim_gain": spec.yaw_rate_trim_gain,
        },
        "scene_id": sim.scene_config.scene_id,
        "config": {
            "action_bounds": {
                "forward_mps": [config.min_forward_velocity_mps, config.max_forward_velocity_mps],
                "yaw_rate_rps": [-max_yaw, max_yaw],
            },
            "arena_m": [config.scene.arena_width_m, config.scene.arena_height_m],
            "camera": {
                "id": config.scene.camera_id,
                "size_px": [config.scene.image_width_px, config.scene.image_height_px],
                "fovy_deg": config.scene.camera_fovy_deg,
                "height_m": round(config.scene.camera_height_m, 4),
            },
            "box_half_extent_m": config.scene.box_half_extent_m,
            "box_classes": {
                k: {"mass_kg": v.mass_kg, "movable": v.movable}
                for k, v in config.scene.box_classes.items()
            },
            "physics_dt_s": sim.physics_dt_s,
            "control_dt_s": spec.control_dt_s,
        },
        "free_motion": free,
        "command_response_grid": grid,
        "pushes": pushes,
        "reset_repeatability": resets,
        "camera": {
            "points_checked": trials.camera_points_checked,
            "points_outside": trials.camera_points_outside[:20],
        },
        "throughput": {
            "blocks": trials.blocks,
            "block_wall_s_total": round(trials.block_wall_s, 2),
            "blocks_per_wall_s": round(trials.blocks / trials.block_wall_s, 2),
            "script_wall_s": round(wall_s, 1),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
    }
    report["exit_code"] = 0 if all(checks.values()) else 1
    report["completion_state"] = "complete"
    sim.close()
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = []
    for path in sorted(args.output_dir.iterdir()):
        if path.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines:
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"evidence checksum changed while finalizing: {name}")
    print(
        json.dumps(
            {"status": report["status"], "checks": checks, "throughput": report["throughput"]},
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
