#!/usr/bin/env python3
"""Deterministic, headless playback of a Go2 checkpoint in upstream MjLab.

Run this script from the pinned unitree_rl_mjlab checkout's Python environment.
It uses the upstream task registry, environment, RSL-RL runner, and checkpoint
loader; no project simulator code participates in the playback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PINNED_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
PINNED_TASK = "Unitree-Go2-Flat"
COMMAND_CASES = (
    ("stand", 0.0, 0.0, 10.0),
    ("forward_0.2", 0.2, 0.0, 4.0),
    ("forward_0.4", 0.4, 0.0, 4.0),
    ("forward_0.6", 0.6, 0.0, 4.0),
    ("turn_-0.96", 0.0, -0.96, 4.0),
    ("turn_+0.96", 0.0, 0.96, 4.0),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def trial_summary(samples: list[dict[str, float]], warmup_steps: int) -> dict[str, float]:
    scored = samples[warmup_steps:]
    if not scored:
        raise RuntimeError("trial has no samples after warmup")
    start, end = samples[0], samples[-1]
    return {
        "mean_forward_mps": sum(row["forward_mps"] for row in scored) / len(scored),
        "mean_yaw_rate_rps": sum(row["yaw_rate_rps"] for row in scored) / len(scored),
        "net_displacement_m": math.hypot(end["x_m"] - start["x_m"], end["y_m"] - start["y_m"]),
        "min_base_height_m": min(row["z_m"] for row in samples),
        "max_tilt_deg": max(row["tilt_deg"] for row in samples),
    }


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


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import src.tasks  # noqa: F401
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    checkout = args.checkout.resolve()
    checkpoint = args.checkpoint.resolve()
    revision = git(checkout, "rev-parse", "HEAD")
    if revision != PINNED_REVISION:
        raise RuntimeError(f"checkout revision {revision} != pinned {PINNED_REVISION}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != args.checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint sha256 {checkpoint_sha256} != expected {args.checkpoint_sha256}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    configure_torch_backends()
    ManagerBasedRlEnv.seed(args.seed)
    env_cfg = load_env_cfg(PINNED_TASK, play=True)
    agent_cfg = load_rl_cfg(PINNED_TASK)
    env_cfg.scene.num_envs = 1
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.curriculum = {}
    env_cfg.events = {
        key: value
        for key, value in env_cfg.events.items()
        if key in {"reset_base", "reset_robot_joints"}
    }
    reset_base = env_cfg.events["reset_base"]
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset_base.params["velocity_range"] = {}
    twist_cfg = env_cfg.commands["twist"]
    twist_cfg.resampling_time_range = (1.0e9, 1.0e9)
    twist_cfg.heading_command = False
    twist_cfg.ranges.heading = None
    twist_cfg.rel_standing_envs = 0.0
    twist_cfg.rel_heading_envs = 0.0
    twist_cfg.init_velocity_prob = 0.0
    twist_cfg.debug_vis = False
    twist_cfg.ranges.lin_vel_y = (0.0, 0.0)

    native = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(native, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(PINNED_TASK) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    robot = env.unwrapped.scene["robot"]
    step_dt = float(env.unwrapped.step_dt)
    warmup_steps = round(args.warmup_s / step_dt)
    cases: list[dict[str, Any]] = []
    trajectory_path = args.output_dir / "trajectories.jsonl"
    try:
        with trajectory_path.open("w", encoding="utf-8") as trajectory:
            for name, forward, yaw_rate, duration_s in COMMAND_CASES:
                twist_cfg.ranges.lin_vel_x = (forward, forward)
                twist_cfg.ranges.ang_vel_z = (yaw_rate, yaw_rate)
                case: dict[str, Any] = {
                    "name": name,
                    "forward_command_mps": forward,
                    "lateral_command_mps": 0.0,
                    "yaw_command_rps": yaw_rate,
                    "duration_s": duration_s,
                    "runs": [],
                }
                for repeat in range(args.repeats):
                    obs, _ = env.reset()
                    command = env.unwrapped.command_manager.get_command("twist")[0].tolist()
                    expected = [forward, 0.0, yaw_rate]
                    if any(
                        abs(float(a) - b) > 1.0e-6
                        for a, b in zip(command, expected, strict=True)
                    ):
                        raise RuntimeError(f"sampled command {command} != expected {expected}")
                    samples: list[dict[str, float]] = []
                    failure: str | None = None
                    for step in range(round(duration_s / step_dt)):
                        with torch.inference_mode():
                            actions = policy(obs)
                        if not bool(torch.isfinite(actions).all()):
                            failure = "non-finite action"
                            break
                        obs, _reward, dones, _extras = env.step(actions)
                        pos = [float(v) for v in robot.data.root_link_pos_w[0].tolist()]
                        quat = [float(v) for v in robot.data.root_link_quat_w[0].tolist()]
                        lin = [float(v) for v in robot.data.root_link_lin_vel_b[0].tolist()]
                        ang = [float(v) for v in robot.data.root_link_ang_vel_b[0].tolist()]
                        up_z = max(-1.0, min(1.0, 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)))
                        row = {
                            "time_s": (step + 1) * step_dt,
                            "x_m": pos[0],
                            "y_m": pos[1],
                            "z_m": pos[2],
                            "forward_mps": lin[0],
                            "lateral_mps": lin[1],
                            "yaw_rate_rps": ang[2],
                            "tilt_deg": math.degrees(math.acos(up_z)),
                        }
                        trajectory.write(
                            json.dumps({"case": name, "repeat": repeat, **row}) + "\n"
                        )
                        samples.append(row)
                        if not finite(list(row.values())):
                            failure = "non-finite state"
                            break
                        if bool(dones[0].item()):
                            failure = "upstream termination"
                            break
                        if row["z_m"] < 0.15 or row["tilt_deg"] > 60.0:
                            failure = "project fall threshold"
                            break
                    summary = trial_summary(samples, min(warmup_steps, len(samples) - 1))
                    case["runs"].append(
                        {
                            "repeat": repeat,
                            "failed": failure is not None,
                            "failure": failure,
                            "samples": len(samples),
                            **summary,
                        }
                    )
                cases.append(case)
    finally:
        env.close()

    checks = evaluate(cases)
    return {
        "schema_version": "go2wm.mjlab-checkpoint-playback.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "completion_state": "complete",
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "source": {
            "repository": git(checkout, "remote", "get-url", "origin"),
            "revision": revision,
            "tracked_dirty": bool(
                git(checkout, "status", "--porcelain", "--untracked-files=no")
            ),
            "untracked_paths": len(
                git(checkout, "ls-files", "--others", "--exclude-standard").splitlines()
            ),
            "harness_sha256": sha256_file(Path(__file__).resolve()),
            "play_sha256": sha256_file(checkout / "scripts" / "play.py"),
            "go2_env_cfg_sha256": sha256_file(
                checkout / "src" / "tasks" / "velocity" / "config" / "go2" / "env_cfgs.py"
            ),
        },
        "checkpoint": {"file": checkpoint.name, "sha256": checkpoint_sha256},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mjlab": package_version("mjlab"),
            "mujoco": package_version("mujoco"),
            "mujoco_warp": package_version("mujoco-warp"),
            "rsl_rl": package_version("rsl-rl-lib"),
            "device": args.device,
        },
        "protocol": {
            "task": PINNED_TASK,
            "seed": args.seed,
            "repeats": args.repeats,
            "step_dt_s": step_dt,
            "warmup_s": args.warmup_s,
            "domain_randomization": False,
            "lateral_velocity_mps": 0.0,
        },
        "checks": checks,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-s", type=float, default=1.0)
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
            "schema_version": "go2wm.mjlab-checkpoint-playback.v1",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "completion_state": "failed",
            "verdict": "ERROR",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8"
    )
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        if sha256_file(args.output_dir / name) != expected:
            raise RuntimeError(f"post-write checksum verification failed for {name}")
    print(json.dumps({"verdict": report["verdict"], "output_dir": str(args.output_dir)}))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
