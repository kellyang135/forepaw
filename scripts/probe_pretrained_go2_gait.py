#!/usr/bin/env python3
"""G0-style feasibility probe: a pretrained Go2 velocity policy on the pinned Menagerie Go2.

Candidate controller: ``policy/go2/robot_lab/policy.pt`` from fan-ziqi/rl_sar
(Apache-2.0), trained in Isaac Lab with robot_lab (Apache-2.0).  Its interface,
taken from ``policy/go2/robot_lab/config.yaml`` and ``rl_sdk.cpp`` in rl_sar:

* 50 Hz policy (dt 0.005 s x decimation 4), 45 inputs:
  body angular velocity x 0.25, projected gravity, command ``[vx, vy, wz]``,
  joint position - default, joint velocity x 0.05, last action;
* joint order FR, FL, RR, RL x (hip, thigh, calf); default pose (0, 0.8, -1.5);
* output: target = default + action x (0.125, 0.25, 0.25) per leg;
  torque = 20 (target - q) - 0.5 dq, clamped to +-23.5 N m.

The probe loads the TorchScript archive with a restricted unpickler (no torch
needed, no code execution), runs it on the *unmodified* Menagerie
``unitree_go2/scene.xml`` whose SHA-256 Person A verified, and measures stand,
walk, turn, command-step response, and pushes into equal-geometry light and
heavy boxes.  It writes ``report.json`` and ``checksums.sha256`` into a new run
directory.  It is feasibility evidence for choosing a controller, not the G0
gate: no project adapter, scene, camera, or front marker is involved yet.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

PINNED_SCENE_SHA256 = "b56123ea2bf09070bf4054f0be9aa418e49041ddddce44dbbf7ab35e8b732641"
PINNED_POLICY_SHA256 = "9f14cb95e74ac9e5e30954da0fc0c33eaa41e337852ebed19fcee869279ade0b"
RL_SAR_COMMIT = "376d42c9b128f963ab08579762d5a216a976ce39"

POLICY_DT = 0.02
LEGS = ("FR", "FL", "RR", "RL")
PARTS = ("hip", "thigh", "calf")
DEFAULT_POSE = np.tile([0.0, 0.8, -1.5], 4)
ACTION_SCALE = np.tile([0.125, 0.25, 0.25], 4)
KP, KD, TORQUE_LIMIT = 20.0, 0.5, 23.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torchscript_mlp(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Read the actor weights from a TorchScript zip without importing torch.

    Only tensor storages and plain containers are reconstructed; any other
    pickled global raises, so the archive cannot execute code.
    """

    archive = zipfile.ZipFile(path)
    root = archive.namelist()[0].split("/")[0]

    class Storage:
        def __init__(self, key: str) -> None:
            self.key = key

    def rebuild(storage: Storage, offset: int, size: tuple, stride: tuple, *_: Any) -> np.ndarray:
        buffer = np.frombuffer(archive.read(f"{root}/data/{storage.key}"), dtype=np.float32)
        return np.lib.stride_tricks.as_strided(
            buffer[offset:], shape=size, strides=[s * 4 for s in stride]
        ).copy()

    class Module(dict):
        def __setstate__(self, state: Any) -> None:
            if isinstance(state, dict):
                self.update(state)

    class Restricted(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return rebuild
            if (module, name) == ("torch._utils", "_rebuild_parameter"):
                return lambda data, *_: data
            if (module, name) == ("collections", "OrderedDict"):
                import collections

                return collections.OrderedDict
            if module.startswith("__torch__"):
                return Module
            if (module, name) == ("torch", "FloatStorage"):
                return "FloatStorage"
            raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")

        def persistent_load(self, pid: Any) -> Storage:
            _, storage_type, key, *_ = pid
            if storage_type != "FloatStorage":
                raise pickle.UnpicklingError(f"unexpected storage {storage_type}")
            return Storage(key)

    obj = Restricted(io.BytesIO(archive.read(f"{root}/data.pkl"))).load()
    actor = obj["actor"]
    layers = {
        k: {"weight": v["weight"], "bias": v["bias"]}
        for k, v in actor.items()
        if isinstance(v, dict) and "weight" in v
    }
    if sorted(layers) != ["0", "2", "4", "6"]:
        raise ValueError(f"unexpected actor layout {sorted(layers)}")
    if layers["0"]["weight"].shape != (512, 45) or layers["6"]["weight"].shape != (12, 128):
        raise ValueError("unexpected actor shapes")
    return layers


class Policy:
    def __init__(self, layers: dict[str, dict[str, np.ndarray]]) -> None:
        self.layers = [(layers[k]["weight"], layers[k]["bias"]) for k in ("0", "2", "4", "6")]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs
        for weight, bias in self.layers[:-1]:
            x = weight @ x + bias
            x = np.where(x > 0, x, np.expm1(x))
        weight, bias = self.layers[-1]
        return weight @ x + bias


class Go2Sim:
    """Menagerie Go2 + optional box, driven by the pretrained policy at 50 Hz."""

    def __init__(
        self,
        scene_xml: Path,
        policy: Policy,
        *,
        physics_dt: float,
        box: dict[str, float] | None = None,
        seed: int = 0,
        init_noise: float = 0.0,
    ) -> None:
        import mujoco

        self.mujoco = mujoco
        spec = mujoco.MjSpec.from_file(str(scene_xml))
        home = mujoco.MjModel.from_xml_path(str(scene_xml)).key_qpos[0].copy()
        for key in list(spec.keys):
            spec.delete(key)
        if box is not None:
            body = spec.worldbody.add_body(name="box", pos=[box["x"], box["y"], box["half"]])
            body.add_freejoint()
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[box["half"]] * 3,
                mass=box["mass"],
                rgba=[0.2, 0.4, 0.9, 1],
            )
        self.model = spec.compile()
        self.model.opt.timestep = physics_dt
        self.substeps = round(POLICY_DT / physics_dt)
        if not math.isclose(self.substeps * physics_dt, POLICY_DT, abs_tol=1e-12):
            raise ValueError("physics_dt must divide the 0.02 s policy period")
        self.data = mujoco.MjData(self.model)
        joints = [f"{leg}_{part}_joint" for leg in LEGS for part in PARTS]
        self.qadr = np.array([self.model.jnt_qposadr[self.model.joint(j).id] for j in joints])
        self.vadr = np.array([self.model.jnt_dofadr[self.model.joint(j).id] for j in joints])
        self.act = np.array([self.model.actuator(j.replace("_joint", "")).id for j in joints])
        rng = np.random.default_rng(seed)
        self.data.qpos[: len(home)] = home
        self.data.qpos[self.qadr] = DEFAULT_POSE + rng.uniform(-init_noise, init_noise, 12)
        yaw = rng.uniform(-init_noise, init_noise) * 2
        self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.policy = policy
        self.last_action = np.zeros(12)
        self.box_qadr = self.model.jnt_qposadr[self.model.body("box").jntadr[0]] if box else None

    def rotation(self) -> np.ndarray:
        matrix = np.zeros(9)
        self.mujoco.mju_quat2Mat(matrix, self.data.qpos[3:7])
        return matrix.reshape(3, 3)

    def policy_step(self, forward: float, yaw_rate: float) -> None:
        d = self.data
        rot = self.rotation()
        obs = np.concatenate(
            [
                d.qvel[3:6] * 0.25,
                rot.T @ np.array([0.0, 0.0, -1.0]),
                [forward, 0.0, yaw_rate],
                d.qpos[self.qadr] - DEFAULT_POSE,
                d.qvel[self.vadr] * 0.05,
                self.last_action,
            ]
        )
        self.last_action = np.clip(self.policy(obs), -100, 100)
        target = DEFAULT_POSE + self.last_action * ACTION_SCALE
        for _ in range(self.substeps):
            torque = KP * (target - d.qpos[self.qadr]) - KD * d.qvel[self.vadr]
            d.ctrl[self.act] = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
            self.mujoco.mj_step(self.model, d)

    def state(self) -> dict[str, float]:
        d = self.data
        rot = self.rotation()
        body_vel = rot.T @ d.qvel[0:3]
        tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1, 1))))
        out = {
            "t": float(d.time),
            "x": float(d.qpos[0]),
            "y": float(d.qpos[1]),
            "z": float(d.qpos[2]),
            "yaw": math.atan2(rot[1, 0], rot[0, 0]),
            "vx": float(body_vel[0]),
            "vy": float(body_vel[1]),
            "wz": float(d.qvel[5]),
            "tilt_deg": tilt,
            "finite": bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()),
        }
        if self.box_qadr is not None:
            out["box_x"] = float(d.qpos[self.box_qadr])
            out["box_y"] = float(d.qpos[self.box_qadr + 1])
        return out


FALL_Z, FALL_TILT = 0.18, 45.0


def run_schedule(sim: Go2Sim, schedule: list[tuple[float, float, float]]) -> list[dict[str, float]]:
    """``schedule`` is ``[(duration_s, forward, yaw_rate), ...]``; logs every policy step."""

    log = []
    for duration, forward, yaw_rate in schedule:
        for _ in range(round(duration / POLICY_DT)):
            sim.policy_step(forward, yaw_rate)
            state = sim.state()
            state["cmd_vx"], state["cmd_wz"] = forward, yaw_rate
            log.append(state)
            if not state["finite"]:
                return log
    return log


def fell(log: list[dict[str, float]]) -> bool:
    return any(s["z"] < FALL_Z or s["tilt_deg"] > FALL_TILT or not s["finite"] for s in log)


def window(log: list[dict[str, float]], start: float, end: float) -> list[dict[str, float]]:
    return [s for s in log if start <= s["t"] < end]


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def locomotion_trials(scene: Path, policy: Policy, physics_dt: float, repeats: int) -> dict:
    results: dict[str, Any] = {}
    stand = []
    for seed in range(repeats):
        sim = Go2Sim(scene, policy, physics_dt=physics_dt, seed=seed, init_noise=0.05)
        log = run_schedule(sim, [(10.0, 0.0, 0.0)])
        drift = math.hypot(log[-1]["x"] - log[0]["x"], log[-1]["y"] - log[0]["y"])
        stand.append(
            {
                "fell": fell(log),
                "xy_drift_m": round(drift, 3),
                "min_z": round(min(s["z"] for s in log), 3),
            }
        )
    results["stand_10s"] = stand

    commands = {
        "forward_0.3": (0.3, 0.0),
        "forward_0.6": (0.6, 0.0),
        "forward_1.0": (1.0, 0.0),
        "turn_left_0.8": (0.0, 0.8),
        "turn_right_0.8": (0.0, -0.8),
        "arc_0.4_left_0.6": (0.4, 0.6),
        "arc_0.4_right_0.6": (0.4, -0.6),
    }
    for name, (vx, wz) in commands.items():
        trials = []
        for seed in range(repeats):
            sim = Go2Sim(scene, policy, physics_dt=physics_dt, seed=100 + seed, init_noise=0.05)
            log = run_schedule(sim, [(1.0, 0.0, 0.0), (6.0, vx, wz)])
            steady = window(log, 3.0, 7.0)
            trials.append(
                {
                    "fell": fell(log),
                    "vx": round(mean([s["vx"] for s in steady]), 3),
                    "vy": round(mean([s["vy"] for s in steady]), 3),
                    "wz": round(mean([s["wz"] for s in steady]), 3),
                    "max_tilt_deg": round(max(s["tilt_deg"] for s in log), 1),
                }
            )
        results[name] = {"command": [vx, wz], "trials": trials}

    # Response inside one 0.5 s project block after a command step.
    steps = []
    for seed in range(repeats):
        sim = Go2Sim(scene, policy, physics_dt=physics_dt, seed=200 + seed, init_noise=0.05)
        log = run_schedule(
            sim, [(2.0, 0.0, 0.0), (2.0, 0.6, 0.0), (2.0, 0.6, 0.8), (2.0, 0.0, 0.0)]
        )
        t0 = 2.0
        rise = next((s["t"] - t0 for s in log if s["t"] > t0 and s["vx"] > 0.9 * 0.55), None)
        steps.append(
            {
                "fell": fell(log),
                "vx_first_block": round(mean([s["vx"] for s in window(log, 2.0, 2.5)]), 3),
                "vx_second_block": round(mean([s["vx"] for s in window(log, 2.5, 3.0)]), 3),
                "time_to_90pct_s": None if rise is None else round(rise, 2),
                "wz_block_after_turn_cmd": round(mean([s["wz"] for s in window(log, 4.0, 4.5)]), 3),
                "vx_last_block_after_stop": round(
                    mean([s["vx"] for s in window(log, 7.5, 8.0)]), 3
                ),
            }
        )
    results["command_step_response"] = steps
    return results


def aimed_push(sim: Go2Sim, vx: float, seconds: float, gain: float = 2.0) -> list[dict[str, float]]:
    """Walk at ``vx`` while steering toward the box centre (privileged, collection-only).

    Open-loop straight commands miss the box because the gait drifts about
    0.1 rad/s in yaw; a scripted collector or the replanning controller corrects
    for that, so the push test does too.
    """

    log = run_schedule(sim, [(1.0, 0.0, 0.0)])
    for _ in range(round(seconds / POLICY_DT)):
        state = sim.state()
        bearing = math.atan2(state["box_y"] - state["y"], state["box_x"] - state["x"])
        error = (bearing - state["yaw"] + math.pi) % (2 * math.pi) - math.pi
        wz = float(np.clip(gain * error, -1.0, 1.0))
        sim.policy_step(vx, wz)
        state = sim.state()
        state["cmd_vx"], state["cmd_wz"] = vx, wz
        log.append(state)
        if not state["finite"]:
            break
    return log


def box_trials(scene: Path, policy: Policy, physics_dt: float, repeats: int, vx: float) -> dict:
    """Equal-geometry cubes; only mass differs.  Several approach angles per box."""

    results: dict[str, Any] = {}
    angles = np.linspace(-30, 30, repeats)
    for half in (0.15, 0.20):
        for label, mass in (("light", 1.0), ("heavy", 20.0)):
            trials = []
            for i, angle in enumerate(angles):
                rad = math.radians(float(angle))
                box = {
                    "x": 1.0 * math.cos(rad),
                    "y": 1.0 * math.sin(rad),
                    "half": half,
                    "mass": mass,
                }
                sim = Go2Sim(
                    scene, policy, physics_dt=physics_dt, box=box, seed=300 + i, init_noise=0.02
                )
                log = aimed_push(sim, vx, 8.0)
                first, last = log[0], log[-1]
                moved = [
                    s
                    for s in log
                    if math.hypot(s["box_x"] - first["box_x"], s["box_y"] - first["box_y"]) > 0.01
                ]
                trials.append(
                    {
                        "approach_deg": round(float(angle), 1),
                        "fell": fell(log),
                        "box_displacement_m": round(
                            math.hypot(
                                last["box_x"] - first["box_x"], last["box_y"] - first["box_y"]
                            ),
                            3,
                        ),
                        "robot_path_m": round(
                            float(
                                sum(
                                    math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                                    for a, b in pairwise(log)
                                )
                            ),
                            3,
                        ),
                        "first_box_motion_s": round(moved[0]["t"], 2) if moved else None,
                        "max_tilt_deg": round(max(s["tilt_deg"] for s in log), 1),
                        "min_base_z_m": round(min(s["z"] for s in log), 3),
                    }
                )
            results[f"cube_{2 * half:.2f}m_{label}_{mass:g}kg"] = trials
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--policy", type=Path, required=True, help="rl_sar policy/go2/robot_lab/policy.pt"
    )
    parser.add_argument(
        "--scene", type=Path, help="Menagerie unitree_go2/scene.xml (default: the pinned package)"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/go2wm-menagerie-cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--physics-dt", type=float, nargs="+", default=[0.005, 0.002])
    parser.add_argument("--push-speed", type=float, default=0.5)
    args = parser.parse_args(argv)

    import mujoco

    scene = args.scene
    if scene is None:
        import mujoco_menagerie as menagerie  # same resolution as scripts/verify_go2_model.py

        scene = Path(menagerie.get("unitree_go2").xml(cache=menagerie.Cache(args.cache_dir)))
    checks = {
        "scene_sha256": sha256(scene),
        "policy_sha256": sha256(args.policy),
    }
    problems = []
    if checks["scene_sha256"] != PINNED_SCENE_SHA256:
        problems.append("scene.xml differs from the Menagerie file Person A verified")
    if checks["policy_sha256"] != PINNED_POLICY_SHA256:
        problems.append(f"policy.pt differs from rl_sar {RL_SAR_COMMIT} robot_lab/policy.pt")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 2

    out = args.output_dir
    if out.exists():
        print(f"refusing to overwrite {out}", file=sys.stderr)
        return 2
    policy = Policy(load_torchscript_mlp(args.policy))
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "go2wm.pretrained-gait-probe.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": (
            "feasibility of a pretrained controller on the unmodified Menagerie Go2; "
            "not the G0 gate (no project adapter, task scene, camera, or marker)"
        ),
        "candidate": {
            "source": "https://github.com/fan-ziqi/rl_sar",
            "commit": RL_SAR_COMMIT,
            "file": "policy/go2/robot_lab/policy.pt",
            "license": "Apache-2.0 (rl_sar); trained with robot_lab (Apache-2.0) in Isaac Lab",
            "interface": {
                "policy_hz": 50,
                "obs": 45,
                "command": "[vx, vy=0, wz]",
                "action": "joint-position offsets x (0.125, 0.25, 0.25)",
                "pd": {"kp": KP, "kd": KD, "torque_limit_nm": TORQUE_LIMIT},
                "default_pose": [0.0, 0.8, -1.5],
            },
        },
        "inputs": checks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
        },
        "fall_rule": {"base_z_below_m": FALL_Z, "tilt_above_deg": FALL_TILT},
        "results": {},
    }
    for dt in args.physics_dt:
        key = f"physics_dt_{dt:g}"
        report["results"][key] = {
            "locomotion": locomotion_trials(scene, policy, dt, args.repeats),
            "boxes": box_trials(scene, policy, dt, args.repeats, args.push_speed),
        }
    report["wall_seconds"] = round(time.perf_counter() - started, 1)

    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "checksums.sha256").write_text(
        f"{sha256(out / 'report.json')}  report.json\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(out), "wall_seconds": report["wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
