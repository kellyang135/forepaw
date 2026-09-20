"""Velocity-command locomotion controllers for the MuJoCo Go2 (DRAFT for SIM review).

A controller turns the project command ``[forward, yaw_rate]`` (lateral fixed at
zero) into joint targets at 50 Hz. Two sources are described here:

* ``RL_SAR_ROBOT_LAB_GO2``: pretrained rl_sar ``policy/go2/robot_lab/policy.pt``
  run with explicit PD torques on the stock Menagerie Go2 scene (proposed D-020);
* ``MJLAB_GO2_FLAT``: a ``Unitree-Go2-Flat`` policy trained with Unitree RL MjLab
  ``1425b15`` and exported to ONNX, run with MuJoCo position actuators on that
  revision's own ``go2.xml`` (D-019).

Both specs come from source audits (rl_sar ``config.yaml``/``rl_sdk.cpp``; MjLab
``deploy.yaml``, ``go2_constants.py``, ``velocity_env_cfg.py``). Both policy
routes have been exercised in the draft adapter; upstream MjLab checkpoint and
deployment playback remain separate acceptance gates.
"""

from __future__ import annotations

import hashlib
import io
import math
import pickle
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

OBSERVATION_TERM_SIZES = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_command": 3,
    "gait_phase": 2,
    "joint_pos_rel": 12,
    "joint_vel_rel": 12,
    "last_action": 12,
}

Policy = Callable[[np.ndarray], np.ndarray]


class LocomotionError(RuntimeError):
    """Raised when a controller, its weights, or its robot model are inconsistent."""


@dataclass(frozen=True, slots=True)
class LocomotionSpec:
    controller_id: str
    source: str
    robot_model: Literal["menagerie_go2_scene", "mjlab_go2_xml"]
    robot_xml_sha256: str
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    observation_terms: tuple[str, ...]
    actuation: Literal["explicit_pd", "position_actuator"]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    effort_limit: tuple[float, ...]
    armature: tuple[float, ...] | None = None
    ang_vel_scale: float = 1.0
    joint_vel_scale: float = 1.0
    gait_period_s: float | None = None
    gait_standing_threshold: float = 0.1
    action_clip: float | None = None
    control_dt_s: float = 0.02
    physics_dt_s: float = 0.005
    init_base_height_m: float = 0.30
    policy_sha256: str | None = None
    # Integral trim on the body gyro's yaw rate (onboard IMU, not privileged
    # state): the policy receives yaw_cmd + integral of (yaw_cmd - measured).
    # 0 disables it. Measured need: rl_sar spins at +0.13 rad/s on zero command.
    yaw_rate_trim_gain: float = 0.0
    yaw_rate_trim_limit_rps: float = 0.6
    yaw_rate_filter_tau_s: float = 0.1

    def __post_init__(self) -> None:
        per_joint = (
            self.default_joint_pos,
            self.action_scale,
            self.stiffness,
            self.damping,
            self.effort_limit,
        )
        if len(self.joint_names) != 12 or any(len(v) != 12 for v in per_joint):
            raise LocomotionError("Go2 controllers need exactly 12 per-joint values")
        if self.armature is not None and len(self.armature) != 12:
            raise LocomotionError("armature needs 12 values")
        unknown = set(self.observation_terms) - set(OBSERVATION_TERM_SIZES)
        if unknown:
            raise LocomotionError(f"unknown observation terms {sorted(unknown)}")
        if ("gait_phase" in self.observation_terms) != (self.gait_period_s is not None):
            raise LocomotionError("gait_phase needs gait_period_s and vice versa")
        substeps = self.control_dt_s / self.physics_dt_s
        if not math.isclose(substeps, round(substeps), abs_tol=1e-9) or substeps < 1:
            raise LocomotionError("control_dt_s must be a multiple of physics_dt_s")

    @property
    def observation_size(self) -> int:
        return sum(OBSERVATION_TERM_SIZES[t] for t in self.observation_terms)

    @property
    def substeps(self) -> int:
        return round(self.control_dt_s / self.physics_dt_s)


def _per_leg(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(v) for v in values) * 4


_RL_SAR_JOINTS = tuple(
    f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")
)
_MJLAB_JOINTS = tuple(
    f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")
)

MENAGERIE_GO2_SCENE_SHA256 = "b56123ea2bf09070bf4054f0be9aa418e49041ddddce44dbbf7ab35e8b732641"
MJLAB_GO2_XML_SHA256 = "077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912"
RL_SAR_POLICY_SHA256 = "9f14cb95e74ac9e5e30954da0fc0c33eaa41e337852ebed19fcee869279ade0b"
MJLAB_SDK_TO_POLICY_JOINT_IDS = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

RL_SAR_ROBOT_LAB_GO2 = LocomotionSpec(
    controller_id="rl_sar-376d42c-go2-robot_lab",
    source=(
        "https://github.com/fan-ziqi/rl_sar@376d42c9b128f963ab08579762d5a216a976ce39"
        ":policy/go2/robot_lab/policy.pt"
    ),
    robot_model="menagerie_go2_scene",
    robot_xml_sha256=MENAGERIE_GO2_SCENE_SHA256,
    joint_names=_RL_SAR_JOINTS,
    default_joint_pos=_per_leg((0.0, 0.8, -1.5)),
    action_scale=_per_leg((0.125, 0.25, 0.25)),
    observation_terms=(
        "base_ang_vel",
        "projected_gravity",
        "velocity_command",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ),
    actuation="explicit_pd",
    stiffness=_per_leg((20.0, 20.0, 20.0)),
    damping=_per_leg((0.5, 0.5, 0.5)),
    effort_limit=_per_leg((23.5, 23.5, 23.5)),
    ang_vel_scale=0.25,
    joint_vel_scale=0.05,
    action_clip=100.0,
    init_base_height_m=0.27,
    policy_sha256=RL_SAR_POLICY_SHA256,
    yaw_rate_trim_gain=1.0,
)

# Joint order is the MjLab go2.xml order (FL, FR, RL, RR); deploy.yaml's
# joint_ids_map only reorders for the Unitree SDK. Left hips default to -0.1.
MJLAB_GO2_FLAT = LocomotionSpec(
    controller_id="unitree_rl_mjlab-1425b15-Unitree-Go2-Flat",
    source=(
        "https://github.com/unitreerobotics/unitree_rl_mjlab"
        "@1425b15f73bd4095f0df53709d7c389c3eb9e790:Unitree-Go2-Flat"
    ),
    robot_model="mjlab_go2_xml",
    robot_xml_sha256=MJLAB_GO2_XML_SHA256,
    joint_names=_MJLAB_JOINTS,
    default_joint_pos=(-0.1, 0.9, -1.8, 0.1, 0.9, -1.8, -0.1, 0.9, -1.8, 0.1, 0.9, -1.8),
    action_scale=(0.25,) * 12,
    observation_terms=(
        "base_ang_vel",
        "projected_gravity",
        "velocity_command",
        "gait_phase",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ),
    actuation="position_actuator",
    stiffness=_per_leg((20.0, 20.0, 40.0)),
    damping=_per_leg((1.0, 1.0, 2.0)),
    effort_limit=_per_leg((23.5, 23.5, 45.0)),
    armature=_per_leg((0.01, 0.01, 0.02)),
    gait_period_s=0.6,
    init_base_height_m=0.32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_mjlab_deploy_yaml(path: Path, spec: LocomotionSpec = MJLAB_GO2_FLAT) -> None:
    """Refuse a MjLab deploy.yaml whose gains, defaults, scale or observations differ."""

    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sdk_to_policy = [int(i) for i in config["joint_ids_map"]]
    expected = {
        "step_dt": spec.control_dt_s,
        "default_joint_pos": list(spec.default_joint_pos),
        "stiffness": list(spec.stiffness),
        "damping": list(spec.damping),
        "scale": list(spec.action_scale),
    }
    actual = {
        "step_dt": float(config["step_dt"]),
        "default_joint_pos": [float(v) for v in config["default_joint_pos"]],
        "stiffness": [float(v) for v in config["stiffness"]],
        "damping": [float(v) for v in config["damping"]],
        "scale": [float(v) for v in config["actions"]["JointPositionAction"]["scale"]],
    }
    for key, value in expected.items():
        if not np.allclose(actual[key], value):
            raise LocomotionError(f"deploy.yaml {key} {actual[key]} != audited {value}")
    if tuple(sdk_to_policy) != MJLAB_SDK_TO_POLICY_JOINT_IDS:
        raise LocomotionError(
            "deploy.yaml joint_ids_map "
            f"{sdk_to_policy} != audited {list(MJLAB_SDK_TO_POLICY_JOINT_IDS)}"
        )
    names = list(config["observations"])
    mapped = [
        {
            "base_ang_vel": "base_ang_vel",
            "projected_gravity": "projected_gravity",
            "velocity_commands": "velocity_command",
            "gait_phase": "gait_phase",
            "joint_pos_rel": "joint_pos_rel",
            "joint_vel_rel": "joint_vel_rel",
            "last_action": "last_action",
        }.get(n, n)
        for n in names
    ]
    if tuple(mapped) != spec.observation_terms:
        raise LocomotionError(f"deploy.yaml observations {names} differ from the audited order")
    for name, term in config["observations"].items():
        if (
            any(float(s) != 1.0 for s in term.get("scale", []))
            or term.get("history_length", 1) != 1
        ):
            raise LocomotionError(f"deploy.yaml observation {name} is scaled or stacked")
    period = config["observations"]["gait_phase"]["params"].get("period")
    if float(period) != spec.gait_period_s:
        raise LocomotionError(f"deploy.yaml gait period {period} != {spec.gait_period_s}")


MJLAB_ACTOR_OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "phase",
    "joint_pos",
    "joint_vel",
    "actions",
)


def check_mjlab_onnx_metadata(path: Path, spec: LocomotionSpec = MJLAB_GO2_FLAT) -> dict[str, str]:
    """Compare the metadata mjlab's exporter embeds in policy.onnx with the audited spec.

    mjlab 1.2.0 ``attach_metadata_to_onnx`` stores joint names, stiffness,
    damping, default joint positions, actor observation names and action scale
    as comma-separated strings. Returns the metadata; raises on any mismatch.
    """

    import onnx

    model = onnx.load(str(path), load_external_data=False)
    meta = {entry.key: entry.value for entry in model.metadata_props}
    required = (
        "joint_names",
        "joint_stiffness",
        "joint_damping",
        "default_joint_pos",
        "observation_names",
        "action_scale",
    )
    missing = [key for key in required if key not in meta]
    if missing:
        raise LocomotionError(f"{path} lacks mjlab export metadata {missing}")

    def names(key: str) -> tuple[str, ...]:
        return tuple(v.strip() for v in meta[key].split(",") if v.strip())

    def numbers(key: str) -> np.ndarray:
        return np.array([float(v) for v in names(key)])

    if names("joint_names") != spec.joint_names:
        raise LocomotionError(f"ONNX joint order {names('joint_names')} != {spec.joint_names}")
    if names("observation_names") != MJLAB_ACTOR_OBSERVATION_NAMES:
        raise LocomotionError(f"ONNX actor observations {names('observation_names')} differ")
    for key, expected in (
        ("joint_stiffness", spec.stiffness),
        ("joint_damping", spec.damping),
        ("default_joint_pos", spec.default_joint_pos),
        ("action_scale", spec.action_scale),
    ):
        values = numbers(key)
        if values.shape == (1,):  # mjlab writes a scalar when every joint shares it
            values = np.repeat(values, 12)
        if values.shape != (12,) or not np.allclose(values, expected, atol=1e-6):
            raise LocomotionError(f"ONNX {key} {values.tolist()} != audited {list(expected)}")
    return meta


class NumpyMLPPolicy:
    """ELU multilayer perceptron evaluated with NumPy (float64)."""

    def __init__(self, layers: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
        self.layers = [(np.asarray(w, np.float64), np.asarray(b, np.float64)) for w, b in layers]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs
        for weight, bias in self.layers[:-1]:
            x = weight @ x + bias
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        weight, bias = self.layers[-1]
        return weight @ x + bias


def load_rl_sar_policy(path: Path, expected_sha256: str = RL_SAR_POLICY_SHA256) -> NumpyMLPPolicy:
    """Read the rl_sar actor from its TorchScript zip without torch or code execution."""

    path = Path(path)
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise LocomotionError(f"{path} sha256 {digest} != pinned {expected_sha256}")
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
    keys = ("0", "2", "4", "6")
    if not all(isinstance(actor.get(k), dict) and "weight" in actor[k] for k in keys):
        raise LocomotionError("unexpected rl_sar actor layout")
    layers = [(actor[k]["weight"], actor[k]["bias"]) for k in keys]
    if layers[0][0].shape != (512, 45) or layers[-1][0].shape != (12, 128):
        raise LocomotionError("unexpected rl_sar actor shapes")
    return NumpyMLPPolicy(layers)


class OnnxPolicy:
    """Single-input, single-output ONNX policy run with onnxruntime on CPU."""

    def __init__(self, path: Path, observation_size: int) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise LocomotionError(f"expected one ONNX input, got {[i.name for i in inputs]}")
        shape = inputs[0].shape
        if len(shape) != 2 or (
            shape[0] not in (1, None) and not isinstance(shape[0], str)
        ):
            raise LocomotionError(f"ONNX input must be rank-2 with batch 1/dynamic, got {shape}")
        if shape[1] not in (observation_size, None) and not isinstance(shape[1], str):
            raise LocomotionError(f"ONNX input {shape} != observation size {observation_size}")
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise LocomotionError(f"expected one ONNX output, got {[o.name for o in outputs]}")
        output_shape = outputs[0].shape
        if (
            len(output_shape) != 2
            or (
                output_shape[0] not in (1, None)
                and not isinstance(output_shape[0], str)
            )
            or (
                output_shape[1] not in (12, None)
                and not isinstance(output_shape[1], str)
            )
        ):
            raise LocomotionError(f"ONNX output must be [1, 12] or dynamic, got {output_shape}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        batch = np.asarray(obs, np.float32)[None, :]
        return np.asarray(self.session.run([self.output_name], {self.input_name: batch})[0][0])


def quat_to_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


class LocomotionController:
    """Runs one policy against one MjModel/MjData pair.

    ``policy_step`` computes new joint targets once per control period;
    ``apply_substep`` writes actuator controls before every physics step.
    """

    def __init__(self, spec: LocomotionSpec, policy: Policy, model: Any, data: Any) -> None:
        import mujoco

        self.spec = spec
        self.policy = policy
        self.model = model
        self.data = data
        self.qadr = np.array(
            [
                model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                for j in spec.joint_names
            ]
        )
        self.vadr = np.array(
            [
                model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                for j in spec.joint_names
            ]
        )
        actuator_names = (
            [j.replace("_joint", "") for j in spec.joint_names]
            if spec.actuation == "explicit_pd"
            else list(spec.joint_names)
        )
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
        if min(ids) < 0:
            missing = [n for n, i in zip(actuator_names, ids, strict=True) if i < 0]
            raise LocomotionError(f"robot model lacks actuators {missing}")
        self.actuator_ids = np.array(ids)
        self.default = np.array(spec.default_joint_pos)
        self.scale = np.array(spec.action_scale)
        self.kp = np.array(spec.stiffness)
        self.kd = np.array(spec.damping)
        self.limit = np.array(spec.effort_limit)
        self.last_action = np.zeros(12)
        self.target = self.default.copy()
        self.policy_steps = 0
        self.yaw_rate_filtered = 0.0
        self.yaw_trim = 0.0
        self.last_policy_command = np.zeros(3)

    def reset(self) -> None:
        self.last_action = np.zeros(12)
        self.target = self.default.copy()
        self.policy_steps = 0
        self.yaw_rate_filtered = 0.0
        self.yaw_trim = 0.0
        self.last_policy_command = np.zeros(3)

    def observation(self, command: np.ndarray) -> np.ndarray:
        rotation = quat_to_rotation(self.data.qpos[3:7])
        parts: list[np.ndarray] = []
        for term in self.spec.observation_terms:
            if term == "base_ang_vel":
                parts.append(np.asarray(self.data.qvel[3:6]) * self.spec.ang_vel_scale)
            elif term == "projected_gravity":
                parts.append(rotation.T @ np.array([0.0, 0.0, -1.0]))
            elif term == "velocity_command":
                parts.append(np.asarray(command, np.float64))
            elif term == "gait_phase":
                assert self.spec.gait_period_s is not None
                if np.linalg.norm(command) < self.spec.gait_standing_threshold:
                    parts.append(np.zeros(2))
                else:
                    t = self.policy_steps * self.spec.control_dt_s
                    phase = (t % self.spec.gait_period_s) / self.spec.gait_period_s
                    parts.append(
                        np.array([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)])
                    )
            elif term == "joint_pos_rel":
                parts.append(np.asarray(self.data.qpos[self.qadr]) - self.default)
            elif term == "joint_vel_rel":
                parts.append(np.asarray(self.data.qvel[self.vadr]) * self.spec.joint_vel_scale)
            elif term == "last_action":
                parts.append(self.last_action)
        obs = np.concatenate(parts)
        if obs.shape != (self.spec.observation_size,):
            raise LocomotionError(f"observation shape {obs.shape} != {self.spec.observation_size}")
        return obs

    def policy_step(self, forward_mps: float, yaw_rate_rps: float) -> np.ndarray:
        spec = self.spec
        if spec.yaw_rate_trim_gain > 0:
            gyro_z = float(self.data.qvel[5])
            alpha = min(1.0, spec.control_dt_s / spec.yaw_rate_filter_tau_s)
            self.yaw_rate_filtered += alpha * (gyro_z - self.yaw_rate_filtered)
            error = yaw_rate_rps - self.yaw_rate_filtered
            self.yaw_trim = float(
                np.clip(
                    self.yaw_trim + spec.yaw_rate_trim_gain * error * spec.control_dt_s,
                    -spec.yaw_rate_trim_limit_rps,
                    spec.yaw_rate_trim_limit_rps,
                )
            )
        command = np.array([forward_mps, 0.0, yaw_rate_rps + self.yaw_trim])
        self.last_policy_command = command
        action = np.asarray(self.policy(self.observation(command)), np.float64).reshape(-1)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise LocomotionError(f"policy returned invalid action {action}")
        if self.spec.action_clip is not None:
            action = np.clip(action, -self.spec.action_clip, self.spec.action_clip)
        self.last_action = action
        self.target = self.default + action * self.scale
        self.policy_steps += 1
        return action

    def apply_substep(self) -> None:
        if self.spec.actuation == "position_actuator":
            self.data.ctrl[self.actuator_ids] = self.target
            return
        q = np.asarray(self.data.qpos[self.qadr])
        dq = np.asarray(self.data.qvel[self.vadr])
        torque = self.kp * (self.target - q) - self.kd * dq
        self.data.ctrl[self.actuator_ids] = np.clip(torque, -self.limit, self.limit)


class HoldPosePolicy:
    """Zero-action policy: holds the default pose. For tests and settling checks only."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(12)
