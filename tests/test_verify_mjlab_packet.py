"""End-to-end test of scripts/verify_mjlab_packet.py on a synthetic MjLab packet.

Needs GO2WM_MJLAB_GO2_XML (unitree_rl_mjlab@1425b15 go2.xml with its assets/),
MuJoCo, onnx and onnxruntime; skips otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_mjlab_packet.py"
DEPLOY_YAML = """\
joint_ids_map: [3,4,5,0,1,2,9,10,11,6,7,8,]
step_dt: 0.02
stiffness: [20, 20, 40, 20, 20, 40, 20, 20, 40, 20, 20, 40]
damping:   [ 1,  1,  2,  1,  1,  2,  1,  1,  2,  1,  1,  2]
default_joint_pos: [-0.1,0.9,-1.8, 0.1,0.9,-1.8, -0.1,0.9,-1.8, 0.1,0.9,-1.8]
actions:
  JointPositionAction:
    scale: [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
observations:
  base_ang_vel: {params: {}, scale: [1.0, 1.0, 1.0], history_length: 1}
  projected_gravity: {params: {}, scale: [1.0, 1.0, 1.0], history_length: 1}
  velocity_commands: {params: {command_name: base_velocity}, history_length: 1}
  gait_phase: {params: {period: 0.6}, scale: [1.0, 1.0], history_length: 1}
  joint_pos_rel: {params: {}, history_length: 1}
  joint_vel_rel: {params: {}, history_length: 1}
  last_action: {params: {}, history_length: 1}
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zero_policy(path: Path) -> None:
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from go2wm.sim.locomotion import MJLAB_ACTOR_OBSERVATION_NAMES, MJLAB_GO2_FLAT

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["actions"])],
        "zero",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 47])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 12])],
        [numpy_helper.from_array(np.zeros((47, 12), np.float32), "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    spec = MJLAB_GO2_FLAT
    for key, value in {
        "joint_names": ",".join(spec.joint_names),
        "joint_stiffness": ",".join(map(str, spec.stiffness)),
        "joint_damping": ",".join(map(str, spec.damping)),
        "default_joint_pos": ",".join(map(str, spec.default_joint_pos)),
        "observation_names": ",".join(MJLAB_ACTOR_OBSERVATION_NAMES),
        "action_scale": ",".join(map(str, spec.action_scale)),
    }.items():
        entry = onnx.StringStringEntryProto()
        entry.key, entry.value = key, value
        model.metadata_props.append(entry)
    onnx.save(model, str(path))


def _packet(tmp_path: Path, go2_xml: Path, *, with_meshes: bool = True) -> Path:
    from go2wm.sim.controller_handoff import (
        EXPECTED_COMMAND_ORDER,
        EXPECTED_JOINT_ORDER,
        PINNED_MODEL_XML_SHA256,
        PINNED_REPOSITORY,
        PINNED_REVISION,
        PINNED_TASK,
        SCHEMA_VERSION,
    )

    packet = tmp_path / "packet"
    (packet / "robot").mkdir(parents=True)
    shutil.copy(go2_xml, packet / "robot" / "go2.xml")
    if with_meshes:
        shutil.copytree(go2_xml.parent / "assets", packet / "robot" / "assets")
    _write_zero_policy(packet / "policy.onnx")
    (packet / "deploy.yaml").write_text(DEPLOY_YAML)
    for name in ("model_10000.pt", "env.yaml", "agent.yaml", "training-full.log"):
        (packet / name).write_text(f"fixture {name}\n")
    paths = {
        "policy_onnx": "policy.onnx",
        "checkpoint": "model_10000.pt",
        "environment_config": "env.yaml",
        "agent_config": "agent.yaml",
        "deployment_config": "deploy.yaml",
        "robot_xml": "robot/go2.xml",
        "training_log": "training-full.log",
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "controller_id": "synthetic-zero-policy",
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
        },
        "files": {k: {"path": v, "sha256": _sha256(packet / v)} for k, v in paths.items()},
    }
    (packet / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return packet


def _run(packet: Path, out: Path, *extra: str) -> tuple[int, dict]:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
    }
    if "MUJOCO_GL" not in env and sys.platform.startswith("linux"):
        env["MUJOCO_GL"] = "egl"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(packet), "--output-dir", str(out), *extra],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=600,
    )
    report = json.loads((out / "verification.json").read_text())
    return completed.returncode, report


@pytest.fixture
def go2_xml() -> Path:
    for module in ("mujoco", "onnx", "onnxruntime", "yaml"):
        pytest.importorskip(module)
    value = os.environ.get("GO2WM_MJLAB_GO2_XML")
    if not value:
        pytest.skip("set GO2WM_MJLAB_GO2_XML to unitree_rl_mjlab@1425b15 go2.xml")
    return Path(value)


def test_structural_checks_pass_on_a_well_formed_packet(tmp_path: Path, go2_xml: Path) -> None:
    code, report = _run(_packet(tmp_path, go2_xml), tmp_path / "out", "--skip-trials")
    statuses = {name: step["status"] for name, step in report["steps"].items()}
    assert statuses == {
        "handoff": "PASS",
        "meshes": "PASS",
        "metadata": "PASS",
        "deploy": "PASS",
        "inference": "PASS",
        "trials": "SKIP",
    }
    assert code == 1  # a skipped step is not a pass


def test_missing_meshes_are_caught(tmp_path: Path, go2_xml: Path) -> None:
    _, report = _run(
        _packet(tmp_path, go2_xml, with_meshes=False), tmp_path / "out", "--skip-trials"
    )
    assert report["steps"]["handoff"]["status"] == "PASS"  # SIM's verifier alone misses this
    assert report["steps"]["meshes"]["status"] == "FAIL"


def test_a_policy_that_cannot_walk_fails_the_trials(tmp_path: Path, go2_xml: Path) -> None:
    if not os.environ.get("GO2WM_SLOW_TESTS"):
        pytest.skip("set GO2WM_SLOW_TESTS=1 (runs the full G0 trials, about a minute)")
    code, report = _run(_packet(tmp_path, go2_xml), tmp_path / "out")
    assert code == 1
    assert report["steps"]["trials"]["status"] == "FAIL"
