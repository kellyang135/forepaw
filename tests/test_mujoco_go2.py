"""Contract tests for the draft MuJoCo Go2 adapter and locomotion specs.

MuJoCo tests need the pinned Menagerie Go2 already cached (no downloads here):
set GO2WM_MENAGERIE_CACHE or use /tmp/go2wm-menagerie-cache. Rendering needs a
working OpenGL backend (e.g. MUJOCO_GL=egl on Linux). Policy-dependent tests
need GO2WM_RL_SAR_POLICY; the MjLab path needs GO2WM_MJLAB_GO2_XML.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from go2wm.contracts import ActionCommand, ObjectState, Pose2D, ResetRequest  # noqa: E402
from go2wm.sim.locomotion import (  # noqa: E402
    MJLAB_GO2_FLAT,
    RL_SAR_ROBOT_LAB_GO2,
    LocomotionError,
    check_mjlab_deploy_yaml,
)

AUDITED_DEPLOY_YAML = """\
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
  velocity_commands:
    params: {command_name: base_velocity}
    scale: [1.0, 1.0, 1.0]
    history_length: 1
  gait_phase: {params: {period: 0.6}, scale: [1.0, 1.0], history_length: 1}
  joint_pos_rel: {params: {}, history_length: 1}
  joint_vel_rel: {params: {}, history_length: 1}
  last_action: {params: {}, history_length: 1}
"""


def test_controller_specs_match_audited_interfaces() -> None:
    assert RL_SAR_ROBOT_LAB_GO2.observation_size == 45
    assert MJLAB_GO2_FLAT.observation_size == 47
    assert RL_SAR_ROBOT_LAB_GO2.substeps == MJLAB_GO2_FLAT.substeps == 4
    assert RL_SAR_ROBOT_LAB_GO2.joint_names[0] == "FR_hip_joint"
    assert MJLAB_GO2_FLAT.joint_names[0] == "FL_hip_joint"
    assert sorted(RL_SAR_ROBOT_LAB_GO2.joint_names) == sorted(MJLAB_GO2_FLAT.joint_names)
    for spec in (RL_SAR_ROBOT_LAB_GO2, MJLAB_GO2_FLAT):
        defaults = dict(zip(spec.joint_names, spec.default_joint_pos, strict=True))
        if spec is MJLAB_GO2_FLAT:
            assert defaults["FL_hip_joint"] == -0.1 and defaults["FR_hip_joint"] == 0.1


def test_mjlab_deploy_yaml_check(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    good = tmp_path / "deploy.yaml"
    good.write_text(AUDITED_DEPLOY_YAML)
    check_mjlab_deploy_yaml(good)
    for old, new in (
        ("period: 0.6", "period: 0.8"),
        ("stiffness: [20, 20, 40", "stiffness: [25, 20, 40"),
        ("  gait_phase:", "  gait_phaseX:"),
        (
            "joint_ids_map: [3,4,5,0,1,2,9,10,11,6,7,8,]",
            "joint_ids_map: [0,1,2,3,4,5,6,7,8,9,10,11]",
        ),
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text(AUDITED_DEPLOY_YAML.replace(old, new))
        with pytest.raises((LocomotionError, KeyError)):
            check_mjlab_deploy_yaml(bad)


# --------------------------------------------------------------------- MuJoCo
def _menagerie_scene() -> Path:
    pytest.importorskip("mujoco")
    cache = Path(os.environ.get("GO2WM_MENAGERIE_CACHE", "/tmp/go2wm-menagerie-cache"))
    scene = cache / "models" / "unitree_go2-98d14ab27a56c362" / "scene.xml"
    if not scene.is_file():
        pytest.skip(f"pinned Menagerie Go2 not cached at {scene}")
    return scene


def _simulator(policy=None, **config_kwargs):
    from go2wm.sim.locomotion import HoldPosePolicy
    from go2wm.sim.mujoco_go2 import MujocoGo2Config, MujocoGo2Simulator

    sim = MujocoGo2Simulator(
        RL_SAR_ROBOT_LAB_GO2,
        policy or HoldPosePolicy(),
        _menagerie_scene(),
        MujocoGo2Config(**config_kwargs),
    )
    try:
        sim._mj.Renderer(sim.model, height=8, width=8).close()
    except Exception as error:  # pragma: no cover - platform dependent
        pytest.skip(f"no OpenGL renderer: {error}")
    return sim


def _objects() -> tuple[ObjectState, ...]:
    return (
        ObjectState("box_light", "blue", Pose2D(0.8, 0.6, 0.0), True),
        ObjectState("box_heavy", "red", Pose2D(0.8, -0.6, 0.3), False),
    )


def test_adapter_satisfies_protocol_and_block_contract() -> None:
    from go2wm.sim import SimulatorAdapter

    sim = _simulator()
    assert isinstance(sim, SimulatorAdapter)
    report = sim.reset(ResetRequest("ep-1", 7, Pose2D(-0.5, 0.0, 0.2), _objects()))
    assert report.settled
    assert [o.object_id for o in report.final_labels.objects] == ["box_light", "box_heavy"]
    assert report.final_observation.width_px == report.final_observation.height_px == 224
    transition = sim.execute_block(ActionCommand(5.0, -9.0))
    assert transition.requested_action.forward_velocity_mps == 5.0
    assert transition.action.forward_velocity_mps == sim.config.max_forward_velocity_mps
    assert transition.action.yaw_rate_rps == -sim.config.max_abs_yaw_rate_rps
    assert len(transition.physics_samples) == 100
    assert math.isclose(
        transition.end_observation.sim_time_s - transition.start_observation.sim_time_s,
        0.5,
        abs_tol=1e-12,
    )
    assert transition.start_observation.sim_time_s == pytest.approx(report.settle_duration_s)
    sim.close()


def test_reset_is_reproducible_per_seed_and_differs_across_seeds() -> None:
    sim = _simulator()
    frames = []
    for seed in (3, 3, 4):
        report = sim.reset(ResetRequest(f"ep-{seed}", seed, Pose2D(0.0, 0.0, 0.0), _objects()))
        transition = sim.execute_block(ActionCommand(0.3, 0.0))
        frames.append((report.final_observation.sha256, transition.end_observation.sha256))
    assert frames[0] == frames[1]
    assert frames[0] != frames[2]
    sim.close()


def test_contacts_are_attributed_to_the_touched_box() -> None:
    sim = _simulator()
    report = sim.reset(ResetRequest("ep-c", 1, Pose2D(0.0, 0.0, 0.0), _objects()))
    robot = report.final_labels.robot_pose
    qadr = sim.built.box_joint_qposadr[1]
    ahead = 0.30 + sim.config.scene.box_half_extent_m  # head reaches ~0.33 m ahead of the base
    sim.data.qpos[qadr : qadr + 3] = [
        robot.x_m + ahead * math.cos(robot.yaw_rad),
        robot.y_m + ahead * math.sin(robot.yaw_rad),
        sim.config.scene.box_half_extent_m,
    ]
    sim._mj.mj_forward(sim.model, sim.data)
    transition = sim.execute_block(ActionCommand(0.0, 0.0))
    assert "box_heavy" in transition.events.contacted_object_ids
    assert "box_light" not in transition.events.contacted_object_ids
    assert transition.events.max_normal_impulse_ns > 0
    sim.close()


@pytest.mark.parametrize(
    ("objects", "message"),
    [
        ((ObjectState("b", "green", Pose2D(1, 0, 0), True),), "unknown appearance"),
        ((ObjectState("b", "red", Pose2D(1, 0, 0), True),), "resistant"),
        ((ObjectState("b", "blue", Pose2D(0.3, 0, 0), True),), "from the robot"),
        (
            (
                ObjectState("a", "blue", Pose2D(1.0, 0, 0), True),
                ObjectState("b", "red", Pose2D(1.2, 0, 0), False),
            ),
            "overlap",
        ),
    ],
)
def test_invalid_reset_requests_are_rejected(objects, message) -> None:
    sim = _simulator()
    with pytest.raises(ValueError, match=message):
        sim.reset(ResetRequest("ep-bad", 1, Pose2D(0, 0, 0), objects))
    sim.close()


def test_execute_before_reset_fails() -> None:
    sim = _simulator()
    with pytest.raises(RuntimeError, match="reset"):
        sim.execute_block(ActionCommand(0.1, 0.0))


def test_rl_sar_policy_walks_and_holds_heading() -> None:
    path = os.environ.get("GO2WM_RL_SAR_POLICY")
    if not path:
        pytest.skip("set GO2WM_RL_SAR_POLICY to the pinned rl_sar robot_lab policy.pt")
    from go2wm.sim.locomotion import load_rl_sar_policy

    sim = _simulator(load_rl_sar_policy(Path(path)))
    report = sim.reset(ResetRequest("ep-w", 2, Pose2D(-1.0, 0.0, 0.0), ()))
    start = report.final_labels.robot_pose
    for _ in range(6):
        transition = sim.execute_block(ActionCommand(0.6, 0.0))
        assert not transition.events.fell
    end = transition.end_labels.robot_pose
    assert end.x_m - start.x_m > 1.2
    assert abs(end.yaw_rad - start.yaw_rad) < 0.15
    sim.close()


def test_mjlab_scene_compiles_with_position_actuators() -> None:
    xml = os.environ.get("GO2WM_MJLAB_GO2_XML")
    if not xml:
        pytest.skip("set GO2WM_MJLAB_GO2_XML to unitree_rl_mjlab@1425b15 go2.xml")
    pytest.importorskip("mujoco")
    from go2wm.sim.locomotion import HoldPosePolicy
    from go2wm.sim.mujoco_go2 import MujocoGo2Simulator

    sim = MujocoGo2Simulator(MJLAB_GO2_FLAT, HoldPosePolicy(), Path(xml))
    assert sim.model.nu == 12
    assert sim.model.opt.timestep == 0.005
    assert sim.controller.observation(np.array([0.3, 0.0, 0.0])).shape == (47,)
    sim.close()


def test_collector_writes_mujoco_episodes_the_strict_loader_accepts(tmp_path: Path) -> None:
    from go2wm.contracts import DatasetSplit, Goal2D
    from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest, write_episode
    from go2wm.learning.dataset import load_dataset

    sim = _simulator()
    collector = EpisodeCollector(sim, CollectionConfig())
    episode = collector.collect(
        EpisodeRequest(
            reset=ResetRequest("mj-ep-1", 11, Pose2D(-0.6, 0.0, 0.0), _objects()),
            split=DatasetSplit.TRAIN,
            goal=Goal2D(1.0, 0.0, 0.2),
            run_id="mujoco-adapter-test",
            metadata={"controller_id": sim.controller_id},
        ),
        (ActionCommand(0.3, 0.0), ActionCommand(0.0, 0.6), ActionCommand(0.0, 0.0)),
    )
    write_episode(tmp_path, episode)
    loaded = load_dataset(tmp_path)
    assert len(loaded.episodes) == 1
    record = loaded.episodes[0]
    assert record.scene_id == sim.scene_config.scene_id
    assert len(record.blocks) == 3
    assert (
        record.blocks[0].transition.start_observation.rgb
        == episode.blocks[0].transition.start_observation.rgb
    )
    sim.close()


def _onnx_with_metadata(
    path: Path,
    overrides: dict[str, str] | None = None,
    *,
    input_size: int = 47,
    output_size: int = 12,
) -> Path:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from go2wm.sim.locomotion import MJLAB_ACTOR_OBSERVATION_NAMES

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["obs", "W"], ["actions"])],
        "zero",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, input_size])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, output_size])],
        [numpy_helper.from_array(np.zeros((input_size, output_size), np.float32), "W")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    meta = {
        "joint_names": ",".join(MJLAB_GO2_FLAT.joint_names),
        "joint_stiffness": ",".join(str(v) for v in MJLAB_GO2_FLAT.stiffness),
        "joint_damping": ",".join(str(v) for v in MJLAB_GO2_FLAT.damping),
        "default_joint_pos": ",".join(str(v) for v in MJLAB_GO2_FLAT.default_joint_pos),
        "observation_names": ",".join(MJLAB_ACTOR_OBSERVATION_NAMES),
        "action_scale": ",".join(str(v) for v in MJLAB_GO2_FLAT.action_scale),
        **(overrides or {}),
    }
    for key, value in meta.items():
        entry = onnx.StringStringEntryProto()
        entry.key, entry.value = key, value
        model.metadata_props.append(entry)
    onnx.save(model, str(path))
    return path


def test_mjlab_onnx_metadata_check(tmp_path: Path) -> None:
    from go2wm.sim.locomotion import check_mjlab_onnx_metadata

    check_mjlab_onnx_metadata(_onnx_with_metadata(tmp_path / "good.onnx"))
    check_mjlab_onnx_metadata(
        _onnx_with_metadata(tmp_path / "scalar.onnx", {"action_scale": "0.25"})
    )
    swapped = list(MJLAB_GO2_FLAT.joint_names)
    swapped[0], swapped[3] = swapped[3], swapped[0]
    for name, override in (
        ("order", {"joint_names": ",".join(swapped)}),
        ("gain", {"joint_stiffness": ",".join(["20"] * 12)}),
        ("obs", {"observation_names": "base_ang_vel,projected_gravity,command"}),
        ("scale", {"action_scale": "0.5"}),
    ):
        with pytest.raises(LocomotionError):
            check_mjlab_onnx_metadata(_onnx_with_metadata(tmp_path / f"{name}.onnx", override))


@pytest.mark.parametrize(("input_size", "output_size"), [(46, 12), (47, 11)])
def test_onnx_policy_rejects_wrong_io_shapes(
    tmp_path: Path, input_size: int, output_size: int
) -> None:
    pytest.importorskip("onnxruntime")
    from go2wm.sim.locomotion import OnnxPolicy

    path = _onnx_with_metadata(
        tmp_path / f"wrong-{input_size}-{output_size}.onnx",
        input_size=input_size,
        output_size=output_size,
    )
    with pytest.raises(LocomotionError):
        OnnxPolicy(path, MJLAB_GO2_FLAT.observation_size)
