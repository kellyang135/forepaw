"""Push-versus-detour task scene around a Go2 robot model (DRAFT for SIM review).

Adds to the robot model: a flat floor, visual arena border, two equal-geometry
box slots, a yellow front marker on the robot's back, and the fixed overhead
224 x 224 camera ``overhead_v1`` (D-003, D-005). Box colour and mass come from
the appearance class; geometry and friction are identical for every class.
All values here are PROPOSED until G0 measurements fix them (P-002, P-003).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .locomotion import (
    MENAGERIE_GO2_SCENE_SHA256,
    MJLAB_GO2_XML_SHA256,
    LocomotionError,
    LocomotionSpec,
    sha256_file,
)

PARKED_XY = (50.0, 50.0)


@dataclass(frozen=True, slots=True)
class BoxClass:
    rgba: tuple[float, float, float, float]
    mass_kg: float
    movable: bool


@dataclass(frozen=True, slots=True)
class TaskSceneConfig:
    scene_version: str = "go2wm-push-detour-draft-v0"
    arena_width_m: float = 4.0
    arena_height_m: float = 4.0
    camera_id: str = "overhead_v1"
    image_width_px: int = 224
    image_height_px: int = 224
    camera_fovy_deg: float = 60.0
    camera_margin_m: float = 0.1
    box_half_extent_m: float = 0.20
    box_friction: float = 1.0
    box_slots: int = 2
    box_classes: dict[str, BoxClass] = field(
        default_factory=lambda: {
            "blue": BoxClass((0.15, 0.35, 0.90, 1.0), 1.0, True),
            "red": BoxClass((0.90, 0.15, 0.12, 1.0), 20.0, False),
        }
    )
    floor_rgba: tuple[float, float, float, float] = (0.30, 0.32, 0.34, 1.0)
    border_rgba: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
    marker_rgba: tuple[float, float, float, float] = (1.0, 0.85, 0.1, 1.0)
    marker_pos_m: tuple[float, float, float] = (0.17, 0.0, 0.085)
    marker_half_size_m: tuple[float, float, float] = (0.06, 0.05, 0.004)

    def __post_init__(self) -> None:
        if self.box_half_extent_m <= 0 or self.box_slots <= 0:
            raise ValueError("box size and slot count must be positive")
        masses = {c.movable: c.mass_kg for c in self.box_classes.values()}
        if len(masses) == 2 and masses[True] >= masses[False]:
            raise ValueError("movable boxes must be lighter than resistant boxes")

    @property
    def camera_height_m(self) -> float:
        half_extent = max(self.arena_width_m, self.arena_height_m) / 2 + self.camera_margin_m
        return half_extent / math.tan(math.radians(self.camera_fovy_deg) / 2)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class BuiltScene:
    model: Any
    robot_root_body: int
    box_body_ids: tuple[int, ...]
    box_geom_ids: tuple[int, ...]
    box_joint_qposadr: tuple[int, ...]
    robot_xml_sha256: str
    scene_id: str


def menagerie_scene_xml(cache_dir: Path) -> Path:
    import mujoco_menagerie as menagerie

    robot = menagerie.get("unitree_go2")
    return Path(robot.xml(cache=menagerie.Cache(cache_dir)))


def _robot_spec(spec_kind: LocomotionSpec, robot_xml: Path) -> tuple[Any, str]:
    import mujoco

    digest = sha256_file(robot_xml)
    expected = {
        "menagerie_go2_scene": MENAGERIE_GO2_SCENE_SHA256,
        "mjlab_go2_xml": MJLAB_GO2_XML_SHA256,
    }[spec_kind.robot_model]
    if digest != expected or digest != spec_kind.robot_xml_sha256:
        raise LocomotionError(f"{robot_xml} sha256 {digest} is not the pinned {expected}")
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    for key in list(spec.keys):
        spec.delete(key)
    if spec_kind.robot_model == "mjlab_go2_xml":
        _apply_mjlab_physics(spec, spec_kind)
    return spec, digest


def _apply_mjlab_physics(spec: Any, controller: LocomotionSpec) -> None:
    """Reproduce what MjLab adds to its bare go2.xml for Unitree-Go2-Flat.

    Sources: ``go2_constants.py`` (position actuators, FULL_COLLISION),
    ``velocity_env_cfg.py`` (timestep 0.005, 10 solver / 20 line-search
    iterations), mjlab 1.2.0 ``MujocoCfg`` defaults, and mjlab's ground plane.
    """

    import mujoco

    spec.option.timestep = controller.physics_dt_s
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    for geom in spec.geoms:
        name = geom.name or ""
        if not name.endswith("_collision") and "_collision" not in name:
            continue
        geom.contype = 1
        geom.conaffinity = 0
        if name in {f"{leg}_foot_collision" for leg in ("FL", "FR", "RL", "RR")}:
            geom.condim = 3
            geom.priority = 1
            geom.friction = [0.6, 0.005, 0.0001]
            geom.solimp = [0.9, 0.95, 0.023, 0.5, 2.0]
        else:
            geom.condim = 1
    joints = {j.name: j for j in spec.joints}
    assert controller.armature is not None
    for index, joint_name in enumerate(controller.joint_names):
        joints[joint_name].armature = controller.armature[index]
        actuator = spec.add_actuator(name=joint_name, target=joint_name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = controller.stiffness[index]
        actuator.biasprm[1] = -controller.stiffness[index]
        actuator.biasprm[2] = -controller.damping[index]
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange = [-controller.effort_limit[index], controller.effort_limit[index]]
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01])


def build_task_scene(
    controller: LocomotionSpec, robot_xml: Path, config: TaskSceneConfig
) -> BuiltScene:
    import mujoco

    spec, robot_digest = _robot_spec(controller, robot_xml)
    if controller.robot_model == "menagerie_go2_scene":
        spec.option.timestep = controller.physics_dt_s

    floor_material = spec.add_material(name="go2wm_floor")
    floor_material.rgba = list(config.floor_rgba)
    floor_material.reflectance = 0.0
    floor_material.specular = 0.0
    for geom in spec.geoms:
        if geom.name == "floor":
            geom.material = "go2wm_floor"
            geom.size = [config.arena_width_m * 5, config.arena_height_m * 5, 0.05]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    for light in list(spec.lights):
        light.castshadow = False

    world = spec.worldbody
    half_w, half_h = config.arena_width_m / 2, config.arena_height_m / 2
    line = 0.015
    for name, pos, size in (
        ("border_n", (0, half_h, 0.001), (half_w + line, line, 0.001)),
        ("border_s", (0, -half_h, 0.001), (half_w + line, line, 0.001)),
        ("border_e", (half_w, 0, 0.001), (line, half_h + line, 0.001)),
        ("border_w", (-half_w, 0, 0.001), (line, half_h + line, 0.001)),
    ):
        world.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(pos),
            size=list(size),
            rgba=list(config.border_rgba),
            contype=0,
            conaffinity=0,
        )
    world.add_camera(
        name=config.camera_id,
        pos=[0.0, 0.0, config.camera_height_m],
        xyaxes=[1, 0, 0, 0, 1, 0],
        fovy=config.camera_fovy_deg,
    )

    base = next(b for b in spec.bodies if b.name == "base_link" or b.name == "base")
    base.add_geom(
        name="front_marker",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=list(config.marker_pos_m),
        size=list(config.marker_half_size_m),
        rgba=list(config.marker_rgba),
        contype=0,
        conaffinity=0,
        density=0,
        group=0,
    )

    half = config.box_half_extent_m
    default_class = next(iter(config.box_classes.values()))
    for slot in range(config.box_slots):
        body = world.add_body(
            name=f"object_{slot}",
            pos=[PARKED_XY[0] + 2 * slot, PARKED_XY[1], half],
        )
        body.add_freejoint(name=f"object_{slot}_free")
        body.add_geom(
            name=f"object_{slot}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[half, half, half],
            mass=default_class.mass_kg,
            rgba=list(default_class.rgba),
            friction=[config.box_friction, 0.005, 0.0001],
            contype=1,
            conaffinity=1,
            condim=3,
        )

    model = spec.compile()
    model.opt.timestep = controller.physics_dt_s
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base.name)
    box_bodies = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"object_{i}")
        for i in range(config.box_slots)
    )
    box_geoms = tuple(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"object_{i}_geom")
        for i in range(config.box_slots)
    )
    box_qpos = tuple(
        int(
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"object_{i}_free")
            ]
        )
        for i in range(config.box_slots)
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "scene": config.digest(),
                "robot_xml_sha256": robot_digest,
                "controller": controller.controller_id,
                "mujoco": mujoco.__version__,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    return BuiltScene(
        model=model,
        robot_root_body=base_id,
        box_body_ids=box_bodies,
        box_geom_ids=box_geoms,
        box_joint_qposadr=box_qpos,
        robot_xml_sha256=robot_digest,
        scene_id=f"{config.scene_version}-{identity}",
    )
