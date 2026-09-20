from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collect_wave_script", ROOT / "scripts/collect_wave.py"
)
assert SPEC is not None and SPEC.loader is not None
collect_wave = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect_wave)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "controller": "mjlab",
        "policy": Path("policy.onnx"),
        "robot_xml": Path("go2.xml"),
        "deploy_yaml": Path("deploy.yaml"),
        "menagerie_cache": Path("menagerie"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_mjlab_collection_route_validates_and_loads_onnx(monkeypatch) -> None:
    from go2wm.sim import locomotion

    calls: list[tuple[str, Path, object]] = []
    policy = object()

    monkeypatch.setattr(
        locomotion,
        "check_mjlab_onnx_metadata",
        lambda path, spec: calls.append(("onnx", path, spec)),
    )
    monkeypatch.setattr(
        locomotion,
        "check_mjlab_deploy_yaml",
        lambda path, spec: calls.append(("deploy", path, spec)),
    )
    monkeypatch.setattr(locomotion, "OnnxPolicy", lambda path, size: policy)

    spec, loaded, robot_xml = collect_wave.resolve_mujoco_controller(_args())

    assert spec is locomotion.MJLAB_GO2_FLAT
    assert loaded is policy
    assert robot_xml == Path("go2.xml")
    assert calls == [
        ("onnx", Path("policy.onnx"), locomotion.MJLAB_GO2_FLAT),
        ("deploy", Path("deploy.yaml"), locomotion.MJLAB_GO2_FLAT),
    ]


def test_mjlab_collection_route_requires_robot_xml() -> None:
    with pytest.raises(SystemExit, match="--robot-xml"):
        collect_wave.resolve_mujoco_controller(_args(robot_xml=None))
