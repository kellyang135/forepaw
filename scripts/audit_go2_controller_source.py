#!/usr/bin/env python3
"""Retain a fail-closed audit of the pinned Unitree Go2 controller source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PINNED_REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
PINNED_XML_SHA256 = "077fc7f70ce2ddcfdab814aa0800efa2c974db6c98594b2f7663f779509fb912"

AUDITED_FILES = {
    "robot_xml": "src/assets/robots/unitree_go2/xmls/go2.xml",
    "robot_constants": "src/assets/robots/unitree_go2/go2_constants.py",
    "environment": "src/tasks/velocity/velocity_env_cfg.py",
    "go2_environment": "src/tasks/velocity/config/go2/env_cfgs.py",
    "runner": "src/tasks/velocity/config/go2/rl_cfg.py",
    "onnx_export": "src/tasks/velocity/rl/runner.py",
    "deployment": "deploy/robots/go2/config/policy/velocity/v0/params/deploy.yaml",
    "publisher_guard": "deploy/robots/go2/main.cpp",
}

TEXT_REQUIREMENTS = {
    "environment": ("timestep=0.005", "decimation=4", "scale=0.25"),
    "go2_environment": (
        "lin_vel_x = (-0.5, 1.0)",
        "lin_vel_y = (-0.5, 0.5)",
        "ang_vel_z = (-0.5, 0.5)",
    ),
    "runner": ('experiment_name="go2_velocity"', "max_iterations=10001"),
    "onnx_export": ('filename = "policy.onnx"', "self.export_policy_to_onnx"),
    "deployment": ("step_dt: 0.02", "JointPositionAction:", "scale: [0.25"),
    "publisher_guard": ("The other process is using the lowcmd channel",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=checkout, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkout = args.checkout.resolve()
    output_dir: Path = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    revision = _git(checkout, "rev-parse", "HEAD")
    remote = _git(checkout, "config", "--get", "remote.origin.url")
    file_report: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, relative in AUDITED_FILES.items():
        path = checkout / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        digest = _sha256(path)
        requirements = TEXT_REQUIREMENTS.get(name, ())
        text = path.read_text(encoding="utf-8") if requirements else ""
        missing_needles = [needle for needle in requirements if needle not in text]
        if missing_needles:
            errors.append(f"{relative} missing expected declarations: {missing_needles}")
        file_report[name] = {
            "path": relative,
            "sha256": digest,
            "required_declarations_present": not missing_needles,
        }

    if revision != PINNED_REVISION:
        errors.append(f"revision {revision} does not match pin {PINNED_REVISION}")
    xml = file_report.get("robot_xml")
    if xml is None or xml["sha256"] != PINNED_XML_SHA256:
        errors.append("Go2 XML checksum does not match the compatibility pin")

    report = {
        "schema_version": "go2wm.go2-controller-source-audit.v1",
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "result": "PASS" if not errors else "FAIL",
        "scope": "source compatibility only; no training, playback, locomotion, or G0 pass",
        "source": {"checkout": str(checkout), "remote": remote, "revision": revision},
        "contract": {
            "task": "Unitree-Go2-Flat",
            "physics_dt_s": 0.005,
            "controller_dt_s": 0.02,
            "physics_substeps_per_control": 4,
            "project_block_policy_calls": 25,
            "project_block_physics_steps": 100,
            "lateral_velocity_mps": 0.0,
        },
        "files": file_report,
        "errors": errors,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "checksums.sha256").write_text(
        f"{_sha256(report_path)}  {report_path.name}\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
