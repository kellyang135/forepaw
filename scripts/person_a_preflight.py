#!/usr/bin/env python3
"""Create a retained, fail-closed Person A / Gate G0 preflight report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(*command: str, cwd: Path = ROOT) -> dict[str, Any]:
    process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": list(command),
        "returncode": process.returncode,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    revision = _run("git", "rev-parse", "HEAD", cwd=path)
    status = _run("git", "status", "--short", cwd=path)
    return {
        "revision": revision["stdout"] if revision["returncode"] == 0 else "NO_COMMIT",
        "dirty": bool(status["stdout"]),
        "status": status["stdout"],
    }


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dimos-source", required=True, type=Path)
    parser.add_argument("--unitree-rl-gym-source", required=True, type=Path)
    parser.add_argument("--model-report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    model_report = json.loads(_read(args.model_report))
    process_source = (
        args.dimos_source / "dimos/simulation/mujoco/mujoco_process.py"
    )
    model_source = args.dimos_source / "dimos/simulation/mujoco/model.py"
    process_text = _read(process_source)
    model_text = _read(model_source)
    alias_fragment = (
        'if robot_name == "unitree_go2":\n        robot_name = "unitree_go1"'
    )
    pretrained = sorted(
        path.relative_to(args.unitree_rl_gym_source).as_posix()
        for path in (args.unitree_rl_gym_source / "deploy/pre_train").glob("*/motion.pt")
    )
    disk = shutil.disk_usage(ROOT)
    optional_modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("dimos", "mujoco", "mujoco_menagerie", "numpy")
    }
    dimos_aliases_go2_to_go1 = alias_fragment in process_text
    dimos_has_go2_policy_case = 'case "unitree_go2"' in model_text
    shipped_go2_policy = any("/go2/" in f"/{item}/" for item in pretrained)
    render_verified = (
        model_report.get("status") == "PASS_MODEL_AND_RENDER"
        and model_report.get("render", {}).get("shape") == [224, 224, 3]
    )
    gait_verified = bool(
        model_report.get("uncontrolled_step_probe", {}).get("gait_verified")
    )

    blockers: list[str] = []
    if dimos_aliases_go2_to_go1:
        blockers.append("pinned dimOS simulation aliases unitree_go2 to unitree_go1")
    if not dimos_has_go2_policy_case:
        blockers.append("pinned dimOS model loader has no unitree_go2 controller case")
    if not shipped_go2_policy:
        blockers.append("pinned official Unitree RL Gym checkout ships no Go2 motion.pt")
    if not gait_verified:
        blockers.append("no true-Go2 locomotion gait has been executed or verified")
    if not optional_modules["dimos"]:
        blockers.append("dimOS is not installed in the project environment")
    if not render_verified:
        blockers.append("224x224 Go2 render probe did not pass")

    report = {
        "schema_version": "go2wm.person-a-preflight.v1",
        "run_id": args.output_dir.name,
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "gate": "G0",
        "result": "BLOCKED" if blockers else "PASS",
        "scope": "Person A environment, source, true-Go2 model, renderer, and gait preflight",
        "project": {
            **_git_state(ROOT),
            "root": str(ROOT),
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "optional_modules": optional_modules,
            "executables": {
                name: shutil.which(name)
                for name in ("docker", "dimos", "nvidia-smi", "uv")
            },
            "pid": os.getpid(),
        },
        "dimos_source": {
            **_git_state(args.dimos_source),
            "path": str(args.dimos_source),
            "mujoco_process_sha256": _sha256(process_source),
            "model_loader_sha256": _sha256(model_source),
            "aliases_go2_to_go1": dimos_aliases_go2_to_go1,
            "has_go2_policy_case": dimos_has_go2_policy_case,
            "go1_policy_case_present": 'case "unitree_go1"' in model_text,
        },
        "unitree_rl_gym_source": {
            **_git_state(args.unitree_rl_gym_source),
            "path": str(args.unitree_rl_gym_source),
            "pretrained_motion_files": pretrained,
            "shipped_go2_policy": shipped_go2_policy,
        },
        "go2_model_probe": {
            "report_path": str(args.model_report),
            "report_sha256": _sha256(args.model_report),
            "status": model_report.get("status"),
            "render_verified_224_rgb": render_verified,
            "gait_verified": gait_verified,
            "model_tree_oid": model_report.get("source", {}).get("model_tree_oid"),
            "menagerie_commit": model_report.get("source", {}).get(
                "menagerie_commit"
            ),
        },
        "blockers": blockers,
        "truthfulness_note": (
            "A compiled/rendered Go2 MJCF is not locomotion evidence. G0 remains "
            "blocked until a true-Go2 gait/controller passes the prescribed motion, "
            "push, reset, camera, and throughput trials."
        ),
    }
    report_path = args.output_dir / "preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "checksums.sha256").write_text(
        f"{_sha256(report_path)}  {report_path.name}\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
