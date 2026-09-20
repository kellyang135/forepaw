#!/usr/bin/env python3
"""Check a Unitree RL MjLab Go2 controller packet end to end with one command.

Steps, each reported PASS / FAIL / SKIP; later steps still run after an
earlier failure when they can, so one run shows every problem:

1. handoff   SIM's ``verify_controller_handoff`` (manifest, pins, checksums,
             validation flags).
2. meshes    the packet's go2.xml compiles, i.e. its ``assets/`` meshes came along.
3. metadata  the export metadata inside policy.onnx matches the audited
             ``MJLAB_GO2_FLAT`` spec (joint order, gains, default pose,
             observations, action scale).
4. deploy    deploy.yaml matches the same spec.
5. inference policy.onnx runs on CPU: 47 inputs, 12 finite outputs, deterministic.
6. trials    ``run_g0_trials.py --controller mjlab`` in the project adapter
             (skipped with --skip-trials).

Writes ``verification.json`` into --output-dir. Evidence for SIM review, not a
G0 sign-off by itself.

    python scripts/verify_mjlab_packet.py <packet_dir> \\
        --output-dir artifacts/runs/$(date -u +%Y%m%dT%H%M%SZ)-mjlab-packet-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packet_file(packet: Path, key: str) -> Path | None:
    try:
        manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
        return packet / manifest["files"][key]["path"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None


def step_handoff(packet: Path) -> dict[str, Any]:
    from go2wm.sim.controller_handoff import verify_controller_handoff

    summary = verify_controller_handoff(packet)
    return {"controller_id": summary.controller_id, "files": summary.file_count}


def step_meshes(packet: Path) -> dict[str, Any]:
    import mujoco

    xml = _packet_file(packet, "robot_xml")
    if xml is None or not xml.is_file():
        raise FileNotFoundError("manifest has no readable robot_xml")
    model = mujoco.MjModel.from_xml_path(str(xml))
    return {"robot_xml": str(xml.relative_to(packet)), "nmesh": model.nmesh, "njnt": model.njnt}


def step_metadata(packet: Path) -> dict[str, Any]:
    from go2wm.sim.locomotion import check_mjlab_onnx_metadata

    onnx_path = _packet_file(packet, "policy_onnx")
    if onnx_path is None:
        raise FileNotFoundError("manifest has no policy_onnx")
    meta = check_mjlab_onnx_metadata(onnx_path)
    return {"run_path": meta.get("run_path", ""), "observation_names": meta["observation_names"]}


def step_deploy(packet: Path) -> dict[str, Any]:
    from go2wm.sim.locomotion import check_mjlab_deploy_yaml

    deploy = _packet_file(packet, "deployment_config")
    if deploy is None:
        raise FileNotFoundError("manifest has no deployment_config")
    check_mjlab_deploy_yaml(deploy)
    return {"deploy_yaml": str(deploy.relative_to(packet))}


def step_inference(packet: Path) -> dict[str, Any]:
    import numpy as np

    from go2wm.sim.locomotion import MJLAB_GO2_FLAT, OnnxPolicy

    onnx_path = _packet_file(packet, "policy_onnx")
    if onnx_path is None:
        raise FileNotFoundError("manifest has no policy_onnx")
    policy = OnnxPolicy(onnx_path, MJLAB_GO2_FLAT.observation_size)
    rng = np.random.default_rng(0)
    observations = [np.zeros(47)] + [rng.normal(0, 0.5, 47) for _ in range(8)]
    outputs = [policy(obs) for obs in observations]
    again = policy(observations[1])
    if any(out.shape != (12,) for out in outputs):
        raise ValueError(f"output shapes {[out.shape for out in outputs]}")
    if not all(np.isfinite(out).all() for out in outputs):
        raise ValueError("non-finite actions")
    if not np.array_equal(again, outputs[1]):
        raise ValueError("inference is not deterministic")
    return {
        "zero_obs_action_abs_max": round(float(np.abs(outputs[0]).max()), 4),
        "random_obs_action_abs_max": round(float(max(np.abs(o).max() for o in outputs[1:])), 4),
    }


def step_trials(packet: Path, output_dir: Path) -> dict[str, Any]:
    onnx_path = _packet_file(packet, "policy_onnx")
    xml = _packet_file(packet, "robot_xml")
    deploy = _packet_file(packet, "deployment_config")
    if onnx_path is None or xml is None:
        raise FileNotFoundError("manifest lacks policy_onnx or robot_xml")
    trials_dir = output_dir / "g0-trials"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_g0_trials.py"),
        "--controller",
        "mjlab",
        "--policy",
        str(onnx_path),
        "--robot-xml",
        str(xml),
        "--output-dir",
        str(trials_dir),
    ]
    if deploy is not None:
        command += ["--deploy-yaml", str(deploy)]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    report_path = trials_dir / "report.json"
    if not report_path.is_file():
        raise RuntimeError(f"trials did not write a report: {completed.stderr[-1500:]}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = [name for name, ok in report["checks"].items() if not ok]
    if failed:
        raise RuntimeError(f"G0 feasibility checks failed: {failed}")
    return {"checks": report["checks"], "report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("packet", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-trials", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    packet = args.packet.resolve()

    steps: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("handoff", lambda: step_handoff(packet)),
        ("meshes", lambda: step_meshes(packet)),
        ("metadata", lambda: step_metadata(packet)),
        ("deploy", lambda: step_deploy(packet)),
        ("inference", lambda: step_inference(packet)),
    ]
    results: dict[str, dict[str, Any]] = {}
    for name, run in steps:
        try:
            results[name] = {"status": "PASS", **run()}
        except Exception as error:  # every failure is recorded, then the next step runs
            results[name] = {
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "trace": traceback.format_exc(limit=3),
            }
    prerequisites = ("meshes", "metadata", "inference")
    if args.skip_trials:
        results["trials"] = {"status": "SKIP", "reason": "--skip-trials"}
    elif any(results[name]["status"] != "PASS" for name in prerequisites):
        results["trials"] = {
            "status": "SKIP",
            "reason": f"needs {', '.join(prerequisites)} to pass",
        }
    else:
        try:
            results["trials"] = {"status": "PASS", **step_trials(packet, args.output_dir)}
        except Exception as error:
            results["trials"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}

    verdict = "PASS" if all(r["status"] == "PASS" for r in results.values()) else "FAIL"
    report = {
        "schema_version": "go2wm.mjlab-packet-check.v0-draft",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "packet": str(packet),
        "verdict": verdict,
        "steps": results,
    }
    verification_path = args.output_dir / "verification.json"
    verification_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    evidence_files = sorted(path for path in args.output_dir.rglob("*") if path.is_file())
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(args.output_dir)}\n"
            for path in evidence_files
        ),
        encoding="utf-8",
    )
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256(args.output_dir / relative) != expected:
            raise RuntimeError(f"post-write checksum verification failed for {relative}")
    for name, result in results.items():
        detail = result.get("error") or result.get("reason") or ""
        print(f"{result['status']:4}  {name:10} {detail}")
    print(f"verdict: {verdict}  ({args.output_dir / 'verification.json'})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
