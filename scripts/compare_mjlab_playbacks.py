#!/usr/bin/env python3
"""Compare upstream-checkpoint and project-adapter MjLab playback reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORWARD_DELTA_LIMIT_MPS = 0.08
YAW_DELTA_LIMIT_RPS = 0.12


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run_checksums(report_path: Path) -> None:
    manifest = report_path.parent / "checksums.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing checksum manifest beside {report_path}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = sha256_file(report_path.parent / name)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {report_path.parent / name}")


def case_medians(report: dict[str, Any]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for case in report["cases"]:
        good = [run for run in case["runs"] if not run["failed"]]
        if len(good) != 5:
            raise ValueError(f"{case['name']} has {len(good)} safe repeats, expected 5")
        result[case["name"]] = (
            statistics.median(run["mean_forward_mps"] for run in good),
            statistics.median(run["mean_yaw_rate_rps"] for run in good),
        )
    return result


def compare(upstream_path: Path, adapter_path: Path) -> dict[str, Any]:
    verify_run_checksums(upstream_path)
    verify_run_checksums(adapter_path)
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    if upstream.get("verdict") != "PASS" or adapter.get("verdict") != "PASS":
        raise ValueError("both input playback reports must have verdict PASS")
    upstream_cases = case_medians(upstream)
    adapter_cases = case_medians(adapter)
    if upstream_cases.keys() != adapter_cases.keys():
        raise ValueError("input reports do not contain the same command cases")
    cases: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for name in upstream_cases:
        upstream_vx, upstream_wz = upstream_cases[name]
        adapter_vx, adapter_wz = adapter_cases[name]
        forward_delta = abs(upstream_vx - adapter_vx)
        yaw_delta = abs(upstream_wz - adapter_wz)
        gated = name != "stand"
        if gated:
            checks[f"{name}: forward delta"] = forward_delta <= FORWARD_DELTA_LIMIT_MPS
            checks[f"{name}: yaw delta"] = yaw_delta <= YAW_DELTA_LIMIT_RPS
        cases.append(
            {
                "name": name,
                "gated": gated,
                "upstream_forward_mps": upstream_vx,
                "adapter_forward_mps": adapter_vx,
                "absolute_forward_delta_mps": forward_delta,
                "upstream_yaw_rate_rps": upstream_wz,
                "adapter_yaw_rate_rps": adapter_wz,
                "absolute_yaw_delta_rps": yaw_delta,
            }
        )
    return {
        "schema_version": "go2wm.mjlab-playback-comparison.v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "criteria": {
            "absolute_forward_delta_limit_mps": FORWARD_DELTA_LIMIT_MPS,
            "absolute_yaw_delta_limit_rps": YAW_DELTA_LIMIT_RPS,
            "source": "docs/MJLAB_PLAYBACK_ACCEPTANCE.md (predeclared)",
        },
        "inputs": {
            "upstream_report": {
                "path": str(upstream_path),
                "sha256": sha256_file(upstream_path),
            },
            "adapter_report": {
                "path": str(adapter_path),
                "sha256": sha256_file(adapter_path),
            },
        },
        "checks": checks,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-report", type=Path, required=True)
    parser.add_argument("--adapter-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    report = compare(args.upstream_report.resolve(), args.adapter_report.resolve())
    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "checksums.sha256").write_text(
        f"{sha256_file(report_path)}  comparison.json\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": report["verdict"], "output_dir": str(args.output_dir)}))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
