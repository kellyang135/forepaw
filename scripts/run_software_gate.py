#!/usr/bin/env python3
"""Run the software-only verification gate and retain raw, checksummed output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(name: str, command: list[str], output_dir: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    stdout_path = output_dir / f"{name}.stdout.txt"
    stderr_path = output_dir / f"{name}.stderr.txt"
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }


def _git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else "NO_COMMIT",
        "dirty": bool(status.stdout.strip()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    python = sys.executable
    coverage_path = args.output_dir / "coverage.json"
    checks = {
        "ruff": [python, "-m", "ruff", "check", "."],
        "pytest_coverage": [
            python,
            "-m",
            "pytest",
            "--cov=go2wm",
            f"--cov-report=json:{coverage_path}",
            "-q",
        ],
        "doctor": [python, "-m", "go2wm", "doctor"],
        "fake_smoke": [python, "-m", "go2wm", "fake-smoke"],
        "candidates": [python, "-m", "go2wm", "candidates"],
        "protocol": [python, str(ROOT / "scripts/verify_protocol.py")],
    }
    results = {
        name: _run(name, command, args.output_dir) for name, command in checks.items()
    }
    passed = all(item["returncode"] == 0 for item in results.values())
    pytest_output = (args.output_dir / "pytest_coverage.stdout.txt").read_text()
    match = re.search(r"(\d+) passed", pytest_output)
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
    report = {
        "schema_version": "go2wm.software-gate.v1",
        "run_id": args.output_dir.name,
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "result": "PASS" if passed else "FAIL",
        "scope": "software contracts only; not a Go2 locomotion or G0 pass",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
        },
        "project": _git_state(),
        "metrics": {
            "tests_passed": int(match.group(1)) if match else -1,
            "line_coverage_percent": coverage.get("totals", {}).get("percent_covered"),
        },
        "checks": results,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    artifact_paths = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    (args.output_dir / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifact_paths)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
