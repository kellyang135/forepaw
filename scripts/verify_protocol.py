#!/usr/bin/env python3
"""Validate checked-in protocol/config files without simulator dependencies."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from go2wm.config import (
    ConfigurationError,
    load_json,
    load_toml,
    validate_eval_scenarios,
    validate_experiment,
    validate_external_sources,
)

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    checks = (
        (
            ROOT / "configs/experiment.toml",
            lambda path: validate_experiment(load_toml(path)),
        ),
        (
            ROOT / "configs/eval_scenarios.json",
            lambda path: validate_eval_scenarios(load_json(path)),
        ),
        (
            ROOT / "configs/external_sources.toml",
            lambda path: validate_external_sources(load_toml(path)),
        ),
    )
    report: dict[str, object] = {"ok": True, "files": {}}
    try:
        for path, validator in checks:
            report["files"][str(path.relative_to(ROOT))] = {
                "sha256": digest(path),
                "warnings": validator(path),
            }
    except (ConfigurationError, OSError, ValueError) as error:
        report["ok"] = False
        report["error"] = str(error)

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

