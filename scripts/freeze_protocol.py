#!/usr/bin/env python3
"""Hash the evaluation protocol after all provisional fields are resolved."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from go2wm.config import load_json, load_toml
from go2wm.evidence import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_FILES = (
    ROOT / "configs/experiment.toml",
    ROOT / "configs/acceptance.toml",
    ROOT / "configs/eval_scenarios.json",
    ROOT / "configs/external_sources.toml",
)


def _assert_freezable(allow_unpinned: bool) -> None:
    experiment = load_toml(PROTOCOL_FILES[0])
    acceptance = load_toml(PROTOCOL_FILES[1])
    scenarios = load_json(PROTOCOL_FILES[2])
    sources = load_toml(PROTOCOL_FILES[3])

    errors: list[str] = []
    if str(experiment.get("status", "")).startswith("PROVISIONAL"):
        errors.append("experiment.toml is still provisional")
    if acceptance.get("status") != "FROZEN":
        errors.append("acceptance.toml status is not FROZEN")
    if scenarios.get("status") != "FROZEN":
        errors.append("eval_scenarios.json status is not FROZEN")
    if not allow_unpinned:
        for name in ("lewm", "dimos", "go2_model"):
            item = sources[name]
            revision = item.get("commit", item.get("revision"))
            if revision == "UNPINNED":
                errors.append(f"external source {name} is unpinned")
            if item.get("verified_locally") is not True:
                errors.append(f"external source {name} is unverified")
    if errors:
        raise ValueError("protocol cannot be frozen:\n- " + "\n- ".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/protocol-lock.json",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Emergency-only override; the resulting lock records this exception.",
    )
    args = parser.parse_args(argv)

    try:
        _assert_freezable(args.allow_unpinned)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision or "NO_COMMIT",
        "allow_unpinned": args.allow_unpinned,
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in PROTOCOL_FILES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

