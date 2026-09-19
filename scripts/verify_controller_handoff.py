#!/usr/bin/env python3
"""Verify a Go2 controller packet before simulator integration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from go2wm.sim.controller_handoff import verify_controller_handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = verify_controller_handoff(args.directory)
    payload = asdict(summary)
    payload["policy_path"] = str(summary.policy_path)
    payload["checkpoint_path"] = str(summary.checkpoint_path)
    print(json.dumps({"status": "PASS", **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
