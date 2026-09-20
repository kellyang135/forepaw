#!/usr/bin/env python3
"""Independently verify a direct-MjLab G1A packet and seal it with checksums."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2wm.data.g1 import verify_g1_packet, write_g1_verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    report = verify_g1_packet(args.packet)
    report_path, checksums_path = write_g1_verification(args.packet, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"verification: {report_path}")
    print(f"outer checksums: {checksums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
