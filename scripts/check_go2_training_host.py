#!/usr/bin/env python3
"""Print a machine-readable Go2 controller training-host preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from go2wm.sim.training_host import training_host_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = training_host_report(args.path)
    report["recorded_at_utc"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if args.output_dir is not None:
        if args.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite {args.output_dir}")
        args.output_dir.mkdir(parents=True)
        report_path = args.output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
        (args.output_dir / "checksums.sha256").write_text(
            f"{digest}  {report_path.name}\n"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
