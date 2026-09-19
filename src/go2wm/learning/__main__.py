"""Person B command line: ``python -m go2wm.learning <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from go2wm.contracts import DatasetSplit

from .dataset import load_dataset
from .diagnostics import coverage_report
from .splits import build_split_manifest, check_leakage, read_split_manifest, write_split_manifest


def _parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        elif part.strip():
            seeds.append(int(part))
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m go2wm.learning")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="full ML pipeline on fake-simulator data")
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--episodes", type=int, default=60)

    splits = sub.add_parser("make-splits", help="freeze a scenario-seed split manifest")
    splits.add_argument("--seeds", required=True, help="e.g. 1-400,1000-1023")
    splits.add_argument("--reserve-test", default="", help="seeds forced into test")
    splits.add_argument("--salt", required=True)
    splits.add_argument("--out", type=Path, required=True)

    check = sub.add_parser("check-data", help="strict load + leak check + coverage report")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--splits", type=Path, required=True)
    check.add_argument("--lenient", action="store_true", help="report rejects instead of stopping")

    train = sub.add_parser("baseline", help="fit/evaluate/publish the linear baseline bundle")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--splits", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--dataset-id", required=True)
    train.add_argument("--backend", default="mujoco")

    export = sub.add_parser("export-lewm", help="write aligned columns for LeWM training")
    export.add_argument("--data", type=Path, required=True)
    export.add_argument("--splits", type=Path, required=True)
    export.add_argument("--split", choices=("train", "validation"), required=True)
    export.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "smoke":
        from .pipeline import run_fake_smoke

        report = run_fake_smoke(args.out, episode_count=args.episodes)
        summary = {
            "evidence_scope": report["evidence_scope"],
            "bundle_id": report["bundle_id"],
            "report": report["report_path"],
            "gate_checks": [
                {"gate": c["gate"], "passed": c["passed"], "criterion": c["criterion"]}
                for c in report["gate_checks"]
            ],
            "planning_smoke": report["planning_smoke"],
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "make-splits":
        manifest = build_split_manifest(
            _parse_seeds(args.seeds),
            salt=args.salt,
            reserved_test_seeds=_parse_seeds(args.reserve_test),
        )
        write_split_manifest(args.out, manifest)
        print(
            json.dumps(
                {
                    "split_id": manifest.split_id,
                    "path": str(args.out),
                    "seeds": {s.value: len(manifest.seeds(s)) for s in DatasetSplit},
                },
                indent=2,
            )
        )
        return 0

    manifest = read_split_manifest(args.splits)
    dataset = load_dataset(args.data, strict=not getattr(args, "lenient", False))

    if args.command == "check-data":
        leak = check_leakage(dataset.episodes, manifest)
        print(
            json.dumps(
                {
                    "episodes": len(dataset.episodes),
                    "rejected": [list(item) for item in dataset.rejected],
                    "leak_ok": leak.ok,
                    "leak_problems": list(leak.problems),
                    "coverage": coverage_report(dataset.episodes),
                },
                indent=2,
            )
        )
        return 0 if leak.ok and not dataset.rejected else 1

    if args.command == "baseline":
        from .pipeline import run_baseline_pipeline

        report = run_baseline_pipeline(
            dataset.episodes,
            manifest,
            args.out,
            dataset_id=args.dataset_id,
            data_backend=args.backend,
        )
        summary = {
            "bundle_id": report["bundle_id"],
            "report": report["report_path"],
            "all_checks_passed": report["all_checks_passed"],
        }
        print(json.dumps(summary, indent=2))
        return 0 if report["all_checks_passed"] else 1

    if args.command == "export-lewm":
        from .lewm_adapter import export_sequences_npz

        leak = check_leakage(dataset.episodes, manifest)
        if not leak.ok:
            print(json.dumps({"error": "leak check failed", "problems": list(leak.problems)}))
            return 1
        chosen = dataset.by_split(DatasetSplit(args.split))
        print(json.dumps(export_sequences_npz(chosen, args.out), indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
