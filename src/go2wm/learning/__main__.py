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

    fake = sub.add_parser("fake-data", help="scripted fake-simulator dataset (rehearsal only)")
    fake.add_argument("--out", type=Path, required=True)
    fake.add_argument("--episodes", type=int, default=60)
    fake.add_argument("--blocks", type=int, default=16)
    fake.add_argument("--image-size", type=int, default=224)
    fake.add_argument("--salt", default="fake-rehearsal-v1")

    cache = sub.add_parser("lewm-cache", help="write lossless train + validation LeWM caches")
    cache.add_argument("--data", type=Path, required=True)
    cache.add_argument("--splits", type=Path, required=True)
    cache.add_argument("--out-dir", type=Path, required=True)
    cache.add_argument("--dataset-id", required=True)
    cache.add_argument("--image-size", type=int, default=224)

    lewm_eval = sub.add_parser("lewm-eval", help="G2/G3 + surprise + bundle for a trained LeWM run")
    lewm_eval.add_argument("--run", type=Path, required=True)
    lewm_eval.add_argument("--checkpoint", default="best", help="best | last | epoch number")
    lewm_eval.add_argument("--lewm-repo", default=None)
    lewm_eval.add_argument("--data", type=Path, required=True)
    lewm_eval.add_argument("--splits", type=Path, required=True)
    lewm_eval.add_argument("--out", type=Path, required=True)
    lewm_eval.add_argument("--dataset-id", required=True)
    lewm_eval.add_argument("--backend", default="mujoco")
    lewm_eval.add_argument("--device", default="cpu")

    control = sub.add_parser(
        "synthetic-control", help="known-answer caches that test whether LeWM learns actions"
    )
    control.add_argument("--out-dir", type=Path, required=True)
    control.add_argument("--train-episodes", type=int, default=120)
    control.add_argument("--val-episodes", type=int, default=20)
    control.add_argument("--seed", type=int, default=0)

    compare = sub.add_parser("compare", help="side-by-side key metrics from ml_report.json files")
    compare.add_argument("reports", type=Path, nargs="+")

    args = parser.parse_args(argv)

    if args.command == "synthetic-control":
        from .lewm_cache import write_synthetic_control

        train_cache, val_cache = write_synthetic_control(
            args.out_dir,
            train_episodes=args.train_episodes,
            val_episodes=args.val_episodes,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "train_cache": str(train_cache.root),
                    "val_cache": str(val_cache.root),
                    "train_clips": len(train_cache.clip_starts(4)),
                    "val_clips": len(val_cache.clip_starts(4)),
                    "pass_criterion": "after training, val pred loss clearly below both "
                    "copy-last and shuffled-action losses",
                },
                indent=2,
            )
        )
        return 0

    if args.command == "compare":
        from .report import compare_reports

        print(
            compare_reports(
                [json.loads(p.read_text(encoding="utf-8")) for p in args.reports],
                [str(p) for p in args.reports],
            )
        )
        return 0

    if args.command == "fake-data":
        from .pipeline import collect_fake_dataset

        seeds = list(range(1, args.episodes + 1))
        fake_manifest = build_split_manifest(seeds, salt=args.salt, notes="fake backend only")
        write_split_manifest(args.out / "splits.json", fake_manifest)
        collect_fake_dataset(
            args.out / "data", fake_manifest, seeds, blocks=args.blocks, image_size=args.image_size
        )
        print(
            json.dumps(
                {
                    "data": str(args.out / "data"),
                    "splits": str(args.out / "splits.json"),
                    "split_id": fake_manifest.split_id,
                    "episodes": args.episodes,
                    "image_size": args.image_size,
                    "scope": "fake simulator; rehearsal and software checks only",
                },
                indent=2,
            )
        )
        return 0

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

    if args.command == "lewm-cache":
        from .lewm_cache import build_training_cache

        leak = check_leakage(dataset.episodes, manifest)
        if not leak.ok:
            print(json.dumps({"error": "leak check failed", "problems": list(leak.problems)}))
            return 1
        written = {}
        for split in (DatasetSplit.TRAIN, DatasetSplit.VALIDATION):
            chosen = dataset.by_split(split)
            if not chosen:
                print(json.dumps({"error": f"no {split.value} episodes"}))
                return 1
            built = build_training_cache(
                chosen,
                args.out_dir / f"cache-{split.value}",
                split=split,
                dataset_id=args.dataset_id,
                split_id=manifest.split_id,
                image_size=args.image_size,
            )
            written[split.value] = {
                "path": str(built.root),
                "episodes": built.manifest["episode_count"],
                "frames": built.frame_count,
                "clips_4_frames": len(built.clip_starts(4)),
            }
        written["note"] = "test episodes are never cached for training"
        print(json.dumps(written, indent=2))
        return 0

    if args.command == "lewm-eval":
        from .pipeline import run_lewm_pipeline
        from .report import gate_table

        report = run_lewm_pipeline(
            dataset.episodes,
            manifest,
            args.out,
            run_dir=args.run,
            checkpoint=args.checkpoint,
            lewm_repo=args.lewm_repo,
            device=args.device,
            dataset_id=args.dataset_id,
            data_backend=args.backend,
        )
        print(gate_table(report))
        print(
            json.dumps(
                {"bundle_id": report["bundle_id"], "report": report["report_path"]}, indent=2
            )
        )
        return 0 if report["all_checks_passed"] else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
