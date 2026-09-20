"""Lightweight command line entry points for contract-first development."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from go2wm.config import (
    load_json,
    load_toml,
    validate_eval_scenarios,
    validate_experiment,
    validate_external_sources,
)
from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
)
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest, validate_episode
from go2wm.planning import build_candidate_library, family_counts
from go2wm.sim import DeterministicFakeSimulator

ROOT = Path(__file__).resolve().parents[2]


def _doctor() -> int:
    warnings = []
    warnings.extend(validate_experiment(load_toml(ROOT / "configs/experiment.toml")))
    warnings.extend(
        validate_eval_scenarios(load_json(ROOT / "configs/eval_scenarios.json"))
    )
    warnings.extend(
        validate_external_sources(load_toml(ROOT / "configs/external_sources.toml"))
    )
    report = {
        "status": "scaffold_ready",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "optional_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("dimos", "h5py", "mujoco", "torch")
        },
        "warnings": warnings,
        "truthfulness_note": (
            "Core contracts and a retained dimOS-to-true-Go2 transport rehearsal "
            "exist; learned Go2 quality, active stop preemption, and final "
            "task performance remain unverified."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _candidates() -> int:
    candidates = build_candidate_library()
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "family_counts": family_counts(candidates),
                "horizon_blocks": len(candidates[0].actions),
                "block_duration_s": candidates[0].actions[0].duration_s,
                "first_action_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _fake_smoke() -> int:
    simulator = DeterministicFakeSimulator()
    collector = EpisodeCollector(simulator, CollectionConfig())
    reset = ResetRequest(
        episode_id="fake-smoke",
        scenario_seed=1,
        robot_pose=Pose2D(-0.8, 0.0, 0.0),
        objects=(
            ObjectState("light", "blue", Pose2D(-0.2, 0.0), True),
            ObjectState("resistant", "red", Pose2D(0.8, 0.6), False),
        ),
    )
    record = collector.collect(
        EpisodeRequest(
            reset=reset,
            split=DatasetSplit.TRAIN,
            goal=Goal2D(1.0, 0.0, 0.15),
            run_id="fake-smoke-run",
            metadata={"purpose": "software-contract-smoke-only"},
        ),
        (
            ActionCommand(0.4, 0.0),
            ActionCommand(0.4, 0.0),
            ActionCommand(0.0, 0.0),
        ),
    )
    validate_episode(record)
    blocks: list[dict[str, Any]] = []
    for block in record.blocks:
        transition = block.transition
        blocks.append(
            {
                "index": block.block_index,
                "start_s": transition.start_observation.sim_time_s,
                "end_s": transition.end_observation.sim_time_s,
                "physics_samples": len(transition.physics_samples),
                "contacts": transition.events.contact_sample_count,
                "fell": transition.events.fell,
            }
        )
    print(
        json.dumps(
            {
                "status": "fake_smoke_passed",
                "evidence_scope": "software contracts only; not MuJoCo or Go2 evidence",
                "history_frames": len(record.initial_history),
                "history_commands": len(record.history_actions),
                "blocks": blocks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="go2wm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate config and inspect optional dependencies")
    subparsers.add_parser("candidates", help="summarize the fixed structured candidate library")
    subparsers.add_parser("fake-smoke", help="exercise collection using the fake backend")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "doctor": _doctor,
        "candidates": _candidates,
        "fake-smoke": _fake_smoke,
    }
    return commands[args.command]()


if __name__ == "__main__":
    sys.exit(main())
