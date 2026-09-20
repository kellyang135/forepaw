"""Collect one data wave with the scripted, boundary-aware collection policy.

    python scripts/collect_wave.py --backend mujoco --policy /tmp/robot_lab_policy.pt \
        --splits artifacts/splits.json --seeds 1-400 --out data/wave1

    python scripts/collect_wave.py --backend fake --splits runs/x/splits.json \
        --seeds 1-60 --out runs/x/data            # rehearsal only

Episodes are written with ``go2wm.data.write_episode`` (write-once, checksummed),
so ``python -m go2wm.learning check-data`` reads them directly. Test-split seeds
are skipped unless ``--include-test`` is passed; they belong to final evaluation.
The policy steers with privileged simulator poses, which D-004 permits during
collection only; every episode's metadata says so. ``summary.json`` reports the
interaction fraction against ``configs/acceptance.toml`` (0.20 to 0.30).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from go2wm.contracts import CameraConfig, DatasetSplit
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest, write_episode
from go2wm.data.scripted_policy import (
    DEFAULT_MODE_WEIGHTS,
    ArenaGeometry,
    CommandBounds,
    ScenarioSampler,
    ScriptedPolicy,
)
from go2wm.learning.splits import read_split_manifest

POLICY_ID = "scripted-collection-policy-v1"


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def build_backend(args: argparse.Namespace):
    if args.backend == "fake":
        from go2wm.sim import DeterministicFakeSimulator, FakeSimulatorConfig

        sim = DeterministicFakeSimulator(
            FakeSimulatorConfig(camera=CameraConfig("overhead", args.image_size, args.image_size))
        )
        arena = ArenaGeometry(
            sim.scene_config.arena_width_m, sim.scene_config.arena_height_m, box_half_extent_m=0.16
        )
        return sim, arena, CommandBounds(), "fake-backend"

    from go2wm.sim.locomotion import RL_SAR_ROBOT_LAB_GO2, load_rl_sar_policy
    from go2wm.sim.mujoco_go2 import MujocoGo2Config, MujocoGo2Simulator
    from go2wm.sim.task_scene import menagerie_scene_xml

    if args.policy is None:
        raise SystemExit("--policy is required for --backend mujoco")
    config = MujocoGo2Config()
    sim = MujocoGo2Simulator(
        RL_SAR_ROBOT_LAB_GO2,
        load_rl_sar_policy(args.policy),
        menagerie_scene_xml(args.menagerie_cache),
        config,
    )
    scene = config.scene
    arena = ArenaGeometry(
        scene.arena_width_m,
        scene.arena_height_m,
        box_half_extent_m=scene.box_half_extent_m,
        robot_box_clearance_m=config.min_robot_box_clearance_m,
    )
    bounds = CommandBounds(
        config.min_forward_velocity_mps,
        config.max_forward_velocity_mps,
        config.max_abs_yaw_rate_rps,
        config.block_duration_s,
    )
    return sim, arena, bounds, sim.controller_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=("fake", "mujoco"), required=True)
    parser.add_argument("--policy", type=Path, help="rl_sar robot_lab policy.pt (mujoco backend)")
    parser.add_argument("--menagerie-cache", type=Path, default=Path("/tmp/go2wm-menagerie-cache"))
    parser.add_argument(
        "--splits", type=Path, required=True, help="frozen split manifest (make-splits)"
    )
    parser.add_argument("--seeds", required=True, help="e.g. 1-400 or 1-50,90")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=40)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--mode-weights", type=json.loads, default=None, help='JSON, e.g. {"push": 0.3, ...}'
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--image-size", type=int, default=224, help="fake backend camera size")
    args = parser.parse_args()

    manifest = read_split_manifest(args.splits)
    sim, arena, bounds, controller_id = build_backend(args)
    sampler = ScenarioSampler(arena, mode_weights=args.mode_weights)
    collector = EpisodeCollector(sim, CollectionConfig(block_duration_s=bounds.block_duration_s))
    run_id = args.run_id or f"{args.out.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    args.out.mkdir(parents=True, exist_ok=True)

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    total_blocks = contact_blocks = 0
    started = time.perf_counter()
    for seed in parse_seeds(args.seeds):
        split = manifest.split_of(seed)
        if split is DatasetSplit.TEST and not args.include_test:
            counts["_skipped"]["test_seed"] += 1
            continue
        scenario = sampler.sample(seed, episode_prefix=args.out.name)
        policy = ScriptedPolicy(scenario, arena, bounds, seed=seed)
        request = EpisodeRequest(
            reset=scenario.reset,
            split=split,
            goal=scenario.goal,
            run_id=run_id,
            metadata={
                "collection_policy": POLICY_ID,
                "collection_mode": scenario.mode,
                "controller_id": controller_id,
                "backend": args.backend,
                "privileged_steering": "collection only (D-004)",
            },
        )
        try:
            record = collector.collect_with_policy(request, policy, args.blocks)
        except Exception as error:  # a refused reset or settle failure is logged, not hidden
            counts["_rejected"][type(error).__name__] += 1
            print(f"seed {seed}: rejected ({error})", file=sys.stderr)
            continue
        write_episode(args.out, record)
        mode = counts[scenario.mode]
        mode["episodes"] += 1
        mode[f"split_{split.value}"] += 1
        mode[f"end_{record.blocks[-1].termination_reason}"] += 1
        for block in record.blocks:
            total_blocks += 1
            mode["blocks"] += 1
            if block.transition.events.contacted_object_ids:
                contact_blocks += 1
                mode["contact_blocks"] += 1
            action = block.transition.action
            if action.forward_velocity_mps == 0.0 and action.yaw_rate_rps == 0.0:
                mode["stop_blocks"] += 1
        if sum(c["episodes"] for k, c in counts.items() if not k.startswith("_")) % 20 == 0:
            rate = total_blocks / (time.perf_counter() - started)
            print(f"{total_blocks} blocks, {rate:.1f} blocks/s", file=sys.stderr)

    wall = time.perf_counter() - started
    interaction = contact_blocks / total_blocks if total_blocks else 0.0
    summary = {
        "schema": "go2wm.collection-summary.v1",
        "run_id": run_id,
        "backend": args.backend,
        "controller_id": controller_id,
        "collection_policy": POLICY_ID,
        "mode_weights": args.mode_weights or DEFAULT_MODE_WEIGHTS,
        "split_id": manifest.split_id,
        "blocks_per_episode": args.blocks,
        "total_blocks": total_blocks,
        "interaction_fraction": round(interaction, 4),
        "interaction_target": [0.20, 0.30],
        "interaction_in_target": 0.20 <= interaction <= 0.30,
        "wall_s": round(wall, 1),
        "blocks_per_wall_s": round(total_blocks / wall, 1) if wall else None,
        "by_mode": {k: dict(v) for k, v in counts.items()},
        "evidence_scope": "rehearsal only: fake backend"
        if args.backend == "fake"
        else "collection run",
    }
    summary_path = args.out / f"summary-{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    keys = ("total_blocks", "interaction_fraction", "interaction_in_target", "blocks_per_wall_s")
    print(json.dumps({k: summary[k] for k in keys}))
    print(f"summary: {summary_path}")
    return 0 if math.isfinite(interaction) else 1


if __name__ == "__main__":
    raise SystemExit(main())
