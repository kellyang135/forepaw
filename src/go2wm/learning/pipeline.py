"""End-to-end ML pipeline: data checks -> fit -> G2/G3 checks -> calibrate -> publish -> plan.

``run_baseline_pipeline`` works on any validated dataset, so the same code path
runs on fake-simulator smoke data today and on MuJoCo data after G1.  Its
report states which backend produced the data; fake-backend numbers are
software-contract evidence only.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from go2wm.contracts import (
    ActionCommand,
    DatasetSplit,
    EpisodeRecord,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
)
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest, write_episode
from go2wm.planning import (
    PlannerSafetyPolicy,
    PlanningRequest,
    RecedingHorizonPlanner,
    RolloutScorer,
    ScoringConfig,
    build_candidate_library,
)
from go2wm.runtime import calibrate_surprise
from go2wm.sim import DeterministicFakeSimulator

from .baseline_model import PooledPixelEncoder, fit_linear_predictor
from .bundle_io import Provenance, publish_bundle
from .dataset import load_dataset
from .diagnostics import coverage_report
from .prediction_eval import evaluate_predictions, g2_checks, g3_checks
from .readouts import constant_mean_errors, fit_linear_readout, readout_errors
from .splits import SplitManifest, build_split_manifest, check_leakage, write_split_manifest
from .surprise_fit import false_stop_fraction, one_step_pairs
from .windows import dataset_windows, labeled_observations

ROOT = Path(__file__).resolve().parents[3]


def git_revision(root: Path = ROOT) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    return process.stdout.strip() if process.returncode == 0 else "NO_COMMIT"


def run_baseline_pipeline(
    episodes: Sequence[EpisodeRecord],
    split_manifest: SplitManifest,
    output_dir: str | Path,
    *,
    dataset_id: str,
    data_backend: str,
    readout_alpha: float = 1.0,
    predictor_alpha: float = 100.0,
    surprise_quantile: float = 0.99,
    min_shuffle_improvement: float = 0.15,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    leak = check_leakage(episodes, split_manifest)
    if not leak.ok:
        raise ValueError("split leak check failed: " + "; ".join(leak.problems))
    train = [e for e in episodes if e.split == DatasetSplit.TRAIN]
    validation = [e for e in episodes if e.split == DatasetSplit.VALIDATION]
    if not train or not validation:
        raise ValueError("pipeline needs both train and validation episodes")
    # Test episodes are deliberately never touched here.

    encoder = PooledPixelEncoder()
    predictor = fit_linear_predictor(encoder, train, alpha=predictor_alpha)

    train_pairs = labeled_observations(train)
    val_pairs = labeled_observations(validation)
    train_latents = np.stack([encoder.encode_array(obs) for obs, _ in train_pairs])
    val_latents = np.stack([encoder.encode_array(obs) for obs, _ in val_pairs])
    readout = fit_linear_readout(
        train_latents, [labels for _, labels in train_pairs], alpha=readout_alpha
    )
    real_errors = readout_errors(readout, val_latents, [labels for _, labels in val_pairs])
    mean_errors = constant_mean_errors(
        [labels for _, labels in train_pairs], [labels for _, labels in val_pairs], readout.spec
    )

    val_windows = dataset_windows(validation, horizon=6)
    prediction_report = evaluate_predictions(
        val_windows, encoder=encoder, predictor=predictor, readout=readout
    )

    calibration_windows = dataset_windows(validation, horizon=1)
    pairs = one_step_pairs(calibration_windows, encoder=encoder, predictor=predictor)
    surprise = calibrate_surprise(
        pairs,
        bundle_id="pending",
        calibration_version="pending",
        quantile=surprise_quantile,
    )

    candidates = build_candidate_library()
    scoring = ScoringConfig()
    safety = PlannerSafetyPolicy()
    reference_input = val_windows[0].model_input
    loaded = publish_bundle(
        out / "bundles",
        encoder=encoder,
        encoder_kind="pooled_pixels",
        predictor=predictor,
        predictor_kind="linear_latent",
        readout=readout,
        surprise=surprise,
        candidates=candidates,
        scoring=scoring,
        safety=safety,
        provenance=Provenance(
            source_revision=git_revision(),
            dataset_id=dataset_id,
            split_id=split_manifest.split_id,
            training_command=f"go2wm.learning baseline ({data_backend})",
            notes="linear latent baseline; not LeWM",
        ),
        reference_input=reference_input,
    )
    in_sample_false_stops = false_stop_fraction(
        calibration_windows, loaded.surprise, encoder=encoder, predictor=predictor
    )

    # One planning cycle through the published bundle, from runtime-visible inputs only.
    window = val_windows[0]
    current_latent = encoder.encode(window.model_input.observations[-1])
    current = readout.decode(current_latent, step=1).robot
    planner = RecedingHorizonPlanner(
        loaded.model_bundle,
        candidates=loaded.candidates,
        scorer=RolloutScorer(loaded.scoring),
        safety_policy=loaded.safety,
    )
    goal = next(e.goal for e in validation if e.episode_id == window.episode_id)
    plan = planner.plan(
        PlanningRequest(
            model_input=window.model_input,
            goal_xy=(goal.x_m, goal.y_m),
            current_robot_xy=current.xy,
        )
    )

    checks = [
        *g2_checks(real_errors, mean_errors, leak_ok=leak.ok, reload_deterministic=True),
        *g3_checks(prediction_report, min_shuffle_improvement=min_shuffle_improvement),
    ]
    report = {
        "evidence_scope": (
            "software contracts only; fake simulator data"
            if data_backend == "fake"
            else f"data backend: {data_backend}; linear baseline, not LeWM"
        ),
        "data_backend": data_backend,
        "dataset_id": dataset_id,
        "split_id": split_manifest.split_id,
        "bundle_id": loaded.bundle_id,
        "bundle_path": str(loaded.path),
        "leak_check": asdict(leak),
        "coverage": coverage_report(episodes),
        "readout_validation_real_latents": asdict(real_errors),
        "readout_validation_constant_mean": asdict(mean_errors),
        "prediction": prediction_report,
        "surprise": {
            "threshold": loaded.surprise.threshold,
            "quantile": loaded.surprise.quantile,
            "calibration_samples": loaded.surprise.normal_sample_count,
            "in_sample_false_stop_fraction": in_sample_false_stops,
            "note": "in-sample on calibration windows; held-out false stops are measured at G6/G7",
        },
        "planning_smoke": {
            "selected_candidate": plan.selected.rollout.candidate_id,
            "selected_family": plan.selected.rollout.family,
            "selection_reason": plan.selection_reason,
            "first_action": [plan.first_action.forward_mps, plan.first_action.yaw_rate_rps],
            "top5": [
                [item.rollout.candidate_id, round(item.score.total, 4)]
                for item in plan.rankings[:5]
            ],
        },
        "gate_checks": [asdict(check) for check in checks],
        "all_checks_passed": all(check.passed for check in checks),
    }
    report_path = out / "ml_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def fake_episode_plan(
    seed: int, *, blocks: int = 16
) -> tuple[ResetRequest, Goal2D, list[ActionCommand], str]:
    """Deterministic scripted scenario for the fake backend (push, blocked, or free)."""

    rng = random.Random(seed)
    kind = rng.choice(("push_light", "push_resistant", "free", "free", "free"))
    light = Pose2D(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6))
    resistant = Pose2D(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6))
    while math.hypot(light.x_m - resistant.x_m, light.y_m - resistant.y_m) < 0.6:
        resistant = Pose2D(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8))
    target = light if kind == "push_light" else resistant if kind == "push_resistant" else None
    if target is not None:
        bearing = rng.uniform(-math.pi, math.pi)
        distance = rng.uniform(0.55, 0.8)
        start = Pose2D(
            target.x_m - distance * math.cos(bearing),
            target.y_m - distance * math.sin(bearing),
            bearing + rng.uniform(-0.08, 0.08),
        )
    else:
        start = Pose2D(
            rng.uniform(-1.1, 1.1), rng.uniform(-1.1, 1.1), rng.uniform(-math.pi, math.pi)
        )
    actions: list[ActionCommand] = []
    while len(actions) < blocks:
        mode = rng.random()
        length = rng.randint(1, 4)
        if target is not None and len(actions) < 5:
            command = ActionCommand(rng.uniform(0.25, 0.5), rng.uniform(-0.1, 0.1))
        elif target is not None and len(actions) < 8:
            command = ActionCommand(0.05, rng.choice((-1, 1)) * rng.uniform(0.9, 1.2))
        elif mode < 0.2:
            command = ActionCommand.stopped()
        elif mode < 0.45:
            command = ActionCommand(
                rng.uniform(0.0, 0.2), rng.choice((-1, 1)) * rng.uniform(0.4, 1.1)
            )
        else:
            command = ActionCommand(rng.uniform(0.15, 0.55), rng.uniform(-0.3, 0.3))
        actions.extend([command] * length)
    goal = Goal2D(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.15)
    reset = ResetRequest(
        episode_id=f"fake-{seed:05d}",
        scenario_seed=seed,
        robot_pose=start,
        objects=(
            ObjectState("light", "blue", light, True),
            ObjectState("resistant", "red", resistant, False),
        ),
    )
    return reset, goal, actions[:blocks], kind


def collect_fake_dataset(
    data_dir: str | Path, split_manifest: SplitManifest, seeds: Sequence[int], *, blocks: int = 16
) -> None:
    simulator = DeterministicFakeSimulator()
    collector = EpisodeCollector(simulator, CollectionConfig())
    for seed in seeds:
        reset, goal, actions, kind = fake_episode_plan(seed, blocks=blocks)
        record = collector.collect(
            EpisodeRequest(
                reset=reset,
                split=split_manifest.split_of(seed),
                goal=goal,
                run_id="synthetic-learning-pipeline",
                metadata={
                    "purpose": "ml-software-smoke-only",
                    "scripted_scenario": kind,
                    "termination_reason": "fixed_length",
                },
            ),
            actions,
        )
        write_episode(data_dir, record)


def run_fake_smoke(output_dir: str | Path, *, episode_count: int = 60) -> dict[str, Any]:
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; smoke runs never overwrite evidence")
    seeds = list(range(1, episode_count + 1))
    manifest = build_split_manifest(seeds, salt="fake-smoke-v1", notes="fake backend only")
    write_split_manifest(out / "splits.json", manifest)
    collect_fake_dataset(out / "data", manifest, seeds)
    dataset = load_dataset(out / "data")
    return run_baseline_pipeline(
        dataset.episodes,
        manifest,
        out,
        dataset_id=f"fake-smoke-{episode_count}",
        data_backend="fake",
    )
