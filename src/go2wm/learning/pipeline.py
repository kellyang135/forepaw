"""End-to-end ML pipeline: data checks -> fit -> G2/G3 checks -> calibrate -> publish -> plan.

``run_baseline_pipeline`` works on any validated dataset, so the same code path
runs on fake-simulator smoke data today and on MuJoCo data after G1.  Its
report states which backend produced the data; fake-backend numbers are
software-contract evidence only.
"""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from go2wm.contracts import (
    ActionCommand,
    CameraConfig,
    DatasetSplit,
    EpisodeRecord,
    Goal2D,
    ObjectState,
    Pose2D,
    ResetRequest,
    RGBObservation,
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
from go2wm.sim import DeterministicFakeSimulator, FakeSimulatorConfig

from .baseline_model import PooledPixelEncoder, fit_linear_predictor, select_predictor_alpha
from .bundle_io import Provenance, publish_bundle
from .dataset import load_dataset
from .diagnostics import coverage_report
from .prediction_eval import evaluate_predictions, g2_checks, g3_checks
from .readouts import (
    constant_mean_errors,
    fit_linear_readout,
    readout_errors,
    select_readout_alpha,
)
from .splits import SplitManifest, build_split_manifest, check_leakage, write_split_manifest
from .surprise_fit import false_stop_fraction, one_step_pairs
from .windows import dataset_windows, episode_sequence, labeled_observations

ROOT = Path(__file__).resolve().parents[3]


def git_revision(root: Path = ROOT) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    return process.stdout.strip() if process.returncode == 0 else "NO_COMMIT"


class MemoEncoder:
    """Encode each observation once.

    Evaluation windows overlap, so the same frame is requested up to nine
    times.  ``prime`` batch-encodes when the wrapped encoder has
    ``encode_batch`` (LeWM), which is much faster on a GPU/MPS device.
    """

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self._memo: dict[tuple[str, str], tuple[float, ...]] = {}

    def prime(self, observations: Sequence[RGBObservation], batch_size: int = 64) -> None:
        todo = [o for o in observations if (o.episode_id, o.observation_id) not in self._memo]
        unique = list({(o.episode_id, o.observation_id): o for o in todo}.values())
        if not unique:
            return
        batch = getattr(self.encoder, "encode_batch", None)
        if batch is not None:
            latents = batch(unique, batch_size=batch_size)
        else:
            latents = [self.encoder.encode(o) for o in unique]
        for observation, latent in zip(unique, latents, strict=True):
            self._memo[(observation.episode_id, observation.observation_id)] = tuple(
                float(v) for v in latent
            )

    def encode(self, observation: RGBObservation) -> tuple[float, ...]:
        key = (observation.episode_id, observation.observation_id)
        if key not in self._memo:
            self.prime([observation])
        return self._memo[key]

    def encode_array(self, observations: Sequence[RGBObservation]) -> np.ndarray:
        self.prime(observations)
        return np.asarray([self.encode(o) for o in observations], dtype=np.float64)


def _all_observations(episodes: Sequence[EpisodeRecord]) -> list[RGBObservation]:
    frames: list[RGBObservation] = []
    for episode in episodes:
        frames.extend(episode_sequence(episode)[0])
    return frames


def split_episodes(
    episodes: Sequence[EpisodeRecord], split_manifest: SplitManifest
) -> tuple[list[EpisodeRecord], list[EpisodeRecord], Any]:
    leak = check_leakage(episodes, split_manifest)
    if not leak.ok:
        raise ValueError("split leak check failed: " + "; ".join(leak.problems))
    train = [e for e in episodes if e.split == DatasetSplit.TRAIN]
    validation = [e for e in episodes if e.split == DatasetSplit.VALIDATION]
    if not train or not validation:
        raise ValueError("pipeline needs both train and validation episodes")
    return train, validation, leak


def evaluate_and_publish(
    episodes: Sequence[EpisodeRecord],
    split_manifest: SplitManifest,
    output_dir: str | Path,
    *,
    encoder: Any,
    encoder_kind: str,
    predictor: Any,
    predictor_kind: str,
    model_label: str,
    dataset_id: str,
    data_backend: str,
    training_command: str,
    readout_alpha: float | str = "cv",
    surprise_quantile: float = 0.99,
    min_shuffle_improvement: float = 0.15,
    extra_files: dict[str, Path] | None = None,
    reference_tolerance: float = 1e-6,
    load_context: dict[str, Any] | None = None,
    latency_repeats: int = 5,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """G2/G3 checks, surprise calibration, bundle publication, and one timed plan.

    Only train and validation episodes are read.  Test episodes are never
    touched, whatever the manifest contains.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, validation, leak = split_episodes(episodes, split_manifest)
    memo = MemoEncoder(encoder)
    started = time.perf_counter()
    memo.prime(_all_observations(train) + _all_observations(validation))
    encode_seconds = time.perf_counter() - started

    train_pairs = labeled_observations(train)
    val_pairs = labeled_observations(validation)
    train_latents = memo.encode_array([obs for obs, _ in train_pairs])
    val_latents = memo.encode_array([obs for obs, _ in val_pairs])
    train_labels = [labels for _, labels in train_pairs]
    readout_cv: dict[str, float] = {}
    if readout_alpha == "cv":
        chosen_alpha, readout_cv = select_readout_alpha(
            train_latents, train_labels, [obs.episode_id for obs, _ in train_pairs]
        )
    else:
        chosen_alpha = float(readout_alpha)
    readout = fit_linear_readout(train_latents, train_labels, alpha=chosen_alpha)
    real_errors = readout_errors(readout, val_latents, [labels for _, labels in val_pairs])
    train_fit_errors = readout_errors(readout, train_latents, [labels for _, labels in train_pairs])
    mean_errors = constant_mean_errors(
        [labels for _, labels in train_pairs], [labels for _, labels in val_pairs], readout.spec
    )

    val_windows = dataset_windows(validation, horizon=6)
    if len(val_windows) < 2:
        raise ValueError("validation needs at least two 6-block windows")
    prediction_report = evaluate_predictions(
        val_windows, encoder=memo, predictor=predictor, readout=readout
    )

    calibration_windows = dataset_windows(validation, horizon=1)
    pairs = one_step_pairs(calibration_windows, encoder=memo, predictor=predictor)
    surprise = calibrate_surprise(
        pairs,
        bundle_id="pending",
        calibration_version="pending",
        quantile=surprise_quantile,
    )

    candidates = build_candidate_library()
    loaded = publish_bundle(
        out / "bundles",
        encoder=encoder,
        encoder_kind=encoder_kind,
        predictor=predictor,
        predictor_kind=predictor_kind,
        readout=readout,
        surprise=surprise,
        candidates=candidates,
        scoring=ScoringConfig(),
        safety=PlannerSafetyPolicy(),
        provenance=Provenance(
            source_revision=git_revision(),
            dataset_id=dataset_id,
            split_id=split_manifest.split_id,
            training_command=training_command,
            notes=model_label,
        ),
        reference_input=val_windows[0].model_input,
        latent_dim=len(train_latents[0]),
        extra_files=extra_files,
        reference_tolerance=reference_tolerance,
        load_context=load_context,
    )
    in_sample_false_stops = false_stop_fraction(
        calibration_windows, loaded.surprise, encoder=memo, predictor=predictor
    )

    # Plan through the reloaded bundle from runtime-visible inputs only, and time it.
    window = val_windows[0]
    planner = RecedingHorizonPlanner(
        loaded.model_bundle,
        candidates=loaded.candidates,
        scorer=RolloutScorer(loaded.scoring),
        safety_policy=loaded.safety,
    )
    goal = next(e.goal for e in validation if e.episode_id == window.episode_id)
    backend = loaded.model_bundle.backend
    timings: list[float] = []
    plan = None
    for _ in range(max(1, latency_repeats)):
        started = time.perf_counter()
        current_latent = backend.encoder.encode(window.model_input.observations[-1])
        current = loaded.readout.decode(current_latent, step=1).robot
        plan = planner.plan(
            PlanningRequest(
                model_input=window.model_input,
                goal_xy=(goal.x_m, goal.y_m),
                current_robot_xy=current.xy,
            )
        )
        timings.append(time.perf_counter() - started)
    assert plan is not None
    ordered = sorted(timings)

    checks = [
        *g2_checks(real_errors, mean_errors, leak_ok=leak.ok, reload_deterministic=True),
        *g3_checks(prediction_report, min_shuffle_improvement=min_shuffle_improvement),
    ]
    report = {
        "evidence_scope": (
            f"software contracts only; fake simulator data; model: {model_label}"
            if data_backend == "fake"
            else f"data backend: {data_backend}; model: {model_label}"
        ),
        "model": model_label,
        "encoder_kind": encoder_kind,
        "predictor_kind": predictor_kind,
        "data_backend": data_backend,
        "dataset_id": dataset_id,
        "split_id": split_manifest.split_id,
        "bundle_id": loaded.bundle_id,
        "bundle_path": str(loaded.path),
        "leak_check": asdict(leak),
        "coverage": coverage_report(episodes),
        "readout_alpha": {"chosen": chosen_alpha, "grouped_cv_scores_m": readout_cv},
        "readout_validation_real_latents": asdict(real_errors),
        "readout_train_fit": asdict(train_fit_errors),
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
        "latency": {
            "scope": "in-process encode + 64-candidate plan through the reloaded bundle; "
            "excludes transport and simulator",
            "repeats": len(timings),
            "median_s": ordered[len(ordered) // 2],
            "max_s": ordered[-1],
            "first_call_s": timings[0],
            "dataset_encode_s": encode_seconds,
            "platform": platform.platform(),
        },
        "gate_checks": [asdict(check) for check in checks],
        "all_checks_passed": all(check.passed for check in checks),
        **(extra_report or {}),
    }
    report_path = out / "ml_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def run_baseline_pipeline(
    episodes: Sequence[EpisodeRecord],
    split_manifest: SplitManifest,
    output_dir: str | Path,
    *,
    dataset_id: str,
    data_backend: str,
    readout_alpha: float | str = "cv",
    predictor_alpha: float | str = "cv",
    surprise_quantile: float = 0.99,
    min_shuffle_improvement: float = 0.15,
) -> dict[str, Any]:
    train, _, _ = split_episodes(episodes, split_manifest)
    encoder = PooledPixelEncoder()
    predictor_cv: dict[str, float] = {}
    if predictor_alpha == "cv":
        chosen, predictor_cv = select_predictor_alpha(encoder, train)
    else:
        chosen = float(predictor_alpha)
    predictor = fit_linear_predictor(encoder, train, alpha=chosen)
    return evaluate_and_publish(
        episodes,
        split_manifest,
        output_dir,
        encoder=encoder,
        encoder_kind="pooled_pixels",
        predictor=predictor,
        predictor_kind="linear_latent",
        model_label="linear latent baseline (pooled pixels + ridge dynamics); not LeWM",
        dataset_id=dataset_id,
        data_backend=data_backend,
        training_command=f"go2wm.learning baseline ({data_backend})",
        readout_alpha=readout_alpha,
        surprise_quantile=surprise_quantile,
        min_shuffle_improvement=min_shuffle_improvement,
        extra_report={
            "predictor_alpha": {"chosen": chosen, "grouped_cv_rollout_rmse": predictor_cv}
        },
    )


def run_lewm_pipeline(
    episodes: Sequence[EpisodeRecord],
    split_manifest: SplitManifest,
    output_dir: str | Path,
    *,
    run_dir: str | Path,
    checkpoint: str = "best",
    lewm_repo: str | Path | None = None,
    device: str = "cpu",
    dataset_id: str,
    data_backend: str,
    readout_alpha: float | str = "cv",
    surprise_quantile: float = 0.99,
    min_shuffle_improvement: float = 0.15,
    reference_tolerance: float = 2e-3,
) -> dict[str, Any]:
    """Evaluate a trained LeWM run with exactly the same code path as the baseline."""

    from .lewm_adapter import WEIGHTS_FILENAME, LeWMWorldModel

    world_model = LeWMWorldModel.from_run(
        run_dir, checkpoint=checkpoint, lewm_repo=lewm_repo, device=device
    )
    run_config = world_model.trained.run_config
    if run_config.get("data", {}).get("split_id") not in (None, split_manifest.split_id):
        raise ValueError(
            f"run was trained on split {run_config['data']['split_id']}, "
            f"evaluating with {split_manifest.split_id}"
        )
    context: dict[str, Any] = {"lewm_repo": lewm_repo, "device": device}
    rehearsal = bool(run_config.get("rehearsal_only")) or data_backend == "fake"
    label = f"LeWM run {run_config.get('run_id', run_dir)} checkpoint {checkpoint}"
    if rehearsal:
        label += " [REHEARSAL ONLY: fake data; not for handoff]"
    return evaluate_and_publish(
        episodes,
        split_manifest,
        output_dir,
        encoder=world_model,
        encoder_kind="lewm",
        predictor=world_model,
        predictor_kind="lewm",
        model_label=label,
        dataset_id=dataset_id,
        data_backend=data_backend,
        training_command=" ".join(run_config.get("command", [])),
        readout_alpha=readout_alpha,
        surprise_quantile=surprise_quantile,
        min_shuffle_improvement=min_shuffle_improvement,
        extra_files={WEIGHTS_FILENAME: world_model.trained.weights_path},
        reference_tolerance=reference_tolerance,
        load_context=context,
        extra_report={
            "lewm_run": {
                "run_dir": str(run_dir),
                "checkpoint": checkpoint,
                "weights_sha256": world_model.trained.weights_sha256,
                "device": device,
                "best": run_config.get("best"),
                "rehearsal_only": rehearsal,
            }
        },
    )


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
    data_dir: str | Path,
    split_manifest: SplitManifest,
    seeds: Sequence[int],
    *,
    blocks: int = 16,
    image_size: int = 32,
) -> None:
    """Scripted fake-simulator episodes; ``image_size=224`` matches the LeWM camera contract."""

    simulator = DeterministicFakeSimulator(
        FakeSimulatorConfig(camera=CameraConfig("overhead", image_size, image_size))
    )
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
