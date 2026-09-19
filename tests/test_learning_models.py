"""Person B model-side tests: readouts, baseline dynamics, evaluation, bundles, LeWM glue."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from _learning_fixtures import push_episode  # noqa: E402

from go2wm.contracts import (  # noqa: E402
    DatasetSplit,
    ObjectState,
    Pose2D,
    RGBObservation,
    StateLabels,
)
from go2wm.learning.baseline_model import (  # noqa: E402
    LinearLatentPredictor,
    PooledPixelEncoder,
    fit_linear_predictor,
)
from go2wm.learning.bundle_io import Provenance, load_bundle, publish_bundle  # noqa: E402
from go2wm.learning.lewm_adapter import (  # noqa: E402
    ActionNormalizer,
    export_sequences_npz,
    lewm_action_context,
    preprocess_rgb,
)
from go2wm.learning.pipeline import run_fake_smoke  # noqa: E402
from go2wm.learning.prediction_eval import (  # noqa: E402
    _shuffle_indices,
    evaluate_predictions,
    g3_checks,
)
from go2wm.learning.readouts import (  # noqa: E402
    LinearLatentReadout,
    TargetSpec,
    fit_linear_readout,
    readout_errors,
)
from go2wm.learning.windows import episode_windows  # noqa: E402
from go2wm.model import ActionBlock, BundleCompatibilityError  # noqa: E402
from go2wm.planning import PlannerSafetyPolicy, ScoringConfig, build_candidate_library  # noqa: E402
from go2wm.runtime import calibrate_surprise  # noqa: E402


def _labels(x: float, y: float, yaw: float, box_x: float, box_y: float) -> StateLabels:
    return StateLabels(
        sim_time_s=0.0,
        robot_pose=Pose2D(x, y, yaw),
        objects=(ObjectState("light", "blue", Pose2D(box_x, box_y), True),),
    )


def test_linear_readout_recovers_a_linear_latent_code() -> None:
    rng = np.random.default_rng(0)
    states = rng.uniform(-1, 1, size=(200, 5))
    labels = [_labels(*row) for row in states]
    spec = TargetSpec(("light",))
    targets = np.stack([spec.encode(item) for item in labels])
    mixing = rng.normal(size=(targets.shape[1], 32))
    latents = targets @ mixing
    readout = fit_linear_readout(latents, labels, alpha=1e-6)
    errors = readout_errors(readout, latents, labels)
    assert errors.robot_position_median_m < 1e-4
    assert errors.object_position_median_m < 1e-4
    decoded = readout.decode(tuple(latents[3]), step=1)
    assert math.isclose(decoded.robot.x_m, states[3, 0], abs_tol=1e-4)
    assert math.isclose(decoded.robot.yaw_rad, states[3, 2], abs_tol=1e-3)
    clone = LinearLatentReadout.from_dict(json.loads(json.dumps(readout.to_dict())))
    assert np.allclose(clone.predict_targets(latents[:5]), readout.predict_targets(latents[:5]))


def test_pooled_encoder_matches_latent_contract() -> None:
    encoder = PooledPixelEncoder()
    observation = push_episode().initial_history[0]
    latent = encoder.encode(observation)
    assert len(latent) == 192
    assert all(0.0 <= value <= 1.0 for value in latent)


def test_linear_predictor_rollout_has_one_latent_per_action_and_round_trips() -> None:
    episode = push_episode()
    encoder = PooledPixelEncoder()
    predictor = fit_linear_predictor(encoder, [episode], alpha=10.0)
    window = episode_windows(episode, horizon=4)[0]
    history = tuple(encoder.encode(obs) for obs in window.model_input.observations)
    rollout = predictor.rollout(history, window.model_input.command_history, window.future_actions)
    assert len(rollout) == 4 and all(len(item) == 192 for item in rollout)
    clone = LinearLatentPredictor.from_dict(json.loads(json.dumps(predictor.to_dict())))
    assert np.allclose(
        clone.rollout(history, window.model_input.command_history, window.future_actions),
        rollout,
    )


def test_shuffled_ablation_never_keeps_a_window_own_commands() -> None:
    for count in (2, 3, 17):
        order = _shuffle_indices(count, seed=4)
        assert sorted(order.tolist()) == list(range(count))
        assert all(order[i] != i for i in range(count))


class _OracleCodec:
    """Encoder/predictor pair whose latent is the true state; used to test the evaluator."""

    def __init__(self, episodes):
        self.by_id = {}
        spec = TargetSpec(("light", "resistant"))
        self.spec = spec
        for episode in episodes:
            for block in episode.blocks:
                t = block.transition
                self.by_id[t.start_observation.observation_id] = spec.encode(t.start_labels)
                self.by_id[t.end_observation.observation_id] = spec.encode(t.end_labels)
            for obs in episode.initial_history:
                self.by_id.setdefault(
                    obs.observation_id, spec.encode(episode.reset_report.final_labels)
                )

    def encode(self, observation: RGBObservation):
        return tuple(self.by_id[observation.observation_id].tolist())


def test_evaluator_reports_perfect_model_and_gate_checks() -> None:
    episode = push_episode()
    codec = _OracleCodec([episode])
    windows = episode_windows(episode, horizon=2)
    truth = {
        w.future_actions: [codec.spec.encode(label) for label in w.future_labels] for w in windows
    }

    class Predictor:
        def rollout(self, history, history_actions, candidate_actions):
            key = tuple(candidate_actions)
            if key in truth:
                return [row.tolist() for row in truth[key]]
            return [np.asarray(history[-1]).tolist()] * len(candidate_actions)

    labels = [w.future_labels[0] for w in windows] + [w.start_labels for w in windows]
    readout = fit_linear_readout(
        np.stack([codec.spec.encode(item) for item in labels]), labels, alpha=1e-9, spec=codec.spec
    )
    report = evaluate_predictions(
        windows, encoder=codec, predictor=Predictor(), readout=readout, horizons=(1, 2)
    )
    one = report["groups"]["all"]["1"]
    assert one["count"] == len(windows)
    assert one["robot_position_median_m"]["model"] < 1e-6
    assert one["robot_position_median_m"]["encoded_real"] < 1e-6
    checks = g3_checks(report, horizon=2)
    assert all(check.passed for check in checks), checks


def _publish(tmp_path: Path, *, threshold_scale: float = 1.0):
    episode = push_episode()
    encoder = PooledPixelEncoder()
    predictor = fit_linear_predictor(encoder, [episode], alpha=10.0)
    pairs = [(t.start_labels, t.start_observation) for t in (b.transition for b in episode.blocks)]
    readout = fit_linear_readout(
        np.stack([encoder.encode_array(obs) for _, obs in pairs]), [lab for lab, _ in pairs]
    )
    window = episode_windows(episode, horizon=6)[0]
    latent = encoder.encode(window.future_observations[0])
    surprise = calibrate_surprise(
        [(latent, tuple(v * threshold_scale + 0.01 for v in latent))],
        bundle_id="pending",
        calibration_version="pending",
    )
    return publish_bundle(
        tmp_path,
        encoder=encoder,
        encoder_kind="pooled_pixels",
        predictor=predictor,
        predictor_kind="linear_latent",
        readout=readout,
        surprise=surprise,
        candidates=build_candidate_library(),
        scoring=ScoringConfig(),
        safety=PlannerSafetyPolicy(),
        provenance=Provenance("rev", "dataset", "splits", "pytest"),
        reference_input=window.model_input,
    )


def test_bundle_publish_reload_and_tamper_detection(tmp_path: Path) -> None:
    bundle = _publish(tmp_path / "a")
    assert bundle.bundle_id.startswith("go2wm-")
    assert bundle.surprise.bundle_id == bundle.bundle_id
    assert len(bundle.candidates) == 64
    reloaded = load_bundle(bundle.path, expected_bundle_id=bundle.bundle_id)
    assert reloaded.bundle_id == bundle.bundle_id
    with pytest.raises(BundleCompatibilityError):
        load_bundle(bundle.path, expected_bundle_id="go2wm-other")

    readout_path = bundle.path / "readout.json"
    payload = json.loads(readout_path.read_text())
    payload["alpha"] = 123.0
    readout_path.write_text(json.dumps(payload))
    with pytest.raises(BundleCompatibilityError, match="checksum"):
        load_bundle(bundle.path)


def test_changing_the_surprise_threshold_changes_bundle_id(tmp_path: Path) -> None:
    first = _publish(tmp_path / "a")
    second = _publish(tmp_path / "b", threshold_scale=1.5)
    assert first.surprise.threshold != second.surprise.threshold
    assert first.bundle_id != second.bundle_id


def test_lewm_action_context_matches_upstream_alignment() -> None:
    history = (ActionBlock(0.0, 0.0), ActionBlock(0.1, 0.0))
    candidate = tuple(ActionBlock(0.2 + 0.01 * i, 0.0) for i in range(6))
    act0, tail = lewm_action_context(history, candidate)
    assert act0 == [history[0], history[1], candidate[0]]
    assert tail == list(candidate[1:])
    with pytest.raises(ValueError):
        lewm_action_context(history[:1], candidate)


def test_lewm_preprocessing_and_action_normalizer() -> None:
    size = 224
    obs = RGBObservation("o", "e", 0.0, "cam", size, size, bytes([255, 0, 128]) * (size * size))
    tensor = preprocess_rgb(obs)
    assert tensor.shape == (3, size, size)
    assert math.isclose(float(tensor[0, 0, 0]), (1.0 - 0.485) / 0.229, rel_tol=1e-5)
    small = RGBObservation("o", "e", 0.0, "cam", 32, 32, bytes(32 * 32 * 3))
    with pytest.raises(ValueError, match="224"):
        preprocess_rgb(small)
    normalizer = ActionNormalizer.fit([ActionBlock(0.0, -1.0), ActionBlock(0.4, 1.0)])
    normalized = normalizer.apply([ActionBlock(0.2, 0.0)])
    assert np.allclose(normalized, 0.0, atol=1e-6)


def test_export_rows_pair_each_frame_with_its_outgoing_command(tmp_path: Path) -> None:
    episode = push_episode()
    out = tmp_path / "train.npz"
    summary = export_sequences_npz([episode], out)
    data = np.load(out)
    frames = len(episode.initial_history) + len(episode.blocks)
    assert summary == {"rows": frames, "episodes": 1}
    assert data["pixels"].shape == (frames, 32, 32, 3)
    assert np.isnan(data["action"][-1]).all()
    block3 = episode.blocks[3].transition.action
    row = len(episode.initial_history) - 1 + 3
    assert np.allclose(data["action"][row], [block3.forward_velocity_mps, block3.yaw_rate_rps])
    assert np.isnan(data["state"][0]).all()
    assert not np.isnan(data["state"][row]).any()
    with pytest.raises(FileExistsError):
        export_sequences_npz([episode], out)


def test_fake_smoke_pipeline_publishes_and_labels_scope(tmp_path: Path) -> None:
    report = run_fake_smoke(tmp_path / "smoke", episode_count=24)
    assert report["evidence_scope"].startswith("software contracts only")
    assert report["leak_check"]["problems"] == ()
    assert Path(report["bundle_path"]).is_dir()
    assert report["planning_smoke"]["selected_candidate"]
    gates = {check["gate"] for check in report["gate_checks"]}
    assert gates == {"G2", "G3"}
    for split in ("train", "validation"):
        assert split in report["coverage"]["splits"]
    with pytest.raises(FileExistsError):
        run_fake_smoke(tmp_path / "smoke", episode_count=24)


def test_split_labels_in_smoke_data_follow_manifest(tmp_path: Path) -> None:
    run_fake_smoke(tmp_path / "s", episode_count=24)
    from go2wm.learning import load_dataset, read_split_manifest

    manifest = read_split_manifest(tmp_path / "s" / "splits.json")
    for episode in load_dataset(tmp_path / "s" / "data").episodes:
        assert episode.split == manifest.split_of(episode.scenario_seed)
    assert DatasetSplit.TEST in {manifest.split_of(s) for s in range(1, 25)}
