"""Real-LeWM tests: run only where torch and a pinned le-wm checkout are available.

Set ``GO2WM_LEWM_REPO`` to the le-wm checkout (commit 8edfeb3...) inside the
LeWM Python environment.  In the dependency-light environment these skip.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
if not os.environ.get("GO2WM_LEWM_REPO"):
    pytest.skip("GO2WM_LEWM_REPO is not set", allow_module_level=True)

import numpy as np  # noqa: E402
from _learning_fixtures import push_episode  # noqa: E402

from go2wm.contracts import ActionCommand, CameraConfig, DatasetSplit  # noqa: E402
from go2wm.data import CollectionConfig, EpisodeCollector, EpisodeRequest  # noqa: E402
from go2wm.learning import lewm_model as lm  # noqa: E402
from go2wm.learning.lewm_adapter import LeWMWorldModel  # noqa: E402
from go2wm.learning.lewm_cache import build_training_cache  # noqa: E402
from go2wm.learning.windows import episode_windows  # noqa: E402
from go2wm.model import ActionBlock  # noqa: E402
from go2wm.sim import DeterministicFakeSimulator, FakeSimulatorConfig  # noqa: E402

REPO = Path(os.environ["GO2WM_LEWM_REPO"])


@pytest.fixture(scope="module")
def src() -> lm.LeWMSource:
    return lm.import_lewm(REPO)


@pytest.fixture(scope="module")
def trained(src: lm.LeWMSource) -> lm.TrainedLeWM:
    torch.manual_seed(0)
    arch = lm.LeWMArchitecture()
    model = lm.build_model(arch, src).eval()
    return lm.TrainedLeWM(
        model=model,
        arch=arch,
        action_stats=lm.ActionStats((0.2, 0.0), (0.2, 0.5)),
        weights_path=Path("unsaved"),
        weights_sha256="0" * 64,
        run_config={"lewm": {"commit": src.commit}},
    )


def episode_224(episode_id: str, seed: int, split: DatasetSplit, blocks: int = 8):
    sim = DeterministicFakeSimulator(FakeSimulatorConfig(camera=CameraConfig("overhead", 224, 224)))
    base = push_episode(episode_id, seed=seed, split=split)
    commands = [b.transition.action for b in base.blocks][:blocks]
    while len(commands) < blocks:
        commands.append(ActionCommand(0.3, 0.2))
    return EpisodeCollector(sim, CollectionConfig()).collect(
        EpisodeRequest(
            reset=_reset_of(base),
            split=split,
            goal=base.goal,
            run_id="torch-test",
        ),
        commands,
    )


def _reset_of(episode):
    from go2wm.contracts import ResetRequest

    labels = episode.reset_report.final_labels
    return ResetRequest(
        episode_id=episode.episode_id,
        scenario_seed=episode.scenario_seed,
        robot_pose=labels.robot_pose,
        objects=labels.objects,
    )


def test_architecture_matches_upstream_config(src: lm.LeWMSource) -> None:
    model = lm.build_model(lm.LeWMArchitecture(), src)
    assert 17.5e6 < lm.count_parameters(model) < 18.5e6
    assert model.action_encoder.patch_embed.in_channels == 2


def test_loss_matches_upstream_lejepa_forward(src: lm.LeWMSource, trained) -> None:
    sys.path.insert(0, str(REPO))
    try:
        upstream = __import__("train")
    except Exception as error:  # upstream train.py imports hydra/lightning/swm
        pytest.skip(f"upstream train.py not importable here: {error}")
    from types import SimpleNamespace as NS

    arch, cfg = trained.arch, lm.LossConfig()
    sigreg = lm.build_sigreg(cfg, src)
    pixels = torch.randint(0, 256, (2, 4, 3, 224, 224), dtype=torch.uint8)
    actions = torch.randn(2, 4, 2)
    actions[:, -1] = float("nan")
    stats = trained.action_stats
    torch.manual_seed(5)
    ours = lm.lejepa_losses(
        trained.model,
        sigreg,
        lm.preprocess_pixels(pixels),
        lm.normalize_actions(actions, stats),
        arch,
        cfg,
    )
    batch = {
        "pixels": lm.preprocess_pixels(pixels),
        "action": (actions - torch.tensor(stats.mean)) / torch.tensor(stats.std),
    }
    upstream_cfg = NS(history_size=3, num_preds=1, loss=NS(sigreg=NS(weight=cfg.sigreg_weight)))
    fake_module = NS(model=trained.model, sigreg=sigreg, log_dict=lambda *a, **k: None)
    torch.manual_seed(5)
    theirs = upstream.lejepa_forward(fake_module, batch, "test", upstream_cfg)
    for key in ("pred_loss", "sigreg_loss", "loss"):
        assert torch.allclose(ours[key], theirs[key], rtol=1e-5, atol=1e-6), key


def test_action_encoder_receives_gradient(src: lm.LeWMSource) -> None:
    """Actions enter only through AdaLN-zero modulation, which is zero at init.

    So the action encoder gets no gradient on the very first step (by upstream
    design) and must get one as soon as the modulation weights move.
    """

    torch.manual_seed(0)
    arch, cfg = lm.LeWMArchitecture(), lm.LossConfig()
    model = lm.build_model(arch, src).train()
    sigreg = lm.build_sigreg(cfg, src)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pixels = lm.preprocess_pixels(torch.randint(0, 256, (4, 4, 3, 224, 224), dtype=torch.uint8))
    actions = torch.randn(4, 4, 2)

    def action_grad() -> float:
        optimizer.zero_grad()
        lm.lejepa_losses(model, sigreg, pixels, actions, arch, cfg)["loss"].backward()
        grads = [p.grad for p in model.action_encoder.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        return sum(float(g.abs().sum()) for g in grads)

    assert action_grad() == 0.0
    optimizer.step()
    assert action_grad() > 0.0


def test_auxiliary_state_loss_only_uses_aligned_rows() -> None:
    from go2wm.learning.lewm_train import auxiliary_state_loss, build_auxiliary_head

    torch.manual_seed(9)
    embeddings = torch.randn(2, 4, 192, requires_grad=True)
    head = build_auxiliary_head(192, 9, 16)
    valid = torch.tensor([[True, False, True, True], [False, True, True, False]])
    batch = {
        "state": torch.randn(2, 4, 9),
        "state_valid": valid,
    }
    loss = auxiliary_state_loss(head, embeddings, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.count_nonzero(embeddings.grad[valid]) > 0
    assert torch.count_nonzero(embeddings.grad[~valid]) == 0


def test_rollout_matches_upstream_jepa_rollout(trained) -> None:
    world = LeWMWorldModel(trained)
    episode = episode_224("ep-roll", 11, DatasetSplit.TRAIN)
    window = episode_windows(episode, horizon=6)[0]
    history = tuple(world.encode(o) for o in window.model_input.observations)
    candidates = [
        window.future_actions,
        tuple(ActionBlock(0.1 * i, -0.3) for i in range(6)),
    ]
    ours = np.asarray(world.rollout_batch(history, window.model_input.command_history, candidates))

    pixels = torch.stack(
        [
            torch.from_numpy(
                np.frombuffer(o.rgb, dtype=np.uint8).reshape(224, 224, 3).copy()
            ).permute(2, 0, 1)
            for o in window.model_input.observations
        ]
    )
    info = {"pixels": lm.preprocess_pixels(pixels)[None, None].expand(1, 2, -1, -1, -1, -1)}
    sequences = [[*window.model_input.command_history, *candidate] for candidate in candidates]
    raw = torch.tensor(
        [[[a.forward_mps, a.yaw_rate_rps] for a in seq] for seq in sequences],
        dtype=torch.float32,
    )[None]
    normalized = lm.normalize_actions(raw, trained.action_stats)
    with torch.no_grad():
        out = trained.model.rollout(info, normalized, history_size=3)
    theirs = out["predicted_emb"][0, :, 3:].numpy()
    assert theirs.shape == ours.shape == (2, 6, 192)
    assert np.allclose(ours, theirs, atol=1e-4, rtol=1e-4)


def test_batched_and_single_paths_agree(trained) -> None:
    world = LeWMWorldModel(trained)
    episode = episode_224("ep-batch", 12, DatasetSplit.TRAIN)
    observations = episode.initial_history
    batch = world.encode_batch(observations)
    single = np.asarray([world.encode(o) for o in observations])
    assert np.allclose(batch, single, atol=1e-5)
    window = episode_windows(episode, horizon=6)[0]
    history = tuple(world.encode(o) for o in window.model_input.observations)
    cands = [window.future_actions, tuple(ActionBlock(0.4, 0.1) for _ in range(6))]
    together = world.rollout_batch(history, window.model_input.command_history, cands)
    alone = [world.rollout(history, window.model_input.command_history, c) for c in cands]
    assert np.allclose(np.asarray(together), np.asarray(alone), atol=1e-5)


def test_train_resume_eval_bundle_end_to_end(tmp_path: Path, src) -> None:
    """Cache -> train -> resume -> lewm-eval -> bundle reload, on tiny fake data."""

    from go2wm.data import write_episode
    from go2wm.learning import lewm_train
    from go2wm.learning.bundle_io import load_bundle
    from go2wm.learning.pipeline import run_lewm_pipeline
    from go2wm.learning.splits import build_split_manifest

    train_eps = [episode_224(f"tr-{i}", i, DatasetSplit.TRAIN) for i in range(1, 4)]
    val_eps = [episode_224(f"va-{i}", 10 + i, DatasetSplit.VALIDATION) for i in range(1, 3)]
    manifest = build_split_manifest([1, 2, 3], salt="x")
    assignments = {
        **{e.scenario_seed: DatasetSplit.TRAIN for e in train_eps},
        **{e.scenario_seed: DatasetSplit.VALIDATION for e in val_eps},
    }
    manifest = type(manifest)(
        split_id="splits-torch-test",
        salt="x",
        fractions=manifest.fractions,
        assignments=assignments,
    )
    for episode in (*train_eps, *val_eps):
        write_episode(tmp_path / "data", episode)
    build_training_cache(
        train_eps,
        tmp_path / "cache-train",
        split=DatasetSplit.TRAIN,
        dataset_id="torch-test",
        split_id=manifest.split_id,
    )
    build_training_cache(
        val_eps,
        tmp_path / "cache-validation",
        split=DatasetSplit.VALIDATION,
        dataset_id="torch-test",
        split_id=manifest.split_id,
    )
    run = tmp_path / "run"
    common = [
        "--lewm-repo",
        str(REPO),
        "--train-cache",
        str(tmp_path / "cache-train"),
        "--val-cache",
        str(tmp_path / "cache-validation"),
        "--out",
        str(run),
        "--batch-size",
        "4",
        "--num-workers",
        "0",
        "--device",
        "cpu",
        "--val-max-batches",
        "1",
        "--log-every",
        "1",
        "--aux-state-weight",
        "0.1",
    ]
    assert lewm_train.main([*common, "--epochs", "1", "--max-steps", "2"]) == 0
    index = json.loads((run / "checkpoints" / "index.json").read_text())
    assert index["best"] == "epoch_001.pt"
    assert lewm_train.main([*common, "--epochs", "2", "--resume"]) == 0
    index = json.loads((run / "checkpoints" / "index.json").read_text())
    assert set(index["checkpoints"]) == {"epoch_001.pt", "epoch_002.pt"}
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    epochs = [m for m in metrics if m["event"] == "epoch"]
    assert [m["epoch"] for m in epochs] == [1, 2]
    assert all(np.isfinite(m["val_pred_loss"]) for m in epochs)
    assert all(np.isfinite(m["val_aux_state_loss"]) for m in epochs)

    reloaded = lm.load_trained(run, checkpoint="last", src=src)
    assert reloaded.weights_sha256 == index["checkpoints"]["epoch_002.pt"]["sha256"]

    from go2wm.learning.dataset import load_dataset

    report = run_lewm_pipeline(
        load_dataset(tmp_path / "data").episodes,
        manifest,
        tmp_path / "eval",
        run_dir=run,
        checkpoint="last",
        lewm_repo=REPO,
        device="cpu",
        dataset_id="torch-test",
        data_backend="fake",
    )
    assert report["encoder_kind"] == "lewm"
    assert "REHEARSAL ONLY" in report["model"]
    run_config = json.loads((run / "run_config.json").read_text())
    assert run_config["rehearsal_only"] is True
    assert run_config["auxiliary"]["enabled"] is True
    assert run_config["auxiliary"]["runtime_inputs_unchanged"] is True
    assert report["gate_checks"]
    bundle = load_bundle(
        report["bundle_path"],
        expected_bundle_id=report["bundle_id"],
        context={"lewm_repo": REPO},
    )
    assert (bundle.path / "lewm_weights.pt").is_file()
    assert bundle.model_bundle.manifest.latent_dim == 192


def test_missing_caches_fail_loudly(tmp_path: Path) -> None:
    from go2wm.learning import lewm_train
    from go2wm.learning.lewm_cache import CacheError

    with pytest.raises(CacheError):
        lewm_train.main(
            [
                "--lewm-repo",
                str(REPO),
                "--train-cache",
                str(tmp_path / "missing"),
                "--val-cache",
                str(tmp_path / "missing"),
                "--out",
                str(tmp_path / "r"),
            ]
        )
